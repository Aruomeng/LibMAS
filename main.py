"""智慧图书馆推荐 Demo 的业务编排与 Agent 实现。

总体模块关系
------------
* ``app.py`` 是 Streamlit 表现层：读取用户配置，调用本模块的
  ``make_llm`` 和 ``build_graph``，并把图执行期间的事件显示在网页上。
* 本模块是业务/编排层：把一次推荐请求转换为共享状态，交给 LangGraph
  中的 Agent 节点按依赖关系协作，最终产出推荐结果、审核结果和执行记录。
* ``data/catalog.json``、``data/reader_profile.json`` 仅供离线演示模式使用；
  在线模式通过 ``crossref_api.py`` 实时检索 Crossref 书目元数据，并接收本次
  用户提供的已读 DOI 列表。
* 命令行入口 ``main()`` 与网页入口 ``app.py`` 共用同一个 ``build_graph``，
  因此两种入口执行的是同一套业务流程。

一次请求的数据流
----------------
用户任务 -> 需求分析 -> (资源检索 || 读者画像) -> 推荐策展 -> 质量审核
         -> 协调汇总 -> 最终答复

LangGraph 的 GraphState 是节点之间共享的业务状态；节点返回状态更新。
网页运行时，节点另通过自定义事件发送实时状态和可查看的输入/输出快照，
由 ``app.py`` 更新卡片与右侧详情面板。自定义事件用于观察进度，不替代
GraphState，也不改变节点的业务结果。

当前实现是用于验证协作链路的固定拓扑基线：在线模式使用 DeepSeek 处理需求
抽取、检索词扩展和候选排序，并调用 Crossref 获取真实书目元数据；约束过滤和
质量审核仍由本地规则执行。它尚未实现运行时动态重规划，也没有接入真实馆藏微服务。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from operator import add
from pathlib import Path
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from crossref_api import search_crossref


# 资源路径固定相对于本文件定位，使 CLI 和从其他目录启动的 Streamlit
# 都能读取同一组样例数据。
ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "data" / "catalog.json"
PROFILE_PATH = ROOT / "data" / "reader_profile.json"

DEFAULT_TASK = (
    "请为我的硕士论文‘基于AI Agent的智慧图书馆信息推荐模型构建研究’"
    "推荐5篇中英文文献，优先2022年及以后，覆盖多智能体协同、智慧图书馆和信息推荐，"
    "优先开放获取，并排除我已读的文献。"
)

# Agent 的稳定节点 ID 同时用于 LangGraph、网页卡片、运行记录和状态事件。
# 修改 ID 时要同步检查 app.py 中的图标映射及展示配置。
AGENT_LABELS = {
    "need_analyst": "需求分析 Agent",
    "catalog_search_agent": "资源检索 Agent",
    "reader_profile_agent": "读者画像 Agent",
    "recommendation_agent": "推荐策展 Agent",
    "quality_review_agent": "质量审核 Agent",
    "coordinator_agent": "协调汇总 Agent",
}

# 将每个节点的系统提示词/执行说明集中管理，避免模型调用和详情面板显示
# 两套文本不一致。确定性节点也显示执行说明，但不会伪装成模型提示词。
AGENT_PROMPTS = {
    "need_analyst": (
        "你是图书馆推荐需求分析Agent。把用户表述转为结构化需求。"
        "区分硬约束和‘优先/尽量’这类软偏好；只记录明确表达的信息，"
        "不确定时使用默认值，不得补造用户条件。"
    ),
    "catalog_search_agent": (
        "你是图书馆学术资源检索Agent。为给定主题提出少量中英文同义检索词，"
        "用于调用学术元数据检索服务。保留主题边界，不要引入无关概念。"
    ),
    "reader_profile_agent": (
        "读取当前任务可用的读者上下文和已读资源标识，并将其提供给推荐流程。"
        "在线模式只使用用户本次输入；离线模式使用本地样例画像。"
    ),
    "recommendation_agent": (
        "你是图书馆推荐策展Agent。只能从给定候选资源中选择，不得编造资源或ID。"
        "优先满足语言覆盖和主题相关性，再考虑年份与开放获取偏好。"
        "理由应简洁并引用候选资源字段。"
    ),
    "quality_review_agent": (
        "逐项核验推荐数量、语言覆盖、已读资源排除和硬性年份约束；"
        "记录每项检查结果及未满足项。当前Demo由确定性规则执行。"
    ),
    "coordinator_agent": (
        "汇总需求、推荐清单和质量审核结果，生成面向用户的推荐答复；"
        "明确标出未通过的约束、资源来源和元数据边界。当前由确定性模板执行。"
    ),
}


class DemandSpec(BaseModel):
    """用户当前推荐任务的结构化需求。"""

    topic: str = Field(description="推荐主题")
    search_terms: list[str] = Field(default_factory=list, description="用于目录检索的中英文关键词")
    languages: list[str] = Field(default_factory=list, description="希望覆盖的语言，只能使用中文或英文")
    require_all_languages: bool = Field(default=False, description="是否要求每种指定语言都出现在结果中")
    min_year: int | None = Field(default=None, description="年份下限；若只是‘优先’则属于软偏好")
    year_is_hard: bool = Field(default=False, description="年份下限是否为必须满足的硬约束")
    top_k: int = Field(default=5, ge=1, le=10, description="希望返回的资源数量")
    prefer_open_access: bool = Field(default=False, description="是否优先开放获取")
    exclude_read_items: bool = Field(default=True, description="是否排除用户已读资源")


class SearchPlan(BaseModel):
    """检索 Agent 可选生成的关键词扩展结果。"""

    terms: list[str] = Field(default_factory=list, description="从主题扩展出的目录检索词")


class RecommendedPick(BaseModel):
    """LLM 推荐排序的单条结构化输出；资源 ID 必须来自候选集。"""

    item_id: str
    reason: str


class RecommendationPlan(BaseModel):
    """LLM 推荐排序结果。后续仍由本地逻辑校验 ID 并补足数量。"""

    recommendations: list[RecommendedPick] = Field(default_factory=list)


class GraphState(TypedDict, total=False):
    """LangGraph 节点共享的业务状态。

    普通字段由节点按键写入；``agent_trace`` 与 ``agent_records`` 使用
    ``operator.add`` 作为 reducer，因此不同节点返回的列表会追加合并。
    这使检索与画像两个并行节点能各自写入记录，而不覆盖对方的数据。
    """

    task: str  # 用户原始任务文本；需求分析节点的输入。
    demand: dict[str, Any]  # DemandSpec 序列化后的需求状态，供检索/推荐/审核使用。
    search_terms: list[str]  # 检索节点最终采用的关键词（含可选 LLM 扩展词）。
    search_results: list[dict[str, Any]]  # 从实时 API 或离线目录筛出的候选资源。
    reader_profile: dict[str, Any]  # 读者兴趣及已读资源等画像数据。
    selected_items: list[dict[str, Any]]  # 推荐策展节点选出的资源及推荐理由。
    review: dict[str, Any]  # 质量审核的逐项检查、是否通过和未满足项。
    final_answer: str  # 协调汇总节点面向用户生成的 Markdown 答复。
    # 下列两个列表用 reducer 追加而非覆盖，聚合每个节点返回的记录。
    agent_trace: Annotated[list[dict[str, str]], add]  # 面向用户的简明协作轨迹。
    agent_records: Annotated[list[dict[str, Any]], add]  # 详情面板用的输入/输出执行快照。


def summarize_resource(item: dict[str, Any], abstract_limit: int = 360) -> dict[str, Any]:
    """为 Agent 提示词和详情面板生成紧凑资源摘要，避免传入整段超长元数据。"""

    abstract = re.sub(r"\s+", " ", str(item.get("abstract", "") or "")).strip()
    if len(abstract) > abstract_limit:
        abstract = abstract[:abstract_limit].rstrip() + "…"
    return {
        "id": item.get("id"),
        "title": item.get("title"),
        "year": item.get("year"),
        "language": item.get("language"),
        "language_basis": item.get("language_basis"),
        "authors": item.get("creator"),
        "venue": item.get("container_title") or item.get("publisher") or item.get("category"),
        "subjects": item.get("keywords", [])[:5],
        "abstract_excerpt": abstract,
        "doi": item.get("doi"),
        "url": item.get("url"),
        "open_access": item.get("open_access"),
        "matched_terms": item.get("matched_terms", [])[:6],
        "search_score": item.get("search_score"),
    }


def make_agent_record(
    node: str,
    input_state: dict[str, Any],
    output_state: dict[str, Any],
    mode: str,
    *,
    prompt_input: str | None = None,
    model_call_attempted: bool = False,
) -> dict[str, Any]:
    """构造网页详情面板使用的单个 Agent 执行快照。

    记录包括提示词/执行说明、模型实际输入、执行模式以及节点的输入与
    输出业务数据。API Key 和模型隐藏推理过程不属于记录字段；运行中可先
    保存输入快照，节点结束后再用完整输出快照覆盖展示。
    """

    is_model_agent = node in {
        "need_analyst",
        "catalog_search_agent",
        "recommendation_agent",
    }
    return {
        "node": node,
        "label": AGENT_LABELS[node],
        "prompt_type": "系统提示词" if is_model_agent else "执行说明",
        "prompt": AGENT_PROMPTS[node],
        "prompt_input": prompt_input,
        "model_call_attempted": model_call_attempted,
        "mode": mode,
        "input_state": input_state,
        "output_state": output_state,
    }


def read_json(path: Path) -> Any:
    """读取 UTF-8 JSON 样例数据；调用方负责提供项目内的数据路径。"""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalise_languages(languages: list[str]) -> list[str]:
    """只保留系统支持的语言标签，并按首次出现顺序去重。"""

    allowed = {"中文", "英文"}
    return list(dict.fromkeys(language for language in languages if language in allowed))


def extract_demand_locally(task: str) -> DemandSpec:
    """用规则从自然语言任务中提取需求，作为可复现的离线基线/回退路径。

    该解析器只覆盖 Demo 预设的表达方式，不等同于生产级自然语言理解。
    年份偏好与年份硬约束分别建模，避免把“优先某年以后”误当成强制过滤。
    """

    count_match = re.search(r"(\d+)\s*篇", task)
    top_k = int(count_match.group(1)) if count_match else 5
    top_k = min(max(top_k, 1), 10)

    year_match = re.search(r"(20\d{2})年(?:及以后|以后|之后|后)", task)
    min_year = int(year_match.group(1)) if year_match else None
    year_is_hard = bool(re.search(r"必须|限定|仅限|不得早于", task)) and min_year is not None

    asks_for_both = bool(
        re.search(r"中英文|中文和英文|中文与英文|英文和中文|英文与中文", task)
    )
    has_chinese = "中文" in task
    has_english = "英文" in task or "英语" in task
    languages = ["中文", "英文"] if asks_for_both else []
    if not asks_for_both and has_chinese:
        languages.append("中文")
    if not asks_for_both and has_english:
        languages.append("英文")
    if not languages:
        languages = ["中文", "英文"]
    require_all_languages = asks_for_both

    lower_task = task.casefold()
    search_terms: list[str] = []
    if "多智能体" in task or "multi-agent" in lower_task or "agent" in lower_task:
        search_terms.extend(
            ["多智能体协同", "多智能体", "multi-agent collaboration", "multi-agent"]
        )
    if "智慧图书馆" in task or "smart library" in lower_task:
        search_terms.extend(["智慧图书馆", "smart library"])
    if "推荐" in task or "recommendation" in lower_task:
        search_terms.extend(
            ["信息推荐", "图书推荐", "library recommendation", "academic recommendation", "recommendation"]
        )
    if "微服务" in task or "microservice" in lower_task:
        search_terms.extend(["微服务", "microservices"])
    if "动态" in task or "dynamic" in lower_task:
        search_terms.extend(["动态需求", "dynamic workflow"])

    return DemandSpec(
        topic=task,
        search_terms=list(dict.fromkeys(search_terms)),
        languages=languages,
        require_all_languages=require_all_languages,
        min_year=min_year,
        year_is_hard=year_is_hard,
        top_k=top_k,
        prefer_open_access=bool(re.search(r"开放获取|open access|免费全文", task, re.I)),
        exclude_read_items=bool(re.search(r"已读|看过|阅读过", task)),
    )


def make_llm(
    api_key: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
) -> Any:
    """构建兼容 OpenAI 接口的 DeepSeek 聊天模型客户端。

    参数优先于 ``.env``/环境变量；密钥只传给模型客户端，不写入共享状态、
    Agent 运行记录或页面输出。
    """

    try:
        from dotenv import load_dotenv
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise SystemExit(
            "DeepSeek依赖尚未安装。请在项目目录运行：uv sync"
        ) from exc

    load_dotenv(ROOT / ".env")
    api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    model_name = model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not api_key or not model_name:
        raise SystemExit(
            "DeepSeek模式需要在 .env 中配置 DEEPSEEK_API_KEY；"
            "也可以直接运行默认的本地演示模式。"
        )

    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": 0,
        "timeout": 25,
        "max_retries": 0,
        # DeepSeek 默认启用 thinking；当前 LangChain 的工具选择会发送 tool_choice，
        # 因此显式关闭 thinking，避免结构化 function-calling 请求被 API 拒绝。
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    return ChatOpenAI(**kwargs)


def safe_structured_call(
    llm: Any,
    schema: type[BaseModel],
    system_prompt: str,
    user_prompt: str,
) -> BaseModel:
    """按 Pydantic Schema 调用模型并取得结构化结果。

    采用 function calling 方式要求模型按 schema 返回；具体节点捕获调用异常，
    再决定使用本地规则回退，避免单次模型错误直接中断整条 Demo 流程。
    """
    messages = [
        ("system", system_prompt),
        ("human", user_prompt),
    ]
    # DeepSeek supports tool calls; function-calling avoids relying on an
    # OpenAI-only JSON Schema response format.
    return llm.with_structured_output(schema, method="function_calling").invoke(messages)


def structured_call_with_progress(
    llm: Any,
    schema: type[BaseModel],
    system_prompt: str,
    user_prompt: str,
    *,
    node: str,
    activity: str,
    heartbeat_seconds: float = 3.0,
) -> BaseModel:
    """异步等待单次 DeepSeek 结构化调用，并周期性向网页发送运行心跳。

    模型请求仍由 LangChain 同步执行；外层节点线程每隔固定时间检查 Future，
    等待期间通过 LangGraph custom event 更新状态卡片，避免页面看起来无响应。
    最终请求超时由 ChatOpenAI 的 timeout 设置限制。
    """

    started_at = time.monotonic()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="deepseek-agent")
    future = executor.submit(
        safe_structured_call,
        llm,
        schema,
        system_prompt,
        user_prompt,
    )
    try:
        while True:
            try:
                return future.result(timeout=heartbeat_seconds)
            except FutureTimeoutError:
                # 若 Future 已结束，这就是模型本身抛出的 TimeoutError，应交由节点报错处理。
                if future.done():
                    raise
                waited_seconds = int(time.monotonic() - started_at)
                emit_agent_status(
                    node,
                    "running",
                    f"DeepSeek 仍在处理{activity}（已等待 {waited_seconds} 秒）",
                )
    finally:
        executor.shutdown(wait=future.done(), cancel_futures=True)


def crossref_search_with_progress(
    query: str,
    rows: int,
    *,
    heartbeat_seconds: float = 3.0,
) -> list[dict[str, Any]]:
    """在 Crossref 请求等待期间周期性更新 Agent 状态，避免 UI 无反馈。"""

    started_at = time.monotonic()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="crossref-search")
    future = executor.submit(search_crossref, query, rows, 8)
    try:
        while True:
            try:
                return future.result(timeout=heartbeat_seconds)
            except FutureTimeoutError:
                if future.done():
                    raise
                waited_seconds = int(time.monotonic() - started_at)
                emit_agent_status(
                    "catalog_search_agent",
                    "running",
                    f"Crossref 仍在检索（已等待 {waited_seconds} 秒）",
                )
    finally:
        executor.shutdown(wait=future.done(), cancel_futures=True)


def emit_agent_status(
    node: str,
    phase: str,
    detail: str,
    agent_data: dict[str, Any] | None = None,
) -> None:
    """向 LangGraph 自定义事件通道发送节点进度，供网页实时呈现。

    ``agent_data`` 是可观察的业务状态快照，不是 LLM 的私有思维链。
    CLI 使用 ``invoke`` 时没有 stream writer，此时忽略写事件异常，业务执行
    仍继续；网页使用 ``stream`` 时由 app.py 消费这些事件。
    """

    try:
        event: dict[str, Any] = {"node": node, "phase": phase, "detail": detail}
        if agent_data is not None:
            event["agent_data"] = agent_data
        get_stream_writer()(event)
    except Exception:
        # The CLI uses graph.invoke() without a stream writer.
        pass


def build_graph(
    llm: Any = None,
    *,
    catalog_source: str = "demo",
    reader_profile: dict[str, Any] | None = None,
    require_llm: bool = False,
):
    """构造并编译推荐工作流，返回可 ``invoke`` 或 ``stream`` 的 LangGraph。

    ``catalog_source="crossref"`` 时，检索节点实时调用 Crossref；该路径失败时
    会报错，不会静默切回虚构样例。``catalog_source="demo"`` 才读取本地样例目录。
    ``reader_profile`` 可注入当前用户提供的画像；未传入时，在线模式使用空画像，
    离线模式读取样例画像。节点只返回状态增量，不直接依赖 Streamlit 页面对象。
    ``require_llm=True`` 时模型调用失败会中止流程，不静默回退到本地规则；用于
    需要确认真实模型参与的在线模式。常规 demo/llm 模式仍保留降级能力。
    """
    if catalog_source not in {"demo", "crossref"}:
        raise ValueError("catalog_source 只支持 'demo' 或 'crossref'")
    catalog = read_json(CATALOG_PATH) if catalog_source == "demo" else []
    if reader_profile is not None:
        profile = reader_profile
    elif catalog_source == "demo":
        profile = read_json(PROFILE_PATH)
    else:
        profile = {
            "profile_id": "current-session",
            "read_item_ids": [],
            "interest_keywords": [],
            "source": "在线模式未提供额外读者画像",
        }

    def need_analyst(state: GraphState) -> dict[str, Any]:
        """把用户原始任务转换为共享状态中的结构化需求。

        优先尝试 LLM + DemandSpec 校验；若离线或调用失败，则使用本地规则。
        先发 running 事件以便 UI 展示真实输入，完成后返回需求、运行记录和轨迹。
        """
        task = state["task"]
        system_prompt = AGENT_PROMPTS["need_analyst"]
        input_state = {"task": task}
        emit_agent_status(
            "need_analyst",
            "running",
            "正在分析主题、数量和约束",
            agent_data=make_agent_record(
                "need_analyst",
                input_state,
                {},
                "LLM结构化抽取进行中" if llm is not None else "离线规则解析",
                prompt_input=task,
                model_call_attempted=llm is not None,
            ),
        )
        fallback_note = "规则解析"
        if llm is not None:
            try:
                # 使用结构化输出约束字段；捕获模型/网络/解析异常，保留离线降级路径。
                demand = structured_call_with_progress(
                    llm,
                    DemandSpec,
                    system_prompt,
                    task,
                    node="need_analyst",
                    activity="需求解析",
                )
                fallback_note = "LLM结构化抽取"
            except Exception as exc:  # retain a local fallback so the demo remains usable
                if require_llm:
                    raise RuntimeError(
                        f"DeepSeek 需求解析调用失败（{type(exc).__name__}）"
                    ) from None
                demand = extract_demand_locally(task)
                fallback_note = f"LLM调用失败，回退到规则解析（{type(exc).__name__}）"
        else:
            demand = extract_demand_locally(task)

        demand_dict = demand.model_dump()
        demand_dict["languages"] = normalise_languages(demand_dict.get("languages", []))
        record = make_agent_record(
            "need_analyst",
            input_state,
            {"demand": demand_dict, "extraction_mode": fallback_note},
            fallback_note,
            prompt_input=task,
            model_call_attempted=llm is not None,
        )
        emit_agent_status(
            "need_analyst", "complete", "需求状态已形成", agent_data=record
        )
        return {
            "demand": demand_dict,
            "agent_records": [record],
            "agent_trace": [
                {
                    "agent": "需求分析Agent",
                    "action": "从用户任务中识别主题、数量、语言和偏好",
                    "result": fallback_note,
                }
            ],
        }

    def catalog_search_agent(state: GraphState) -> dict[str, Any]:
        """从实时 Crossref 或离线目录检索候选资源，并输出检索词与排序结果。

        LLM 负责扩展关键词；在线模式把关键词实际发送到 Crossref Works API，
        离线模式查询本地样例。语言和硬性年份条件由本地逻辑过滤，软偏好由规则
        计分；在线结果保留 DOI、来源链接和元数据证据供推荐节点及用户核验。
        """
        demand = state["demand"]
        terms = list(demand.get("search_terms", []))
        note = "使用离线本地样例目录" if catalog_source == "demo" else "调用 Crossref 实时检索"
        system_prompt = AGENT_PROMPTS["catalog_search_agent"]
        input_state = {"task": state["task"], "demand": demand}
        prompt_input = json.dumps(
            {"task": state["task"], "existing_terms": terms},
            ensure_ascii=False,
        )
        initial_detail = (
            "正在向 DeepSeek 请求检索词扩展"
            if catalog_source == "crossref" and llm is not None
            else "正在准备在线检索"
            if catalog_source == "crossref"
            else "正在扩展检索词并查询本地样例目录"
        )
        emit_agent_status(
            "catalog_search_agent",
            "running",
            initial_detail,
            agent_data=make_agent_record(
                "catalog_search_agent",
                input_state,
                {},
                "LLM检索词扩展进行中" if llm is not None else "本地关键词检索",
                prompt_input=prompt_input,
                model_call_attempted=llm is not None,
            ),
        )

        if llm is not None:
            try:
                plan = structured_call_with_progress(
                    llm,
                    SearchPlan,
                    system_prompt,
                    prompt_input,
                    node="catalog_search_agent",
                    activity="检索词扩展",
                )
                terms = list(dict.fromkeys(terms + plan.terms))
                note = "LLM扩展检索词后查询本地目录"
                if catalog_source == "crossref":
                    emit_agent_status(
                        "catalog_search_agent",
                        "running",
                        f"DeepSeek 已生成 {len(terms)} 个检索词，准备调用 Crossref",
                        agent_data=make_agent_record(
                            "catalog_search_agent",
                            input_state,
                            {"search_terms": terms},
                            "DeepSeek 检索词扩展完成，等待 Crossref 查询",
                            prompt_input=prompt_input,
                            model_call_attempted=True,
                        ),
                    )
            except Exception as exc:
                if require_llm:
                    raise RuntimeError(
                        f"DeepSeek 检索词扩展调用失败（{type(exc).__name__}）"
                    ) from None
                note = f"检索词扩展失败，使用原始关键词（{type(exc).__name__}）"

        if catalog_source == "crossref":
            # 将主题和少量扩展词合成一次检索，减少 Crossref 请求次数与触发限流的机会。
            topic = str(demand.get("topic", "")).strip()
            query_parts = [topic] if topic else []
            query_parts.extend(
                term.strip()
                for term in terms
                if term.strip() and (not topic or term.casefold() not in topic.casefold())
            )
            query = " ".join(dict.fromkeys(query_parts[:3]))[:240]
            if not query:
                query = str(state.get("task", "")).strip()[:240]
            queries = [query]
            rows_per_query = max(12, min(20, demand.get("top_k", 5) * 3))
            fetched: dict[str, dict[str, Any]] = {}
            for query_index, query in enumerate(queries, start=1):
                emit_agent_status(
                    "catalog_search_agent",
                    "running",
                    f"正在请求 Crossref（{query_index}/{len(queries)}）：{query[:36]}",
                    agent_data=make_agent_record(
                        "catalog_search_agent",
                        input_state,
                        {
                            "search_terms": terms,
                            "crossref_query": query,
                            "query_progress": f"{query_index}/{len(queries)}",
                        },
                        "等待 Crossref 实时元数据响应",
                        prompt_input=prompt_input,
                        model_call_attempted=llm is not None,
                    ),
                )
                query_results = crossref_search_with_progress(
                    query,
                    rows=rows_per_query,
                )
                for rank, item in enumerate(query_results):
                    item.setdefault("crossref_rank", rank)
                    item.setdefault("matched_query", query)
                    fetched.setdefault(item["id"], item)
                emit_agent_status(
                    "catalog_search_agent",
                    "running",
                    f"Crossref 第 {query_index} 次请求返回 {len(query_results)} 条，正在整理候选",
                    agent_data=make_agent_record(
                        "catalog_search_agent",
                        input_state,
                        {
                            "search_terms": terms,
                            "crossref_query": query,
                            "returned_count": len(query_results),
                            "unique_candidate_count": len(fetched),
                        },
                        "Crossref 实时元数据已返回",
                        prompt_input=prompt_input,
                        model_call_attempted=llm is not None,
                    ),
                )
            source_catalog = list(fetched.values())
            note = f"Crossref 实时元数据检索（{len(source_catalog)}条去重候选）"
        else:
            source_catalog = catalog

        ranked: list[tuple[float, dict[str, Any]]] = []
        # 先执行硬过滤，再对剩余资源按匹配词数计分；软偏好只加分，不直接排除。
        for item in source_catalog:
            if (
                demand.get("languages")
                and item.get("language") not in demand["languages"]
                and item.get("language") != "未知"
            ):
                continue
            if demand.get("year_is_hard") and (
                item.get("year") is None
                or item["year"] < (demand.get("min_year") or 0)
            ):
                continue
            text = " ".join(
                [
                    str(item.get("title", "")),
                    str(item.get("abstract", "")),
                    *[str(value) for value in item.get("keywords", [])],
                ]
            ).casefold()
            matched = [term for term in terms if term.casefold() in text]
            # 在线候选已由 Crossref 的相关性检索返回；无字面匹配时保留 API 排名，
            # 避免中英文关键词差异导致真实候选被误删。离线样例仍要求命中关键词。
            if not matched and catalog_source == "demo":
                continue
            api_rank_bonus = (
                max(0.0, 1.0 - item.get("crossref_rank", 0) / max(rows_per_query, 1))
                if catalog_source == "crossref"
                else 0.0
            )
            score = float(len(matched) * 2) + api_rank_bonus
            if demand.get("min_year") and item.get("year") and item["year"] >= demand["min_year"]:
                score += 1.0
            if demand.get("prefer_open_access") and item.get("open_access"):
                score += 1.25
            result = dict(item)
            result["search_score"] = round(score, 2)
            result["matched_terms"] = matched
            ranked.append((score, result))

        # 分数降序、年份降序、ID 升序构成稳定排序；只把有限候选交给推荐 Agent。
        ranked.sort(key=lambda pair: (-pair[0], -(pair[1].get("year") or 0), pair[1]["id"]))
        results = [item for _, item in ranked[: max(12, demand.get("top_k", 5) * 3)]]
        record = make_agent_record(
            "catalog_search_agent",
            input_state,
            {"search_terms": terms, "search_results": results, "search_mode": note},
            note,
            prompt_input=prompt_input,
            model_call_attempted=llm is not None,
        )
        emit_agent_status(
            "catalog_search_agent",
            "complete",
            f"找到{len(results)}条候选资源",
            agent_data=record,
        )
        return {
            "search_terms": terms,
            "search_results": results,
            "agent_records": [record],
            "agent_trace": [
                {
                    "agent": "资源检索Agent",
                    "action": "扩展检索词并查询馆藏目录",
                    "result": f"{note}；返回候选资源{len(results)}条",
                }
            ],
        }

    def reader_profile_agent(_: GraphState) -> dict[str, Any]:
        """读取当前运行模式提供的读者上下文，为推荐节点提供已读资源等信息。

        在线模式使用本次会话用户输入，离线模式读取本地画像样例；它和资源检索
        节点并行运行，不调用 LLM，也不修改画像数据。
        """
        input_state = {"profile_source": profile.get("source", "data/reader_profile.json")}
        emit_agent_status(
            "reader_profile_agent",
            "running",
            "正在读取兴趣与已读记录",
            agent_data=make_agent_record(
                "reader_profile_agent",
                input_state,
                {},
                "读取当前任务的读者上下文中",
            ),
        )
        record = make_agent_record(
            "reader_profile_agent",
            input_state,
            {"reader_profile": profile},
            "当前任务读者上下文读取",
        )
        emit_agent_status(
            "reader_profile_agent", "complete", "读者画像已载入", agent_data=record
        )
        return {
            "reader_profile": profile,
            "agent_records": [record],
            "agent_trace": [
                {
                    "agent": "读者画像Agent",
                    "action": "读取当前可用的读者上下文与已读记录",
                    "result": f"读取已读资源{len(profile.get('read_item_ids', []))}条",
                }
            ],
        }

    def recommendation_agent(state: GraphState) -> dict[str, Any]:
        """融合候选资源、结构化需求和读者画像，形成最终推荐列表。

        先按已读记录筛选；有语言全覆盖要求时先满足覆盖，再按候选分数和偏好补足。
        LLM 只能对现有候选 ID 排序/给理由，本地校验 ID 并负责兜底和数量截断。
        """
        demand = state["demand"]
        profile_data = state["reader_profile"]
        read_ids = set(profile_data.get("read_item_ids", []))
        candidates = list(state.get("search_results", []))
        # 已读排除只在需求明确启用时执行；审核节点会再次独立检查该约束。
        if demand.get("exclude_read_items"):
            candidates = [item for item in candidates if item["id"] not in read_ids]

        # 为模型构造有上限的候选子集：语言覆盖候选优先，再按资源检索排序补足。
        # 全量元数据仍保留在工作流状态中，但不重复塞入 LLM 请求和详情面板快照。
        candidate_limit = max(6, min(12, demand.get("top_k", 5) * 2))
        model_candidates: list[dict[str, Any]] = []
        model_candidate_ids: set[str] = set()
        if demand.get("require_all_languages"):
            for language in demand.get("languages", []):
                match = next(
                    (item for item in candidates if item.get("language") == language),
                    None,
                )
                if match is not None and match["id"] not in model_candidate_ids:
                    model_candidates.append(match)
                    model_candidate_ids.add(match["id"])
        for item in candidates:
            if item["id"] not in model_candidate_ids:
                model_candidates.append(item)
                model_candidate_ids.add(item["id"])
            if len(model_candidates) >= candidate_limit:
                break
        model_candidate_summaries = [
            summarize_resource(item, abstract_limit=360) for item in model_candidates
        ]
        input_state = {
            "demand": demand,
            "reader_profile": profile_data,
            "search_result_count": len(state.get("search_results", [])),
            "eligible_candidate_count": len(candidates),
            "llm_candidate_count": len(model_candidate_summaries),
            "llm_candidates": model_candidate_summaries,
        }
        prompt_input = json.dumps(
            {
                "demand": demand,
                "candidates": model_candidate_summaries,
                "already_read_ids": sorted(read_ids),
            },
            ensure_ascii=False,
        )
        emit_agent_status(
            "recommendation_agent",
            "running",
            "正在向 DeepSeek 请求候选排序"
            if llm is not None and candidates
            else "正在合并检索结果和读者画像",
            agent_data=make_agent_record(
                "recommendation_agent",
                input_state,
                {},
                "LLM排序进行中" if llm is not None and candidates else "本地确定性排序",
                prompt_input=prompt_input,
                model_call_attempted=llm is not None and bool(candidates),
            ),
        )

        selected: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        languages = demand.get("languages", [])

        # 如果用户要求覆盖所有指定语言，先各选一条，剩余名额再按综合排序补足。
        if demand.get("require_all_languages"):
            for language in languages:
                item = next(
                    (candidate for candidate in candidates
                     if candidate["language"] == language and candidate["id"] not in used_ids),
                    None,
                )
                if item is not None:
                    selected.append(item)
                    used_ids.add(item["id"])

        remaining = [item for item in candidates if item["id"] not in used_ids]
        # 本地排序是 LLM 不可用时的基线：检索相关性优先，其次开放获取偏好、年份。
        default_order = sorted(
            remaining,
            key=lambda item: (
                -item.get("search_score", 0),
                -(1 if demand.get("prefer_open_access") and item.get("open_access") else 0),
                -(item.get("year") or 0),
                item["id"],
            ),
        )
        reasons: dict[str, str] = {}
        selection_note = "本地确定性排序"
        system_prompt = AGENT_PROMPTS["recommendation_agent"]

        if llm is not None and candidates:
            try:
                plan = structured_call_with_progress(
                    llm,
                    RecommendationPlan,
                    system_prompt,
                    prompt_input,
                    node="recommendation_agent",
                    activity="候选排序",
                )
                by_id = {item["id"]: item for item in model_candidates}
                model_order: list[dict[str, Any]] = []
                for pick in plan.recommendations:
                    item = by_id.get(pick.item_id)
                    if item is not None and item["id"] not in used_ids:
                        model_order.append(item)
                        used_ids.add(item["id"])
                        reasons[item["id"]] = pick.reason
                default_order = model_order + [
                    item for item in default_order if item["id"] not in used_ids
                ]
                selection_note = "LLM候选排序并经本地ID校验"
            except Exception as exc:
                if require_llm:
                    raise RuntimeError(
                        f"DeepSeek 推荐排序调用失败（{type(exc).__name__}）"
                    ) from None
                # Deterministic ranking below is the documented fallback.
                selection_note = f"LLM调用失败，回退本地排序（{type(exc).__name__}）"

        for item in default_order:
            if item["id"] not in {selected_item["id"] for selected_item in selected}:
                selected.append(item)
            if len(selected) >= demand.get("top_k", 5):
                break

        selected = selected[: demand.get("top_k", 5)]
        output = []
        for item in selected:
            enriched = dict(item)
            enriched["recommendation_reason"] = reasons.get(
                item["id"],
                "主题关键词匹配；"
                + (
                    "符合年份偏好；"
                    if demand.get("min_year")
                    and item.get("year")
                    and item["year"] >= demand["min_year"]
                    else ""
                )
                + ("开放获取；" if item.get("open_access") else "")
                + f"资源类型：{item.get('category', '未标注')}。",
            )
            output.append(enriched)

        record = make_agent_record(
            "recommendation_agent",
            input_state,
            {"selected_items": output, "selection_mode": selection_note},
            selection_note,
            prompt_input=prompt_input,
            model_call_attempted=llm is not None and bool(candidates),
        )
        emit_agent_status(
            "recommendation_agent",
            "complete",
            f"形成{len(output)}条推荐",
            agent_data=record,
        )
        return {
            "selected_items": output,
            "agent_records": [record],
            "agent_trace": [
                {
                    "agent": "推荐策展Agent",
                    "action": "结合检索结果与已读记录进行筛选和排序",
                    "result": f"{selection_note}；形成推荐清单{len(output)}条",
                }
            ],
        }

    def quality_review_agent(state: GraphState) -> dict[str, Any]:
        """独立复核推荐清单是否满足数量、已读排除、语言和硬年份约束。"""
        demand = state["demand"]
        items = state.get("selected_items", [])
        read_ids = set(state.get("reader_profile", {}).get("read_item_ids", []))
        input_state = {
            "demand": demand,
            "selected_items": items,
            "read_item_ids": sorted(read_ids),
        }
        emit_agent_status(
            "quality_review_agent",
            "running",
            "正在核验推荐约束",
            agent_data=make_agent_record(
                "quality_review_agent",
                input_state,
                {},
                "本地确定性规则执行中",
            ),
        )
        # 审核采用可解释的布尔规则，不让生成模型自行判断是否“基本满足”。
        # 软偏好（如优先某年份）不作为失败条件，只有 year_is_hard 才强制检查。
        checks = {
            "数量达到目标": len(items) >= demand.get("top_k", 5),
            "未包含已读资源": not demand.get("exclude_read_items")
            or all(item["id"] not in read_ids for item in items),
            "语言覆盖满足要求": not demand.get("require_all_languages")
            or set(demand.get("languages", [])).issubset({item["language"] for item in items}),
            "硬性年份约束满足": not demand.get("year_is_hard")
            or all(
                item.get("year") is not None
                and item["year"] >= demand.get("min_year", 0)
                for item in items
            ),
        }
        review = {
            "passed": all(checks.values()),
            "checks": checks,
            "issues": [name for name, passed in checks.items() if not passed],
        }
        record = make_agent_record(
            "quality_review_agent",
            input_state,
            {"review": review},
            "本地确定性规则",
        )
        emit_agent_status(
            "quality_review_agent",
            "complete" if review["passed"] else "warning",
            "全部检查通过" if review["passed"] else "存在未满足项",
            agent_data=record,
        )
        return {
            "review": review,
            "agent_records": [record],
            "agent_trace": [
                {
                    "agent": "质量审核Agent",
                    "action": "核验数量、语言、已读排除与硬约束",
                    "result": "通过" if review["passed"] else "存在未满足项",
                }
            ],
        }

    def coordinator_agent(state: GraphState) -> dict[str, Any]:
        """将需求、推荐项与审核结论整理成含可核验来源的 Markdown 答复。"""
        demand = state["demand"]
        items = state.get("selected_items", [])
        review = state.get("review", {})
        input_state = {"demand": demand, "selected_items": items, "review": review}
        emit_agent_status(
            "coordinator_agent",
            "running",
            "正在汇总Agent结果",
            agent_data=make_agent_record(
                "coordinator_agent",
                input_state,
                {},
                "本地模板汇总中",
            ),
        )
        lines = [
            "## 推荐结果",
            f"需求主题：{demand.get('topic', '未指定')}",
            f"共推荐 {len(items)} 条资源；质量核验：{'通过' if review.get('passed') else '需人工检查'}。",
            "",
        ]
        for index, item in enumerate(items, start=1):
            year_label = item.get("year") or "年份未知"
            oa_status = (
                "是（元数据许可标记）"
                if item.get("open_access") is True
                else "否"
                if item.get("open_access") is False
                else "未知（来源未提供充分许可信息）"
            )
            lines.extend(
                [
                    f"{index}. **{item['title']}**（{item.get('language', '未知')}，{year_label}）",
                    f"   - 作者：{item.get('creator') or '元数据未提供'}；来源：{item.get('container_title') or item.get('publisher') or item.get('category', '未标注')}",
                    f"   - 推荐理由：{item.get('recommendation_reason', '与当前主题相关。')}",
                    f"   - DOI/来源链接：{item.get('url') or '元数据未提供'}",
                    f"   - 数据来源：{item.get('source', '未标注')}；语言依据：{item.get('language_basis', '未标注')}；开放获取：{oa_status}",
                ]
            )
        if review.get("issues"):
            lines.extend(["", "待处理项：" + "、".join(review["issues"])])
        if catalog_source == "demo":
            lines.extend(
                ["", "> 提醒：当前目录是用于验证流程的演示数据，不能作为真实文献引用。"]
            )
        else:
            lines.extend(
                [
                    "",
                    "> 元数据由 Crossref 实时检索；请通过 DOI/来源链接核对出版方页面。Crossref 书目记录不等于全文开放。",
                ]
            )
        final_answer = "\n".join(lines)
        record = make_agent_record(
            "coordinator_agent",
            input_state,
            {"final_answer": final_answer},
            "本地模板汇总",
        )
        emit_agent_status(
            "coordinator_agent",
            "complete",
            f"汇总{len(items)}条推荐",
            agent_data=record,
        )
        return {
            "final_answer": final_answer,
            "agent_records": [record],
            "agent_trace": [
                {
                    "agent": "协调汇总Agent",
                    "action": "汇总各Agent结果与审核状态",
                    "result": f"输出{len(items)}条推荐并附带审核结论",
                }
            ],
        }

    # StateGraph 是协作编排骨架：节点负责局部能力，边负责先后依赖与并行关系。
    workflow = StateGraph(GraphState)
    workflow.add_node("need_analyst", need_analyst)
    workflow.add_node("catalog_search_agent", catalog_search_agent)
    workflow.add_node("reader_profile_agent", reader_profile_agent)
    workflow.add_node("recommendation_agent", recommendation_agent)
    workflow.add_node("quality_review_agent", quality_review_agent)
    workflow.add_node("coordinator_agent", coordinator_agent)

    workflow.add_edge(START, "need_analyst")
    # 需求分析完成后扇出为两个可并行的子任务：查资源、读画像。
    workflow.add_edge("need_analyst", "catalog_search_agent")
    workflow.add_edge("need_analyst", "reader_profile_agent")
    # 该多源边形成同步汇合点：两个前置节点都完成后才启动推荐策展。
    workflow.add_edge(["catalog_search_agent", "reader_profile_agent"], "recommendation_agent")
    workflow.add_edge("recommendation_agent", "quality_review_agent")
    workflow.add_edge("quality_review_agent", "coordinator_agent")
    workflow.add_edge("coordinator_agent", END)
    return workflow.compile()


def main() -> None:
    """命令行入口；与网页入口共享同一图，只是输出方式不同。"""
    parser = argparse.ArgumentParser(description="智慧图书馆推荐的最小多智能体协作Demo")
    parser.add_argument(
        "--mode",
        choices=["demo", "llm", "live"],
        default="demo",
        help="demo=本地样例；llm=DeepSeek+本地样例；live=DeepSeek+Crossref实时元数据",
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help="输入一条推荐任务")
    parser.add_argument("--show-graph", action="store_true", help="打印LangGraph Mermaid结构")
    args = parser.parse_args()

    llm = make_llm() if args.mode in {"llm", "live"} else None
    catalog_source = "crossref" if args.mode == "live" else "demo"
    profile = (
        {
            "profile_id": "cli-current-session",
            "read_item_ids": [],
            "interest_keywords": [],
            "source": "CLI当前任务（未提供已读 DOI）",
        }
        if args.mode == "live"
        else None
    )
    graph = build_graph(
        llm=llm,
        catalog_source=catalog_source,
        reader_profile=profile,
        require_llm=args.mode == "live",
    )
    if args.show_graph:
        print(graph.get_graph().draw_mermaid())
        print()

    # CLI 采用一次性 invoke，因此没有网页实时事件；结果从最终状态读取。
    result = graph.invoke({"task": args.task, "agent_trace": [], "agent_records": []})
    print("=" * 72)
    mode_labels = {
        "demo": "离线演示规则 + 本地样例数据",
        "llm": "DeepSeek + 本地样例数据",
        "live": "DeepSeek + Crossref 实时元数据",
    }
    print(f"运行模式：{mode_labels[args.mode]}")
    print("=" * 72)
    print(result["final_answer"])
    print("\n## 协作过程")
    for entry in result.get("agent_trace", []):
        print(f"- {entry['agent']}：{entry['action']}；{entry['result']}。")


if __name__ == "__main__":
    main()
