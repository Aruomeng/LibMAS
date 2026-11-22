"""智慧图书馆推荐 Demo 的 Streamlit 可视化与交互入口。

总体模块关系
------------
* ``main.py`` 实现推荐业务：定义共享状态、Agent 节点和 LangGraph 工作流；
  它不依赖 Streamlit，因此既能供网页调用，也能从命令行运行。
* 本模块实现表现层：读取任务、运行模式和 API Key 等用户配置，调用
  ``main.make_llm``/``main.build_graph``，再把图的事件和状态增量展示出来。
* ``data/catalog.json`` 与 ``data/reader_profile.json`` 由 ``main.py`` 的节点
  读取；网页不直接操作这些业务数据，而是消费图节点产生的运行事件/结果。

网页运行链路
------------
侧栏配置 -> build_graph(...) -> graph.stream(...) -> Agent 自定义事件与状态更新
          -> 卡片状态/详情抽屉/协作轨迹/最终答复

自定义事件适合即时显示“运行中”状态和节点输入快照；LangGraph 的 updates
则携带节点已完成后的正式状态更新。网页分别消费两类数据，并将跨 Streamlit
重绘需要保留的结果放入 ``st.session_state``。密钥仅在创建模型客户端时使用，
不保存到 Agent 运行记录中。右侧面板展示业务输入/输出和提示词，不展示隐藏推理。
"""

from __future__ import annotations

import copy
import base64
import html
import json
import os
import re
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from main import AGENT_LABELS, AGENT_PROMPTS, DEFAULT_TASK, build_graph, make_llm


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# 展示顺序、节点 ID、全局名称和卡片副标题。节点 ID 与 main.py 的图节点一致，
# 因而运行事件可以直接映射到对应卡片和详情面板。
AGENTS = [
    ("need_analyst", AGENT_LABELS["need_analyst"], "需求解析"),
    ("catalog_search_agent", AGENT_LABELS["catalog_search_agent"], "目录检索"),
    ("reader_profile_agent", AGENT_LABELS["reader_profile_agent"], "读者画像"),
    ("recommendation_agent", AGENT_LABELS["recommendation_agent"], "推荐策展"),
    ("quality_review_agent", AGENT_LABELS["quality_review_agent"], "质量审核"),
    ("coordinator_agent", AGENT_LABELS["coordinator_agent"], "结果汇总"),
]

# 每个 Agent 卡片的内嵌 SVG 图标；采用 data URI 嵌入按钮标签，避免依赖外部图片。
AGENT_SVGS = {
    "need_analyst": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#7c9cff" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v15H6.5A2.5 2.5 0 0 0 4 20.5z"/><path d="M4 5.5v15M8 7h8M8 10.5h7"/></svg>',
    "catalog_search_agent": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#7c9cff" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="10.8" cy="10.8" r="6.8"/><path d="m16 16 4.5 4.5M8 8.5h5.5M8 11h4"/></svg>',
    "reader_profile_agent": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#7c9cff" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="3.5"/><path d="M5 20c.7-3.3 3.2-5 7-5s6.3 1.7 7 5M4 4h2M18 4h2"/></svg>',
    "recommendation_agent": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#7c9cff" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3 1.7 5.3L19 10l-5.3 1.7L12 17l-1.7-5.3L5 10l5.3-1.7zM19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8z"/></svg>',
    "quality_review_agent": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#7c9cff" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 20 6v5.5c0 4.5-3.2 7.6-8 9.5-4.8-1.9-8-5-8-9.5V6z"/><path d="m8.5 12 2.2 2.2 4.8-5"/></svg>',
    "coordinator_agent": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#7c9cff" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M8 5h12M8 12h12M8 19h12M3.5 5h.01M3.5 12h.01M3.5 19h.01"/><path d="m2.5 5 1 1 2-2M2.5 12l1 1 2-2M2.5 19l1 1 2-2"/></svg>',
}

# 内部 phase 到中文状态名/CSS 样式类的映射，与 main.py 发出的事件 phase 对齐。
STATUS_META = {
    "waiting": {"label": "等待运行", "class": "waiting"},
    "running": {"label": "运行中", "class": "running"},
    "complete": {"label": "已完成", "class": "complete"},
    "warning": {"label": "需检查", "class": "warning"},
    "error": {"label": "执行失败", "class": "error"},
}


def initial_statuses() -> dict[str, dict[str, str]]:
    """为每次新任务建立所有 Agent 的初始 UI 状态。"""

    return {
        node: {"phase": "waiting", "detail": "等待任务启动"}
        for node, _, _ in AGENTS
    }


def inject_styles() -> None:
    """注入卡片、状态标识和右侧固定详情面板的页面样式。"""

    st.markdown(
        """
        <style>
          div[class*="st-key-agent-card-"] {
            border: 1px solid rgba(128,139,156,.26); border-radius: 15px;
            padding: 8px 11px 10px; margin: 0 0 14px;
            background: linear-gradient(145deg, rgba(255,255,255,.075), rgba(120,140,170,.035));
            box-shadow: 0 5px 18px rgba(20,32,52,.06);
            transition: border-color .18s ease, transform .18s ease;
          }
          div[class*="st-key-agent-card-"]:hover {
            border-color: rgba(108,143,255,.72); transform: translateY(-1px);
          }
          div[class*="st-key-agent-card-"] [data-testid="stButton"] > button {
            width:100%; min-height:76px; justify-content:flex-start; text-align:left;
            border:0; background:transparent; box-shadow:none; padding:10px 8px;
          }
          div[class*="st-key-agent-card-"] [data-testid="stButton"] > button:hover {
            color:inherit; border:0; background:rgba(110,135,190,.08); box-shadow:none;
          }
          div[class*="st-key-agent-card-"] [data-testid="stButton"] p {
            text-align:left; line-height:1.35; white-space:pre-line;
          }
          div[class*="st-key-agent-card-"] [data-testid="stButton"] img {
            width:24px; height:24px; max-height:24px; object-fit:contain; vertical-align:middle;
          }
          .agent-live-status { display:flex; align-items:center; gap:7px; font-size:12px; line-height:1.4; }
          .agent-live-status svg { width:15px; height:15px; flex:0 0 15px; fill:none; stroke:currentColor; stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; }
          .agent-live-status.waiting { color:#939baa; }
          .agent-live-status.running { color:#76a7ff; }
          .agent-live-status.complete { color:#45bd8a; }
          .agent-live-status.warning { color:#e7a93b; }
          .agent-live-status.error { color:#ec7777; }
          div.st-key-agent_detail_panel {
            position:fixed; z-index:999; top:5.25rem; right:1rem;
            width:min(440px, calc(100vw - 2rem)); max-height:calc(100vh - 6.25rem);
            overflow-y:auto; border:1px solid rgba(128,139,156,.25); border-radius:16px;
            padding:16px; background:var(--background-color, #111827);
            box-shadow:-12px 12px 36px rgba(10,18,34,.24);
            animation: agent-drawer-in .22s ease-out;
          }
          @keyframes agent-drawer-in { from { opacity:.45; transform:translateX(12px); } to { opacity:1; transform:translateX(0); } }
          .drawer-head { display:flex; align-items:center; gap:10px; }
          .drawer-head svg { width:25px; height:25px; color:#7c9cff; fill:none; stroke:currentColor; stroke-width:1.7; stroke-linecap:round; stroke-linejoin:round; }
          .drawer-note { color:#8b94a4; font-size:12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def status_markup(state: dict[str, str]) -> str:
    """将单个节点的运行状态渲染为带 SVG 状态图标的安全 HTML 片段。"""

    phase = state.get("phase", "waiting")
    meta = STATUS_META.get(phase, STATUS_META["waiting"])
    icon = {
        "waiting": '<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="7"/><path d="M10 6v4l2.5 1.5"/></svg>',
        "running": '<svg viewBox="0 0 20 20"><path d="M2 10h3l2-5 3.5 10 2.3-6H18"/></svg>',
        "complete": '<svg viewBox="0 0 20 20"><path d="m4 10 4 4 8-8"/></svg>',
        "warning": '<svg viewBox="0 0 20 20"><path d="M10 3 18 17H2z"/><path d="M10 7v4m0 3h.01"/></svg>',
        "error": '<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="7"/><path d="m7.5 7.5 5 5m0-5-5 5"/></svg>',
    }.get(phase, "")
    detail = html.escape(str(state.get("detail", "")))
    return (
        f'<div class="agent-live-status {meta["class"]}">{icon}'
        f'<span><b>{meta["label"]}</b> · {detail}</span></div>'
    )


def record_map(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """把按执行顺序保存的 Agent 记录转成按节点 ID 查找的字典。"""

    return {record.get("node"): record for record in records}


def parse_read_dois(raw_value: str) -> list[str]:
    """把用户输入的 DOI/doi.org 链接转为与在线 Crossref 资源 ID 一致的格式。"""

    identifiers: set[str] = set()
    for value in re.split(r"[,，;；\s]+", raw_value):
        doi = value.strip().strip("<>[]()")
        doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
        doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE).strip().rstrip(".,;，；")
        if doi.casefold().startswith("10."):
            identifiers.add(f"doi:{doi.casefold()}")
    return sorted(identifiers)


def agent_card_label(node: str, label: str, role: str) -> str:
    """生成可点击卡片的按钮内容：内嵌 SVG、Agent 名称和职责副标题。"""

    svg_data = base64.b64encode(AGENT_SVGS[node].encode("utf-8")).decode("ascii")
    return f"![{role}](data:image/svg+xml;base64,{svg_data}) **{label}**  \n{role}"


def update_detail_panel(
    slots: dict[str, Any],
    node: str,
    statuses: dict[str, dict[str, str]],
    live_details: dict[str, dict[str, Any]],
    records: dict[str, dict[str, Any]],
) -> None:
    """把选中 Agent 的最新运行快照写入右侧面板的 Streamlit 占位区。

    ``live_details`` 优先保存当前运行中的实时快照；节点结束后，正式
    ``agent_records`` 提供完整输出。如果还没有运行记录，则显示提示词和等待状态。
    """

    record = live_details.get(node) or records.get(node)
    status = statuses.get(node, {"phase": "waiting", "detail": "等待任务启动"})
    if "title" in slots:
        slots["title"].markdown(f"### {AGENT_LABELS[node]}")
    slots["status"].markdown(status_markup(status), unsafe_allow_html=True)
    slots["mode"].caption(
        f"执行路径：{record.get('mode', '尚无运行记录') if record else '尚无运行记录'}"
    )
    if record:
        slots["prompt"].code(
            record.get("prompt", AGENT_PROMPTS[node]),
            language="text",
            wrap_lines=True,
            height=130,
        )
        prompt_input = record.get("prompt_input")
        if prompt_input is not None:
            slots["message"].code(
                str(prompt_input),
                language="text",
                wrap_lines=True,
                height=180,
            )
            slots["message_label"].caption(
                "本次传给模型的消息"
                if record.get("model_call_attempted")
                else "模型输入模板（本次未调用模型）"
            )
        else:
            slots["message"].info("该节点使用本地逻辑，没有单独的模型消息。")
            slots["message_label"].empty()
        input_state = record.get("input_state")
        slots["input"].json(input_state if input_state else {"状态": "等待输入"})
        output_state = record.get("output_state")
        if output_state:
            slots["output"].json(output_state)
        else:
            slots["output"].info("正在执行，输出状态尚未生成。")
    else:
        slots["prompt"].code(
            AGENT_PROMPTS[node], language="text", wrap_lines=True, height=130
        )
        slots["message_label"].empty()
        slots["message"].info("尚未运行，没有实际模型消息。")
        slots["input"].info("尚未运行，没有实际输入状态。")
        slots["output"].info("尚未运行，没有实际输出状态。")


def render_detail_panel(
    parent: Any,
    node: str,
    statuses: dict[str, dict[str, str]],
    live_details: dict[str, dict[str, Any]],
    records: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """构造可关闭的右侧详情抽屉，并返回后续可原位刷新的 UI 占位对象。"""

    with parent.container(key="agent_detail_panel"):
        title_col, close_col = st.columns([0.83, 0.17])
        with title_col:
            title_slot = st.empty()
        with close_col:
            if st.button("×", key="close_agent_detail", help="收起详情面板"):
                st.session_state.selected_agent = None
                st.session_state.follow_active_agent = False
                st.rerun()
        status_slot = st.empty()
        status_tab, prompt_tab, input_tab, output_tab = st.tabs(
            ["状态", "提示词", "输入", "输出"]
        )
        slots: dict[str, Any] = {"title": title_slot, "status": status_slot}
        with status_tab:
            slots["mode"] = st.empty()
            st.caption("当前状态和执行路径来自实际流程事件。")
        with prompt_tab:
            slots["prompt"] = st.empty()
            slots["message_label"] = st.empty()
            slots["message"] = st.empty()
            st.markdown(
                '<div class="drawer-note">仅展示调用提示词与消息，不展示模型隐藏推理过程。</div>',
                unsafe_allow_html=True,
            )
        with input_tab:
            slots["input"] = st.empty()
        with output_tab:
            slots["output"] = st.empty()
    update_detail_panel(slots, node, statuses, live_details, records)
    return slots


def render_saved_run() -> None:
    """显示最近一次已完成任务的答复、需求/审核状态、轨迹和图结构。"""

    result = st.session_state.get("final_answer", "")
    if result:
        with st.expander("查看推荐结果", expanded=False):
            st.markdown(result)

    if st.session_state.get("run_complete"):
        with st.expander("查看需求状态、审核结果与协作轨迹", expanded=False):
            left, right = st.columns(2)
            with left:
                st.markdown("**结构化需求状态**")
                st.json(st.session_state.get("demand_state") or {})
            with right:
                st.markdown("**质量审核**")
                st.json(st.session_state.get("review_state") or {})
            for item in st.session_state.get("traces", []):
                st.markdown(
                    f"- **{item['agent']}**：{item['action']}；{item['result']}。"
                )
        with st.expander("查看 LangGraph 工作流结构", expanded=False):
            graph = build_graph(llm=None)
            st.code(graph.get_graph().draw_mermaid(), language="mermaid")


def run_app() -> None:
    """Streamlit 主入口，串联配置、页面布局、工作流流式执行和状态展示。

    每轮 Streamlit 交互都会重新运行本函数，所以运行结果放在
    ``st.session_state``；而单次图执行期间，局部占位对象用于逐事件刷新页面。
    用户手动选中的 Agent 会保持固定；没有手动选择时，详情面板可自动跟随当前节点。
    """
    st.set_page_config(
        page_title="智慧图书馆多智能体推荐 Demo",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()
    st.title("智慧图书馆多智能体推荐")
    st.caption("多 Agent 分工协作 · 并行检索 · 共享状态 · 推荐审核")

    # 第一阶段：选择真实在线或离线演示。真实在线会实际调用 DeepSeek 与 Crossref；
    # Crossref 返回书目元数据，DeepSeek 负责需求理解、检索词扩展和候选排序。
    with st.sidebar:
        st.header("任务设置")
        live_mode_label = "真实在线（DeepSeek + Crossref）"
        demo_mode_label = "离线演示（规则 + 本地样例）"
        mode = st.radio(
            "运行模式",
            [live_mode_label, demo_mode_label],
            help="真实在线会调用 DeepSeek API 和 Crossref 实时书目检索；离线演示只使用本地样例。",
        )
        live_mode = mode == live_mode_label
        model_options = ["deepseek-flash", "deepseek-v4-pro"]
        configured_model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        model_index = model_options.index(configured_model) if configured_model in model_options else 0
        model_name = st.selectbox(
            "DeepSeek 模型",
            model_options,
            index=model_index,
            disabled=not live_mode,
        )
        api_key = st.text_input(
            "DeepSeek API Key",
            value=os.getenv("DEEPSEEK_API_KEY", ""),
            type="password",
            disabled=not live_mode,
            help="也可在项目目录的 .env 中设置 DEEPSEEK_API_KEY。密钥仅用于 DeepSeek 请求，不进入 Agent 运行状态。",
        )
        read_dois_text = st.text_area(
            "已读 DOI（可选，逗号或换行分隔）",
            placeholder="例如：10.1234/example 或 https://doi.org/10.1234/example",
            height=72,
            disabled=not live_mode,
            help="真实在线模式只按你在本次页面输入的 DOI 排除已读资源，不读取本地演示画像。",
        )
        task = st.text_area("推荐任务", value=DEFAULT_TASK, height=150)
        run_clicked = st.button("开始协作推荐", type="primary", use_container_width=True)
        st.caption("在线数据来自 Crossref；若遇 HTTP 429，可在 .env 配置 CROSSREF_MAILTO 后重启。")
        st.caption("密钥只用于模型请求，不会写入 Agent 运行状态或导出记录。")

    # 点击运行时清空旧一轮的运行结果，避免新旧 Agent 状态混在同一详情面板。
    if run_clicked:
        st.session_state.agent_statuses = initial_statuses()
        st.session_state.agent_records = []
        st.session_state.agent_live_details = {}
        st.session_state.final_answer = ""
        st.session_state.demand_state = None
        st.session_state.review_state = None
        st.session_state.traces = []
        st.session_state.run_complete = False
        st.session_state.run_error = ""
        st.session_state.last_run_mode = (
            f"真实在线 · DeepSeek {model_name} + Crossref"
            if live_mode
            else "离线演示 · 规则 + 本地样例"
        )
        if not st.session_state.get("selected_agent"):
            st.session_state.selected_agent = "need_analyst"
            st.session_state.follow_active_agent = True
        else:
            st.session_state.follow_active_agent = False

    # 第二阶段：从 session_state 取出页面重绘后仍需保留的数据，并复制状态字典，
    # 以便本轮流式执行时更新而不意外修改旧对象。
    statuses = copy.deepcopy(st.session_state.get("agent_statuses", initial_statuses()))
    live_details = st.session_state.get("agent_live_details", {})
    records = st.session_state.get("agent_records", [])
    records_by_node = record_map(records)
    selected_agent = st.session_state.get("selected_agent")
    follow_active_agent = bool(st.session_state.get("follow_active_agent", False))

    # 有选中的 Agent 时给右侧详情抽屉预留布局；否则主内容区使用完整宽度。
    if selected_agent:
        content_col, detail_col = st.columns([0.68, 0.32], gap="large")
    else:
        content_col = st.container()
        detail_col = None

    with content_col:
        st.subheader("智能体状态")
        card_columns = st.columns(2 if selected_agent else 3, gap="small")
        status_slots: dict[str, Any] = {}
        # 每张卡片的按钮本身就是详情入口，点击后固定选中该 Agent 并触发重绘。
        for index, (node, label, role) in enumerate(AGENTS):
            with card_columns[index % len(card_columns)]:
                with st.container(key=f"agent-card-{node}"):
                    if st.button(
                        agent_card_label(node, label, role),
                        key=f"select_agent_{node}",
                        use_container_width=True,
                    ):
                        st.session_state.selected_agent = node
                        st.session_state.follow_active_agent = False
                        st.rerun()
                    status_slots[node] = st.empty()
                    status_slots[node].markdown(
                        status_markup(statuses.get(node, {"phase": "waiting", "detail": "等待任务启动"})),
                        unsafe_allow_html=True,
                    )

        if st.session_state.get("run_error"):
            st.error(st.session_state.run_error)
        elif not st.session_state.get("run_complete") and not run_clicked:
            st.caption("点击智能体卡片查看详情；运行时面板会跟随当前执行的 Agent。")
        elif st.session_state.get("run_complete"):
            saved_mode = st.session_state.get("last_run_mode", "")
            if saved_mode:
                st.caption(f"最近运行：{saved_mode}")
            st.success("推荐任务已完成。")
            render_saved_run()

    detail_slots: dict[str, Any] | None = None
    if detail_col is not None and selected_agent:
        detail_slots = render_detail_panel(
            detail_col,
            selected_agent,
            statuses,
            live_details,
            records_by_node,
        )

    # 只有点击“开始协作推荐”的这一轮才向下启动图执行；普通点击卡片/折叠面板
    # 只重绘 UI，不会重复调用推荐流程。
    if not run_clicked:
        return
    if not task.strip():
        st.session_state.run_error = "请先输入推荐任务。"
        st.rerun()
    if live_mode and not api_key.strip():
        st.session_state.run_error = "当前未检测到 DeepSeek API Key。请在侧栏输入，或在本地 .env 中设置。"
        st.rerun()

    try:
        llm = (
            make_llm(
                api_key=api_key.strip(),
                model_name=model_name,
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            )
            if live_mode
            else None
        )
        live_profile = (
            {
                "profile_id": "current-session",
                "read_item_ids": parse_read_dois(read_dois_text),
                "interest_keywords": [],
                "source": "本次页面输入（已读 DOI）",
            }
            if live_mode
            else None
        )
        graph = build_graph(
            llm=llm,
            catalog_source="crossref" if live_mode else "demo",
            reader_profile=live_profile,
            require_llm=live_mode,
        )
    except Exception as exc:
        st.session_state.run_error = f"初始化失败（{type(exc).__name__}）。请检查依赖和 DeepSeek 配置。"
        st.rerun()

    # 第三阶段：准备流式输出占位区。Agent 卡片与详情面板已先渲染，后续收到事件
    # 时可原位更新，而不需要等整个推荐工作流结束才显示进度。
    answer_slot = st.empty()
    trace_slot = st.empty()
    demand_slot = st.empty()
    review_slot = st.empty()
    traces: list[dict[str, str]] = []
    records_by_node = {}
    try:
        with st.status("多智能体任务执行中…", expanded=True) as run_status:
            # custom：节点主动发出的实时进度/输入输出快照；updates：节点返回并
            # 合并进 LangGraph 共享状态的正式业务结果。两路数据用途不同，分别处理。
            for part in graph.stream(
                {"task": task.strip(), "agent_trace": [], "agent_records": []},
                stream_mode=["custom", "updates"],
                version="v2",
            ):
                if part["type"] == "custom":
                    # 实时事件用于更新 Agent 状态和运行中详情；自动跟随只在用户没有
                    # 手动锁定某张卡片时生效，避免面板跳离用户正在查看的 Agent。
                    event = part["data"]
                    node = event.get("node")
                    if node in statuses:
                        if event.get("phase") == "running" and follow_active_agent:
                            selected_agent = node
                            st.session_state.selected_agent = node
                        statuses[node] = {
                            "phase": event.get("phase", "running"),
                            "detail": event.get("detail", ""),
                        }
                        st.session_state.agent_statuses = copy.deepcopy(statuses)
                        status_slots[node].markdown(
                            status_markup(statuses[node]), unsafe_allow_html=True
                        )
                        agent_data = event.get("agent_data")
                        if agent_data:
                            live_details[node] = agent_data
                            st.session_state.agent_live_details = copy.deepcopy(live_details)
                        if detail_slots and node == selected_agent:
                            update_detail_panel(
                                detail_slots,
                                selected_agent,
                                statuses,
                                live_details,
                                records_by_node,
                            )

                elif part["type"] == "updates":
                    # 节点完成后处理正式状态增量：累积协作轨迹/运行记录，并分别
                    # 刷新结构化需求、质量审核和最终推荐答复。
                    for node, update in part["data"].items():
                        if "agent_trace" in update:
                            traces.extend(update["agent_trace"])
                            st.session_state.traces = copy.deepcopy(traces)
                        if "agent_records" in update:
                            for record in update["agent_records"]:
                                records_by_node[record["node"]] = record
                            st.session_state.agent_records = [
                                records_by_node[key]
                                for key, _, _ in AGENTS
                                if key in records_by_node
                            ]
                            if detail_slots and selected_agent in records_by_node:
                                update_detail_panel(
                                    detail_slots,
                                    selected_agent,
                                    statuses,
                                    live_details,
                                    records_by_node,
                                )
                        if "demand" in update:
                            st.session_state.demand_state = update["demand"]
                            demand_slot.json(update["demand"])
                        if "review" in update:
                            st.session_state.review_state = update["review"]
                            review_slot.json(update["review"])
                        if "final_answer" in update:
                            st.session_state.final_answer = update["final_answer"]
                            answer_slot.markdown(update["final_answer"])

                if traces:
                    trace_slot.markdown(
                        "\n".join(
                            f"- **{item['agent']}**：{item['action']}；{item['result']}。"
                            for item in traces
                        )
                    )

            run_status.update(label="协作流程完成", state="complete", expanded=False)
            st.session_state.run_complete = True
            st.session_state.follow_active_agent = False
    except Exception as exc:
        # 将中断时仍标记为 running 的节点改为 error，并保存已经取得的部分记录，
        # 让页面能呈现失败位置和已有上下文；错误信息不包含密钥。
        for node, state in statuses.items():
            if state["phase"] == "running":
                statuses[node] = {
                    "phase": "error",
                    "detail": "流程中断，请检查连接和模型配置",
                }
        st.session_state.agent_statuses = copy.deepcopy(statuses)
        st.session_state.agent_records = [
            records_by_node[key] for key, _, _ in AGENTS if key in records_by_node
        ]
        safe_detail = getattr(exc, "user_message", None)
        st.session_state.run_error = (
            f"{safe_detail} 密钥内容不会显示。"
            if safe_detail
            else f"流程执行失败（{type(exc).__name__}）。请检查网络和 DeepSeek 配置；密钥内容不会显示。"
        )
        st.session_state.follow_active_agent = False
        st.rerun()

    st.rerun()


if __name__ == "__main__":
    run_app()
