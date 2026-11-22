"""Crossref 学术元数据 API 适配器。

该模块将 Crossref 返回的真实作品元数据统一为推荐流程使用的资源字段。
Crossref REST API 可公开检索，无需 API Key；可通过 ``CROSSREF_MAILTO``
提供联系邮箱以遵循其 polite pool 使用建议。接口只提供书目元数据，
并不保证全文开放或可下载。
"""

from __future__ import annotations

import html
import json
import os
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class CrossrefAPIError(RuntimeError):
    """可安全展示给用户的 Crossref 检索错误。"""

    def __init__(self, message: str, user_message: str):
        super().__init__(message)
        self.user_message = user_message


def _first_text(value: Any) -> str:
    """Crossref 的标题、期刊名等字段通常是字符串数组，取首个非空值。"""

    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").strip()


def _publication_year(work: dict[str, Any]) -> int | None:
    """从 Crossref 的多种日期字段中取可用年份。"""

    for field in ("published", "published-online", "published-print", "issued", "created"):
        date_parts = (work.get(field) or {}).get("date-parts", [])
        try:
            year = int(date_parts[0][0])
        except (IndexError, TypeError, ValueError):
            continue
        if 1000 <= year <= 9999:
            return year
    return None


def _authors(work: dict[str, Any]) -> str:
    """将 Crossref 作者字段转为便于展示的姓名字符串。"""

    names = []
    for author in work.get("author", []):
        name = " ".join(
            part.strip()
            for part in (author.get("given", ""), author.get("family", ""))
            if part and part.strip()
        )
        if name:
            names.append(name)
    if not names:
        return ""
    return ", ".join(names[:6]) + (" 等" if len(names) > 6 else "")


def _abstract(work: dict[str, Any]) -> str:
    """清除 Crossref 摘要中的 JATS/HTML 标签并规整空白。"""

    raw = str(work.get("abstract", "") or "")
    clean = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    return re.sub(r"\s+", " ", clean).strip()


def _language(work: dict[str, Any], title: str) -> tuple[str, str]:
    """优先使用元数据语言；缺失时基于标题字符做启发式推断并标明依据。"""

    raw = str(work.get("language", "") or "").casefold()
    if raw.startswith(("zh", "chi", "zho")):
        return "中文", "Crossref语言字段"
    if raw.startswith(("en", "eng")):
        return "英文", "Crossref语言字段"
    if re.search(r"[\u3400-\u9fff]", title):
        return "中文", "根据标题字符推断"
    if re.search(r"[A-Za-z]", title):
        return "英文", "根据标题字符推断"
    return "未知", "元数据未提供"


def _open_access_license(work: dict[str, Any]) -> tuple[bool | None, str | None]:
    """只把明确的 CC BY/CC0 许可标为开放；其他情况保留为未知。"""

    licenses = work.get("license", []) or []
    urls = [str(item.get("URL", "")) for item in licenses if item.get("URL")]
    for url in urls:
        lower_url = url.casefold()
        if "creativecommons.org/publicdomain/zero" in lower_url:
            return True, url
        if re.search(r"creativecommons\.org/licenses/by(?:-sa)?/", lower_url):
            return True, url
    return None, urls[0] if urls else None


def normalize_work(work: dict[str, Any]) -> dict[str, Any] | None:
    """把 Crossref 原始记录映射为系统统一资源结构；缺标题的记录会被跳过。"""

    title = _first_text(work.get("title"))
    if not title:
        return None

    doi = str(work.get("DOI", "") or "").strip()
    doi_id = doi.casefold()
    item_url = str(work.get("URL", "") or "").strip()
    source_url = f"https://doi.org/{doi}" if doi else item_url
    language, language_basis = _language(work, title)
    open_access, access_license = _open_access_license(work)
    subjects = [str(value) for value in (work.get("subject", []) or []) if value]
    container = _first_text(work.get("container-title"))
    work_type = str(work.get("type", "") or "")
    category_names = {
        "journal-article": "期刊论文",
        "proceedings-article": "会议论文",
        "book-chapter": "图书章节",
        "posted-content": "预印本/在线稿件",
        "report": "研究报告",
        "dissertation": "学位论文",
    }
    item_id = f"doi:{doi_id}" if doi_id else f"crossref:{item_url or title.casefold()}"

    return {
        "id": item_id,
        "title": title,
        "creator": _authors(work),
        "year": _publication_year(work),
        "language": language,
        "language_basis": language_basis,
        "keywords": subjects,
        "abstract": _abstract(work),
        "open_access": open_access,
        "access_license": access_license,
        "category": category_names.get(work_type, work_type or "学术资源"),
        "container_title": container,
        "publisher": str(work.get("publisher", "") or ""),
        "doi": doi,
        "url": source_url,
        "source": "Crossref REST API（实时元数据）",
        "crossref_type": work_type,
    }


def search_crossref(
    query: str,
    rows: int = 15,
    timeout_seconds: float = 10,
) -> list[dict[str, Any]]:
    """调用 Crossref Works 检索端点并返回统一格式的真实元数据。

    函数不回退到本地样例数据。请求失败时抛出带安全提示的异常，由上层显示
    错误并让用户决定是否切换到离线演示。
    """

    query = query.strip()
    if not query:
        return []

    params: dict[str, str] = {
        "query.bibliographic": query,
        "rows": str(max(1, min(rows, 50))),
    }
    mailto = os.getenv("CROSSREF_MAILTO", "").strip()
    if mailto:
        params["mailto"] = mailto
    request = Request(
        f"https://api.crossref.org/works?{urlencode(params)}",
        headers={
            "Accept": "application/json",
            "User-Agent": "SmartLibraryRecommendationDemo/0.1",
        },
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 429:
            raise CrossrefAPIError(
                "Crossref HTTP 429",
                "Crossref 当前限制请求频率（HTTP 429）。请稍等后重试；频繁使用时可在 .env 配置 CROSSREF_MAILTO=你的联系邮箱。",
            ) from None
        raise CrossrefAPIError(
            f"Crossref HTTP {exc.code}",
            f"Crossref 学术资源接口返回 HTTP {exc.code}，请稍后重试。",
        ) from None
    except (URLError, TimeoutError, OSError):
        raise CrossrefAPIError(
            "Crossref connection failed",
            "暂时无法连接 Crossref 学术资源接口，请检查网络后重试。",
        ) from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise CrossrefAPIError(
            "Crossref returned invalid JSON",
            "Crossref 返回的数据格式异常，请稍后重试。",
        ) from None

    raw_items = payload.get("message", {}).get("items", [])
    normalized = [normalize_work(item) for item in raw_items]
    return [item for item in normalized if item is not None]
