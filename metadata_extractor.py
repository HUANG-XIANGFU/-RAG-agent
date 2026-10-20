"""入库前元数据管道：清洗 → 规则/LLM 混合提取 → 置信度筛选 → 低置信度复核报告。

规则提取确定性字段（语言/年份/关键词/类型/领域），LLM 提取语义字段（标题/作者/
摘要/期刊/关键词）并给置信度；同字段取高者，低于阈值的进入复核清单。LLM 出错或
非 JSON 时静默降级，绝不阻断入库。复核清单追加写 review/review.csv，不阻塞入库。

使用：
    from metadata_extractor import extract_metadata
    clean_md, file_meta, review = extract_metadata(md, meta)   # meta 来自 parse_document
"""

import csv
import json
import os
import re
import time
from collections import Counter
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm

# ---- 元数据 Schema（字段 / 类型 / 来源 / 置信度阈值） ----
# 低于阈值 -> 进入人工复核报告。source/parser/pages 为直接继承字段，不在表内。
METADATA_SCHEMA: dict[str, dict[str, Any]] = {
    "title":    {"type": "str",  "source": "llm|rule", "threshold": 0.5},
    "authors":  {"type": "str",  "source": "llm",      "threshold": 0.6},
    "year":     {"type": "int",  "source": "rule|llm", "threshold": 0.7},
    "venue":    {"type": "str",  "source": "llm",      "threshold": 0.6},
    "keywords": {"type": "list", "source": "rule|llm", "threshold": 0.55},
    "abstract": {"type": "str",  "source": "llm",      "threshold": 0.5},
    "doc_type": {"type": "str",  "source": "rule",     "threshold": 1.0},
    "language": {"type": "str",  "source": "rule",     "threshold": 0.9},
    "domain":   {"type": "str",  "source": "rule",     "threshold": 0.75},
}

# 领域映射表：文件名或正文命中关键词 -> 领域标签（可扩展）
_DOMAIN_MAP: list[tuple[tuple[str, ...], str]] = [
    (("watermark", "watermarking", "数字水印"), "数字水印"),
    (("large language model", "llm", "大语言模型"), "大语言模型"),
    (("retrieval", "检索", "rag"), "信息检索"),
    (("knowledge graph", "知识图谱"), "知识图谱"),
    (("agent", "智能体"), "智能体"),
    (("image", "图像", "vision", "视觉"), "计算机视觉"),
]

# LLM 提取提示词：严格 JSON 输出，每字段带置信度
_LLM_PROMPT = (
    "你是学术文献元数据提取助手。请从给定的论文文本中提取以下字段并给出置信度(0-1)：\n"
    "title（标题）、authors（作者，逗号分隔）、abstract（摘要）、"
    "venue（期刊/会议名，无法确定则留空）、keywords（关键词列表）。\n"
    '仅输出 JSON，格式：{"title": "...", "title_confidence": 0.9, '
    '"authors": "...", "authors_confidence": 0.8, "abstract": "...", '
    '"abstract_confidence": 0.7, "venue": "...", "venue_confidence": 0.6, '
    '"keywords": ["..."], "keywords_confidence": 0.8}。'
    "无法确定时字段留空字符串或空列表。不要输出任何其他文字。"
)

_REVIEW_HEADER = ["source", "field", "value", "confidence", "status", "ts"]


# ---------------- 入库前数据清洗 ----------------

def _is_table_separator(line: str) -> bool:
    """判断是否为 Markdown 表格分隔线行（| --- | :---: | 等），入库前应删除。"""
    s = line.strip()
    if not s.startswith("|") or "-" not in s:
        return False
    return bool(re.fullmatch(r"\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?", s))


def clean_markdown(md: str) -> str:
    """清洗解析出的 Markdown：页眉页脚/页码/OCR 伪像/重复行/多余空行。

    顺序执行：
    1. 统一换行（\\r\\n|\\r -> \\n）
    2. 行内去不可见/乱码字符（\\uFFFD 等 OCR 伪像）、行尾去空白
    3. 删除纯符号行、纯页码行、Markdown 表格分隔线行
    4. 连续相同行去重（防 MinerU 重复输出）
    5. 短行（<=40 字符）全文重复出现 >=4 次的页眉/页脚律动行仅保留首次
    6. 行内多余空格折叠（OCR 碎空格）；折叠多余空行
    7. 分页式题注归一（Table 11 of 12 -> Table 11）
    """
    if not md:
        return ""
    text = md.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    cleaned: list[str] = []
    for raw in lines:
        line = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\uFFFD]", "", raw)  # 乱码/控制字符
        line = re.sub(r"[ \t]+$", "", line)          # 行尾去空白
        line = re.sub(r" {2,}", " ", line)           # 行内多空格折叠（OCR 碎空格）
        if not line.strip():
            cleaned.append("")
            continue
        s = line.strip()
        if re.fullmatch(r"[\W_]+", s):              # OCR 伪像：纯符号行
            continue
        if re.fullmatch(r"(?:page\s*)?\d{1,3}", s, flags=re.I):  # 纯页码行
            continue
        if _is_table_separator(s):                  # Markdown 表格分隔线行
            continue
        if cleaned and cleaned[-1] == line:         # 连续相同行去重
            continue
        cleaned.append(line)

    # 页眉/页脚律动行：短行全文重复 >=4 次且非标题/表格行 -> 仅保留首次出现
    counts = Counter(
        l.strip() for l in cleaned
        if l.strip() and len(l.strip()) <= 40
        and not l.strip().startswith(("#", "|"))
    )
    repeats = {t for t, c in counts.items() if c >= 4}
    if repeats:
        kept: set[str] = set()
        filtered: list[str] = []
        for l in cleaned:
            s = l.strip()
            if s in repeats:
                if s in kept:
                    continue
                kept.add(s)
            filtered.append(l)
        cleaned = filtered

    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)          # 折叠多余空行
    # 压缩公式间距噪音：MinerU 公式识别（Unimernet）对复杂公式间距输出大量
    # 连续的 \quad 命令（如 \begin{array} 内 \quad \quad \quad ...），
    # 每个 \quad 会撑大公式块体积、挤占分块与上下文。3 个及以上连续 \quad
    # 合并为 1 个，公式语义不变（间距是视觉排版信息，检索/问答不需要）。
    text = re.sub(r"(?:\\quad\s*){3,}", r"\\quad ", text)
    # 分页式题注归一：Table 11 of 12 / 图 11 / 12 -> Table 11 / 图 11
    text = re.sub(
        r"\b(Table|表|图)\s*(\d+(?:[.\-]\d+)*)\s*(?:of|/)\s*\d+\b",
        lambda m: f"{m.group(1)} {m.group(2)}",
        text,
        flags=re.I,
    )
    return text.strip() + "\n"


# ---------------- 规则提取（确定性字段，高置信度） ----------------

def _extract_by_rules(clean_md: str, meta: dict) -> dict[str, tuple[Any, float]]:
    """正则规则提取：语言/年份/关键词/文档类型/领域/标题回退。"""
    out: dict[str, tuple[Any, float]] = {}
    head = clean_md[:2000]

    # language：中文占比 >= 0.7 视为中文文档
    if head.strip():
        zh = len(re.findall(r"[\u4e00-\u9fff]", head))
        ratio = zh / max(len(head), 1)
        out["language"] = ("zh", 0.9) if ratio >= 0.7 else ("en", 0.9)

    # year：首屏合理年份（1950-2100），取最小
    years = [
        int(m) for m in re.findall(r"\b(?:19|20)\d{2}\b", head)
        if 1950 <= int(m) <= 2100
    ]
    if years:
        out["year"] = (min(years), 0.95)

    # keywords：Keywords:/关键词: 行拆分
    kw_lines = re.findall(
        r"(?im)^\s*(?:keywords?|关键词)\s*[:：]\s*(.+)$", clean_md[:4000]
    )
    kw: list[str] = []
    for line in kw_lines:
        for part in re.split(r"[;,；，]", line):
            p = part.strip().strip(".").strip()
            if p and p not in kw:
                kw.append(p)
    if kw:
        out["keywords"] = (kw[:20], 0.9)

    # doc_type：按文件名后缀
    suffix = os.path.splitext(str(meta.get("source", "")))[1].lower().lstrip(".")
    if suffix in ("pdf", "docx", "doc", "pptx", "ppt"):
        out["doc_type"] = (suffix, 1.0)

    # domain：文件名优先（高置信），正文命中次之
    source_hit = _match_domain(str(meta.get("source", "")))
    text_hit = _match_domain(clean_md[:3000])
    if source_hit:
        out["domain"] = (source_hit, 0.9)
    elif text_hit:
        out["domain"] = (text_hit, 0.75)

    # title 回退：文件名去后缀（低置信，供 LLM 缺失时兜底）
    stem = os.path.splitext(str(meta.get("source", "")))[0]
    if stem:
        out["title"] = (stem, 0.5)

    return out


def _match_domain(text: str) -> str:
    """在文本中查找领域映射表首个命中。"""
    lower = text.lower()
    for keys, domain in _DOMAIN_MAP:
        for k in keys:
            if k in lower:
                return domain
    return ""


# ---------------- LLM 提取（语义字段，静默降级） ----------------

def _extract_by_llm(clean_md: str, meta: dict) -> dict[str, tuple[Any, float]]:
    """LLM 提取 title/authors/abstract/venue/keywords，每字段带置信度。

    出错或输出非 JSON 时静默降级为空 dict（参考 memory._parse_facts 模式）。
    """
    sample = clean_md[:4000].strip()
    if not sample:
        return {}
    try:
        llm = get_llm()
        resp = llm.invoke(
            [
                SystemMessage(content=_LLM_PROMPT),
                HumanMessage(content=f"论文文本：\n{sample}"),
            ]
        )
    except Exception:
        return {}
    data = _parse_json_loose(getattr(resp, "content", "") or "")
    if not data:
        return {}

    out: dict[str, tuple[Any, float]] = {}
    for field in ("title", "authors", "abstract", "venue", "keywords"):
        value = data.get(field)
        conf = float(data.get(f"{field}_confidence", 0.0) or 0.0)
        conf = max(0.0, min(1.0, conf))
        if field == "abstract" and isinstance(value, str):
            value = value.strip()[:500]              # 摘要截断，控制入库体积
        if isinstance(value, list):
            value = [str(v).strip() for v in value if str(v).strip()][:20]
            if not value:
                continue
        elif isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        else:
            continue
        out[field] = (value, conf if conf > 0 else 0.5)
    return out


def _parse_json_loose(raw: str) -> Optional[dict]:
    """从 LLM 输出中提取 JSON 对象；非 JSON 返回 None。"""
    text = raw.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


# ---------------- 置信度合并与筛选 ----------------

def _merge_with_confidence(
    rule_f: dict[str, tuple[Any, float]],
    llm_f: dict[str, tuple[Any, float]],
) -> tuple[dict, list[dict]]:
    """同字段取置信度高者（平局取规则）；低于阈值进复核清单。

    返回 (final_meta, review_list)；review_list 元素含 field/value/confidence/status，
    source 由 write_review_report 补齐。
    """
    final: dict = {}
    review: list[dict] = []
    for field, spec in METADATA_SCHEMA.items():
        rv, rc = rule_f.get(field, ("", 0.0))
        lv, lc = llm_f.get(field, ("", 0.0))

        if field == "keywords":
            value, conf = _merge_keywords(rv, rc, lv, lc)
            if not value:
                continue
            final[field] = value
        else:
            if rc >= lc:
                value, conf = rv, rc
            else:
                value, conf = lv, lc
            if value in (None, "", []):
                continue
            final[field] = value

        if conf < spec["threshold"]:
            review.append(
                {
                    "field": field,
                    "value": str(value),
                    "confidence": round(float(conf), 2),
                    "status": "low",
                }
            )
    return final, review


def _merge_keywords(
    rv: Any, rc: float, lv: Any, lc: float
) -> tuple[list[str], float]:
    """规则与 LLM 关键词取并集，置信度取 max。"""
    merged: list[str] = []
    for v in (rv, lv):
        if isinstance(v, list):
            for kw in v:
                s = str(kw).strip()
                if s and s not in merged:
                    merged.append(s)
    if not merged:
        return [], 0.0
    return merged[:20], max(rc, lc)


# ---------------- 低置信度人工复核报告（不阻塞入库） ----------------

def write_review_report(source: str, review_list: list[dict]) -> None:
    """把低置信度字段追加到 review/review.csv（UTF-8-sig）。失败仅告警。"""
    if not review_list:
        return
    try:
        review_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "review"
        )
        os.makedirs(review_dir, exist_ok=True)
        csv_path = os.path.join(review_dir, "review.csv")
        is_new = not os.path.exists(csv_path)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(_REVIEW_HEADER)
            for item in review_list:
                writer.writerow(
                    [
                        source,
                        item.get("field", ""),
                        item.get("value", ""),
                        item.get("confidence", ""),
                        item.get("status", "low"),
                        ts,
                    ]
                )
        print(
            f"[metadata] 已追加 {len(review_list)} 条低置信度字段到复核报告 {csv_path}"
        )
    except Exception as e:
        print(f"[metadata] 写入复核报告失败（不阻断入库）：{e}")


# ---------------- 顶层入口 ----------------

def extract_metadata(md: str, meta: dict) -> tuple[str, dict, list[dict]]:
    """入库前完整管道：清洗 -> 规则提取 -> LLM 提取 -> 置信度合并 -> 复核清单。

    返回 (clean_md, file_meta, review_list)：
    - clean_md：清洗后的 Markdown（用于分块入库，doc_id 基于它生成）；
    - file_meta：文档级元数据（含继承的 source/parser/pages）；
    - review_list：低置信度字段清单，由调用方交给 write_review_report。
    """
    clean = clean_markdown(md)
    rule = _extract_by_rules(clean, meta)
    llm = _extract_by_llm(clean, meta)
    final, review = _merge_with_confidence(rule, llm)
    for k in ("source", "parser", "pages"):
        v = meta.get(k)
        if v is not None and v != "":
            final[k] = v
    return clean, final, review
