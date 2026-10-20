"""混合检索：稠密(DashScope qwen3.7-text-embedding + Milvus) + 关键词(BM25)
→ RRF 融合(score = Σ 1/(60+rank)) → Reranker 重排 → 父子映射返回 top5 父文档。

父子文档策略：父块（大）存 Postgres 供 LLM 完整上下文；子块（小）向量存 Milvus、
全文进 BM25。分块为结构感知自定义切分：标题开新块、正文累积切块、表格独立成块并
附着题注（HTML 表转纯文本、大表按行切分重复表头），保证表格数据可检索。

依赖服务：DashScope（DASHSCOPE_API_KEY）、Milvus standalone（http://localhost:19530，
Docker 部署；连接类错误自动重连并尝试拉起容器）、Postgres
（postgresql://agent:agent@localhost:5432/agent_docs）。可用 MILVUS_URI / PG_URI 覆盖。
"""

import hashlib
import html
import os
import re
import threading
from typing import Any, Optional

import jieba
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_embeddings, get_reranker
from docstore import PostgresDocStore

# Milvus 连接配置
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
HYBRID_COLLECTION = "agent_hybrid_child"

# 分块尺寸
PARENT_CHUNK_SIZE = 1500
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 50

# Markdown 标题行（MinerU 输出 # / ## / ###，回退解析器同样输出 # 系列）
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

# 检索参数
DENSE_K = 10
SPARSE_K = 10
RRF_K = 60
RERANK_CANDIDATES = 20
FINAL_K = 5

# 全局单例
_hybrid: Optional["HybridRetriever"] = None
_INDEX_LOCK = threading.RLock()


def _stable_id(text: str) -> str:
    """根据文档正文生成稳定 doc_id（内容哈希）。"""
    return "doc-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _child_key(doc: Document) -> str:
    """子块稳定键：父 doc_id + 内容哈希。"""
    pid = doc.metadata.get("doc_id", "")
    return f"{pid}:{hashlib.sha1(doc.page_content.encode('utf-8')).hexdigest()[:12]}"


def _drop_collection_now(collection: str = HYBRID_COLLECTION) -> None:
    """删除 Milvus 集合（换 Embedding 模型后旧向量失效，须整集合重建）。

    集合不存在视为正常（首次运行或已清空）；删除失败时抛出并给出容器排查指引。
    """
    try:
        from pymilvus import MilvusClient

        client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
        try:
            if client.has_collection(collection):
                client.drop_collection(collection)
        finally:
            client.close()
    except Exception as e:
        raise RuntimeError(
            f"删除 Milvus 集合 {collection} 失败：{e}\n"
            "请确认 Milvus Standalone 容器存活：docker ps -a | findstr milvus"
        ) from e


def _jieba_tokenize(text: str) -> list[str]:
    """BM25 分词：中文走 jieba，数字/字母整体保留。

    关键：不把 "表11" 拆成 "表1" 与 "1"，避免查询"表11"时误命中"表1"；
    额外生成 "表11" / "table11" 组合 token，精确区分图表编号。
    """
    tokens: list[str] = []
    # 按 中文串 / 数字字母串 分段，数字字母串保持整体
    for seg in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_.\-]+", text):
        if not seg:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", seg):
            tokens.extend(
                t for t in jieba.cut_for_search(seg) if len(t.strip()) > 1
            )
        else:
            tokens.append(seg.lower())
    # 图表编号组合 token：表11 / Table11 / 表11.2
    for m in re.finditer(r"(?:表|表格|Table|TABLE)\s*\d+(?:[.\-]\d+)*", text):
        tokens.append(re.sub(r"\s+", "", m.group(0)).lower())
    return tokens


# 表格题注行：表 11：... / Table 11: ...（数字后须为冒号或行尾，避免误吞正文引用句）
# 表格/图编号部分：阿拉伯数字（11, 11.2, 3-1）或罗马数字（XIII）
#    部分论文（如 IEEE 版式）用罗马数字编号，只认 \d+ 会漏掉 "TABLE XIII"，
#    导致题注不挂表、检索"表 XIII"命中不了。
_NUM_PATTERN = r"(?:\d+(?:[.\-]\d+)*|[IVXLCDM]+(?:[.\-][IVXLCDM]+)?)"

_TABLE_CAPTION_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:表|表格|Table|TABLE)\s*"
    + _NUM_PATTERN
    + r"\s*(?:\*\*)?\s*(?:[:：.]|$)"
)


def _is_caption_line(line: str) -> bool:
    """判断一行是否为表格题注（表 11：... / Table 11: ... / TABLE XIII ...）。"""
    s = line.strip()
    if not s:
        return False
    if _TABLE_CAPTION_RE.match(s):
        return True
    # 无分隔符题注（"表 11 性能对比"）：仅接受短行，避免误判正文引用句
    return len(s) <= 30 and bool(
        re.match(
            r"^\s*(?:\*\*)?\s*(?:表|表格|Table|TABLE)\s*" + _NUM_PATTERN, s
        )
    )


def _caption_from_text(text: str, adjacent_to_table: bool = False) -> str:
    """从文本块中提取表格题注行（题注通常独立成行）。

    邻近感知：当该文本块紧邻表格（adjacent_to_table=True）时，松弛守卫。
    理由：无冒号的长题注（如 "TABLE XIII PSNR AND BA COMPARISONS FOR
    JPEG COMPRESSION"）不满足 _is_caption_line 的短行/分隔符要求，但仍
    是题注；此时"行是否以 表/Table+编号 开头" + "前邻是表格"两项同时成立
    已是强信号，不会误吞正文里的 "Table 4" 引用句（前者要求行首是关键字）。
    """
    for line in text.split("\n"):
        s = line.strip()
        if not s or s == "**":
            continue
        if _is_caption_line(s):
            return s.strip("**").strip()
        if adjacent_to_table and re.match(
            r"^(?:表|表格|Table|TABLE)\s*" + _NUM_PATTERN, s
        ):
            return s.strip("**").strip()
    return ""


def _html_table_to_text(table_html: str) -> str:
    """HTML 表格转纯文本：每行 = 单元格以 | 分隔。

    MinerU 输出的表格为 HTML，若不转纯文本，向量化/BM25 只会索引到
    标签噪音，表格数据无法被检索到，喂给 LLM 也不可读。

    针对复杂表格/OCR 扫描件：
    - 块级标签（<br>、单元格内换行）统一转空格，保证"一行一单元格"不跨行错乱；
    - HTML 实体（&nbsp;、&amp; 等）解码还原真实内容，避免乱码残留；
    - 合并单元格（colspan/rowspan）内容保留一次（不做列展开），空单元格
      保留空位（| 数量不变），维持列对齐可读。
    """
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.S)
    lines = []
    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.S)
        cleaned = []
        for c in cells:
            c = re.sub(r"<br\s*/?>", " ", c, flags=re.I)      # 单元格内换行 -> 空格
            c = re.sub(r"</?(?:td|th|tr|table)\b[^>]*>", " ", c, flags=re.I)
            c = re.sub(r"<[^>]+>", " ", c)                    # 其余标签 -> 空格
            c = html.unescape(c)                              # HTML 实体解码
            c = re.sub(r"\s+", " ", c).strip()
            cleaned.append(c)                                 # 空单元格也保留占位
        if cleaned:
            lines.append(" | ".join(cleaned))
    return "\n".join(lines)


# ---- 表号归一化：把 表一/表1/table1/TABLE I 统一成阿拉伯编号 ----
_CN_NUM = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _cn_numeral_to_int(s: str) -> int | None:
    """汉字数字 -> int（支持 一~十九、二十、二十三 等）。"""
    if s in _CN_NUM:
        return _CN_NUM[s]
    if s.startswith("十") and len(s) > 1 and s[1] in _CN_NUM:
        return 10 + _CN_NUM[s[1]]
    parts = s.split("十")
    if len(parts) == 2:
        lo = _CN_NUM.get(parts[0], 1) * 10 if parts[0] else 10
        hi = _CN_NUM.get(parts[1], 0) if parts[1] else 0
        return lo + hi
    return None


def _roman_to_int(s: str) -> int | None:
    """罗马数字 -> int（I/II/XIII 等）。"""
    if not s:
        return None
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        v = _ROMAN.get(ch, 0)
        if v == 0:
            return None
        total += -v if v < prev else v
        prev = v
    return total if total > 0 else None


def _extract_table_number(text: str) -> int | None:
    """从文本中提取“表格编号”，返回归一化后的阿拉伯编号；无则 None。

    同时识别 表1 / 表 一 / Table 1 / TABLE I / tableXIII 等写法，便于
    检索把 表一、表1、table1、TABLE I 视为同一号表格。
    """
    if not text:
        return None
    compact = re.sub(r"\s+", " ", text)

    # 中文或阿拉伯：表 1 / 表一 / 表格 十二
    m = re.search(r"(?:表|表格)\s*([一二两三四五六七八九十百]+|\d+)", compact)
    if m:
        num = m.group(1)
        return int(num) if num.isdigit() else _cn_numeral_to_int(num)

    # 英文：table 1 / Table XIII（允许 table1 无空格）
    m = re.search(r"table\s*([IVXLCDM]+|\d+)", compact, re.I)
    if m:
        num = m.group(1)
        return int(num) if num.isdigit() else _roman_to_int(num)
    return None


# 意图门控：出现在查询里但不代表“点名要找这张表”的填充词（表号前置时剔除）
_TABLE_LOOKUP_FILLER = set("的的是所述和其该个第号吗呢啥为因由在及与中里对从向以等将被把又或这那")


def _is_table_lookup_query(query: str) -> bool:
    """判断查询是否为“点名查表”（应把表格硬前置）。

    规则：去除查询中的表号引用段后，剩余“有效字符”（去掉填充词/标点后）若
    很少（<= 8 个），说明查询基本就是在指这张表 -> 硬前置；反之是长篇顺带提
    编号（如“如表1所述，由于...”），不做硬前置（仍保留召回兜底），交给语义排序。
    """
    if not query:
        return False
    compact = re.sub(r"\s+", "", query)

    m = re.search(r"(?:表|表格)\s*([一二两三四五六七八九十百]+|\d+)", compact)
    if not m:
        m = re.search(r"[\u4e00-\u9fffA-Za-z]*?table\s*([ivxlcdm]+|\d+)", compact, re.I)
    if not m:
        return False
    leftover = compact[: m.start()] + compact[m.end():]
    meaningful = [
        c
        for c in leftover
        if c not in _TABLE_LOOKUP_FILLER and not c.isspace()
        and c not in "?:：，,。.？！?;；"
    ]
    return len(meaningful) <= 8


def _split_raw_blocks(md_text: str) -> list[tuple[str, str]]:
    """把 Markdown 按行切分为原始块：(类型, 内容)。

    类型：heading（# 标题行）、table_html（<table> 整表）、
    table_md（连续 | 行）、text（连续普通文本）。
    """
    lines = md_text.split("\n")
    blocks: list[tuple[str, str]] = []
    i, n = 0, len(lines)
    while i < n:
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("<table"):
            # MinerU 常把整张表输出为单行 <table>...</table>（含闭合标签在同一行）。
            # 若当前行已含 </table> 直接成块，避免 while 从下一行才开始找闭合标签，
            # 把后续表间正文/下一张表一并吞进同一块，导致表格合并、表间内容丢失。
            if "</table>" in lines[i]:
                blocks.append(("table_html", lines[i]))
                i += 1
                continue
            buf = [lines[i]]
            i += 1
            while i < n and "</table>" not in lines[i]:
                buf.append(lines[i])
                i += 1
            if i < n:  # 收尾的 </table>
                buf.append(lines[i])
                i += 1
            blocks.append(("table_html", "\n".join(buf)))
            continue
        if stripped.startswith("|"):
            buf = [lines[i]]
            i += 1
            while i < n and lines[i].strip().startswith("|"):
                buf.append(lines[i])
                i += 1
            blocks.append(("table_md", "\n".join(buf)))
            continue
        m = _HEADING_RE.match(stripped)
        if m:
            blocks.append(("heading", lines[i].rstrip()))
            i += 1
            continue
        buf = [lines[i]]
        i += 1
        while i < n:
            s = lines[i].strip()
            if not s or s.startswith("<table") or s.startswith("|") \
                    or _HEADING_RE.match(s):
                break
            buf.append(lines[i])
            i += 1
        blocks.append(("text", "\n".join(buf)))
    return blocks


# Milvus 连接类错误关键词：连接被拒/进程不可达/超时等（大小写不敏感匹配）
_MILVUS_CONN_ERR_KEYWORDS = (
    "connection refused",
    "connectex",
    "failed to connect",
    "connection reset",
    "connection closed",
    "timed out",
    "timeout",
    "unavailable",
    "10061",  # WinError 10061 目标计算机积极拒绝
    "10054",  # WinError 10054 远程主机强迫关闭连接
    "eof occurred",
)


def _is_milvus_connection_error(exc: Exception) -> bool:
    """判断异常是否为 Milvus 连接类错误（网络不可达/服务未启动等）。"""
    msg = str(exc).lower()
    # 排除 Embedding 服务错误：DashScope 调用失败（配额用尽/密钥无效/请求超时）
    # 的消息同样可能含 "timeout"/"unavailable" 等字样，若不排除会被误判为
    # Milvus 连接错误，触发无意义的 Milvus 重连重试
    if _is_embedding_service_error(exc):
        return False
    return any(k in msg for k in _MILVUS_CONN_ERR_KEYWORDS)


def _is_embedding_service_error(exc: Exception) -> bool:
    """判断异常是否为 Embedding 服务错误（DashScope 调用失败等）。

    覆盖 DashScope 托管 API 的典型错误特征：
    - 403 配额用尽（AllocationQuota.FreeTierOnly / Free quota exhausted）；
    - 密钥无效（InvalidApiKey / Authentication）或参数错误（InvalidParameter）；
    - 调用返回体常含 request_id / status_code 等字段。
    """
    msg = str(exc).lower()
    return (
        "dashscope" in msg
        or "request_id" in msg
        or "allocationquota" in msg
        or "free quota" in msg
        or "invalidapikey" in msg
        or "invalidparameter" in msg
    )


class HybridRetriever:
    """混合检索器：稠密(DashScope qwen3.7-text-embedding) + 关键词(BM25) + RRF + DashScope Reranker。"""

    def __init__(self) -> None:
        import time

        self.embeddings = get_embeddings()
        self._connect_milvus()
        self.docstore = PostgresDocStore(table="parent_docs_hybrid")
        # Reranker 重排器（默认 DashScope gte-rerank-v2 API）
        self.reranker = get_reranker()
        # 父块由 _split_to_parents 做结构感知自定义切分，不再使用通用切分器
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP
        )
        self._children: list[Document] = []
        self._bm25 = None
        # 表号索引：归一化表号 -> 该号表格父块 doc_id 列表（用于检索时优先排表格）
        self._table_by_num: dict[int, list[str]] = {}
        t0 = time.perf_counter()
        self._init_children_from_postgres()
        print(
            f"[retrieval] 检索器初始化耗时 {time.perf_counter() - t0:.1f}s"
            "（首次检索会包含此耗时）"
        )

    # ---- 分块：Markdown 感知 ----
    def _connect_milvus(self, attempts: int = 3) -> None:
        """建立 Milvus 连接，失败自动重试。

        项目使用 Docker 部署的 Milvus Standalone（稳定的 19530 端口），
        容器停止时会出现"进程崩溃、端口消失"类连接错误。重试仍失败时
        给出容器检查/启动指引，避免用户手动敲 SQL 排查。
        """
        import time

        last_exc: Exception | None = None
        for i in range(1, attempts + 1):
            try:
                self.vectorstore = Milvus(
                    embedding_function=self.embeddings,
                    connection_args={"uri": MILVUS_URI, "token": MILVUS_TOKEN},
                    collection_name=HYBRID_COLLECTION,
                    drop_old=False,
                    # 元数据全部存入动态 JSON 字段：不同块类型的 metadata key
                    # 不一致（如文本块有 has_title，表格块没有），若按默认固定
                    # schema 建表，后插入的块会因缺少字段报 DataNotMatchException
                    enable_dynamic_field=True,
                )
                return
            except Exception as e:
                last_exc = e
                if i < attempts:
                    wait = 2 * i
                    print(
                        f"[milvus] 连接失败（第 {i}/{attempts} 次），{wait}s 后重试：{e}"
                    )
                    # 首次失败即尝试自动拉起已停止的 Milvus 容器，让重试能恢复
                    if i == 1:
                        started = _try_start_milvus_container()
                        if started:
                            print(
                                "[milvus] 检测到容器未运行，已自动启动："
                                + ", ".join(started)
                                + "，等待就绪...",
                                flush=True,
                            )
                    time.sleep(wait)
        raise RuntimeError(
            f"Milvus 连接失败（已重试 {attempts} 次）：{last_exc}\n"
            f"目标地址：{MILVUS_URI}\n"
            "排查指引（Milvus Standalone 由 Docker 提供）：\n"
            "  docker ps -a | findstr milvus   # 查看容器状态\n"
            "  docker start <milvus容器名>      # 启动已停止的容器\n"
            "  # 若未部署：docker compose up -d（Milvus standalone 编排文件）"
        ) from last_exc

    def _drop_collection(self) -> None:
        """删除 Milvus 子块集合（换 Embedding 模型后向量维度/空间均变，须整集合重建）。"""
        _drop_collection_now()

    def _add_texts_with_retry(
        self, texts: list[str], metadatas: list[dict], attempts: int = 3
    ) -> None:
        """写入 Milvus 子块，连接类错误时自动重连并重试。

        捕获连接错误后重建 Milvus 客户端（服务进程被重启/恢复后端口
        回归即可续写）；非连接类错误直接抛出，不重试。
        """
        import time

        last_exc: Exception | None = None
        for i in range(1, attempts + 1):
            try:
                self.vectorstore.add_texts(texts, metadatas=metadatas)
                return
            except Exception as e:
                # Embedding 服务错误（DashScope 配额/密钥/参数等）：与 Milvus 无关，
                # 重连重试无意义，直接给出排查指引
                if _is_embedding_service_error(e):
                    raise RuntimeError(
                        f"Embedding 服务调用失败（与 Milvus 无关）：{e}\n"
                        "排查指引（DashScope qwen3.7-text-embedding）：\n"
                        "  1. 检查 DASHSCOPE_API_KEY 是否配置且有效；\n"
                        "  2. 若报 403 配额错误（AllocationQuota.FreeTierOnly），\n"
                        "     请到百炼控制台充值或关闭「仅使用免费额度」模式；\n"
                        "  3. 确认网络可达 dashscope.aliyuncs.com"
                    ) from e
                if not _is_milvus_connection_error(e):
                    raise  # 业务/schema 错误，重试无意义
                last_exc = e
                print(f"[milvus] 写入失败（第 {i}/{attempts} 次），尝试重连：{e}")
                self._connect_milvus()
                time.sleep(1)
        raise RuntimeError(
            f"Milvus 写入失败（已重连 {attempts} 次）：{last_exc}\n"
            "请检查 Milvus Standalone 容器是否存活：docker ps -a | findstr milvus"
        ) from last_exc

    def _split_to_parents(self, doc: Document) -> list[Document]:
        """结构感知自定义切分（边扫描边切，避免二次遍历）。

        核心改进 vs 旧版：
        1. 表格独立成块，且题注（"表 11：..."）附着到表格块——解决
           "查表11命中的是表1"的问题；
        2. HTML 表格转纯文本（单元格以 | 分隔），使表格数据可被向量化、
           BM25 检索到，而非索引成一堆标签噪音；
        3. 标题开新块、正文按行累积切块、超大表格按行切分并重复表头。
        """
        blocks = _split_raw_blocks(doc.page_content)
        # 先行扫描：为每个表格块寻找题注（"表 11：..."），并把题注从前邻
        # text 块中剥离（避免题注同时在文本块重复出现），随后传给表格块附着。
        caption_by_idx: dict[int, str] = {}
        for idx, (btype, content) in enumerate(blocks):
            if btype in ("table_html", "table_md"):
                prev = blocks[idx - 1] if idx - 1 >= 0 else None
                caption = ""
                # 题注对 HTML 表格与 Markdown 表格同等处理：若只处理 table_html，
                # 回退解析器输出的 Markdown 表格会缺失题注，查询"表 11"时
                # 命中前邻文本块而非表格块，造成表格检索张冠李戴
                if prev and prev[0] == "text":
                    caption = _caption_from_text(prev[1], adjacent_to_table=True)
                if caption:
                    # 从前邻 text 块剥离该题注行
                    lines = [
                        l for l in prev[1].split("\n")
                        if l.strip() != caption
                    ]
                    blocks[idx - 1] = ("text", "\n".join(lines))
                caption_by_idx[idx] = caption

        parents: list[Document] = []
        current_lines: list[str] = []
        current_len = 0
        current_title = ""    # 最近一次标题，作为后续块的 section 元数据
        block_has_title = False

        def flush_text() -> None:
            nonlocal current_lines, current_len, block_has_title
            if current_lines:
                meta = {
                    **doc.metadata,
                    "chunk_type": "text",
                    "has_title": block_has_title,
                }
                if current_title:
                    meta["section"] = current_title
                parents.append(
                    Document(page_content="\n".join(current_lines), metadata=meta)
                )
            current_lines = []
            current_len = 0
            block_has_title = False

        for idx, (btype, content) in enumerate(blocks):
            if btype in ("table_html", "table_md"):
                # 表格前先收尾文本块，并附带扫描阶段确定的题注
                flush_text()
                caption = caption_by_idx.get(idx, "")
                self._add_table_parents(
                    parents, content, btype, doc, current_title, caption
                )
            elif btype == "heading":
                flush_text()
                current_title = _HEADING_RE.match(content.strip()).group(2).strip()
                current_lines = [content]
                current_len = len(content.strip())
                block_has_title = True
            else:  # text：普通文本累积分块（_split_raw_blocks 已保证块内无空行）
                for line in content.split("\n"):
                    stripped = line.strip()
                    if current_lines and current_len + len(stripped) > PARENT_CHUNK_SIZE:
                        flush_text()
                    current_lines.append(line)
                    current_len += len(stripped)

        flush_text()
        return parents

    def _add_table_parents(
        self, parents: list[Document], table_text: str, kind: str,
        doc: Document, section: str, caption: str = "",
    ) -> None:
        """表格独立成块：HTML 表转纯文本，题注附着，超大表按行切分重复表头。

        - kind: "table_html"（MinerU）或 "table_md"（回退解析器）。
        - caption 为表格题注（"表 11：..."），写入元数据，查询"表 11"时能
          精确命中表格块，避免返回"表 1"。
        """
        if kind == "table_html":
            body = _html_table_to_text(table_text)
        else:
            body = table_text

        if not body.strip():
            return

        meta = {**doc.metadata, "chunk_type": "table"}
        if section:
            meta["section"] = section
        if caption:
            meta["caption"] = caption
            body = caption + "\n" + body

        if len(body) <= PARENT_CHUNK_SIZE:
            parents.append(Document(page_content=body, metadata=meta))
            return
        pieces = self._split_table_into_chunks(body, PARENT_CHUNK_SIZE)
        total = len(pieces)
        for idx, piece in enumerate(pieces):
            parents.append(
                Document(
                    page_content=piece,
                    metadata={
                        **meta,
                        "table_chunk_index": idx,
                        "total_table_chunks": total,
                    },
                )
            )

    def _split_table_into_chunks(self, table_body: str, chunk_size: int) -> list[str]:
        """表格按行切分：每块重复首行表头，保证每块结构完整可独立检索。"""
        lines = [l for l in table_body.split("\n") if l.strip()]
        if len(lines) <= 1:
            return [table_body]
        # 真实表头行 = 第一条以 | 开头的行（body 可能以题注行开头，
        # 题注只应保留在首块，后续块重复的是表头而非题注）
        head_idx = next(
            (i for i, l in enumerate(lines) if l.lstrip().startswith("|")), 0
        )
        head = lines[head_idx]
        prefix = lines[:head_idx]  # 题注等非表格行
        chunks: list[str] = []
        current = "\n".join([*prefix, head])
        cur_size = len(current)
        for row in lines[head_idx + 1:]:
            if cur_size + len(row) + 1 > chunk_size and current != head:
                chunks.append(current)
                current = head + "\n" + row
                cur_size = len(current)
            else:
                current += "\n" + row
                cur_size += len(row)
        if current != head:
            chunks.append(current)
        return chunks or [table_body]

    # ---- 初始化与同步 ----
    def _children_from_parents(self, parents: list[Document]) -> list[Document]:
        """父文档 → 子块（保留原 metadata 并补 doc_id），供向量化与 BM25 使用。"""
        children: list[Document] = []
        for p in parents:
            if p is None:
                continue
            for child in self.child_splitter.split_documents([p]):
                children.append(
                    child.__class__(
                        page_content=child.page_content,
                        metadata={
                            **child.metadata,
                            "doc_id": p.metadata.get("doc_id", ""),
                        },
                    )
                )
        return children

    def _init_children_from_postgres(self) -> None:
        """从 Postgres 读全部父文档 → 重切子块 → 重建 BM25。"""
        keys = list(self.docstore.yield_keys())
        if not keys:
            return
        parents = self.docstore.mget(keys)
        children = self._children_from_parents(parents)
        self._children = children
        self._bm25 = self._build_bm25(children)
        self._table_by_num = self._build_table_index(parents)

    def rebuild_vectors(self) -> dict[str, int]:
        """用当前 Embedding 模型重建全部子块向量（运行中换向量模型后必须执行）。

        换 Embedding 后旧向量即失效（维度与向量空间都变了），故按
        「删旧集合 → 以新模型重建集合 → 重切子块并重新向量化」处理。
        子块由 Postgres 父文档现切，不重新 MinerU 解析，父文档与 BM25 不受影响。
        返回 {parents, children}；中途失败会抛出，可再次调用本方法重试。
        """
        keys = list(self.docstore.yield_keys())
        parents = (
            [p for p in self.docstore.mget(keys) if p is not None] if keys else []
        )
        self._drop_collection()
        self._connect_milvus()   # 新集合在首次写入时按新向量维度自动创建
        children = self._children_from_parents(parents)
        self._children = children
        if children:
            self._add_texts_with_retry(
                [c.page_content for c in children],
                [c.metadata for c in children],
            )
        self._bm25 = self._build_bm25(children)
        self._table_by_num = self._build_table_index(parents)
        return {"parents": len(parents), "children": len(children)}

    def _build_table_index(self, parents: list) -> dict[int, list[str]]:
        """从父块构建表号索引：归一化表号 -> 表格父块 doc_id 列表。

        仅收录 chunk_type=table 且能解析出表号的父块；表号取题注，若无题注再
        回退到正文开头，用 _extract_table_number 归一化（表一/表1/table1/TABLE I
        归为同一编号）。
        """
        idx: dict[int, list[str]] = {}
        for p in parents:
            if p is None:
                continue
            meta = p.metadata or {}
            if meta.get("chunk_type") != "table":
                continue
            src = meta.get("caption") or (p.page_content or "")
            num = _extract_table_number(src)
            if num is None:
                continue
            doc_id = meta.get("doc_id", "")
            if doc_id:
                idx.setdefault(num, []).append(doc_id)
        return idx

    def _build_bm25(self, children: list[Document]):
        """用 jieba 分词构建 BM25 检索器。"""
        from langchain_community.retrievers import BM25Retriever

        if not children:
            return BM25Retriever.from_documents(
                [Document(page_content="")], preprocess_func=_jieba_tokenize
            )
        return BM25Retriever.from_documents(
            children, preprocess_func=_jieba_tokenize
        )

    # ---- 索引 ----
    def add_documents(self, docs: list[Document], ids: list[str]) -> int:
        """幂等地把文档加入检索库。

        去重逻辑：父块 id 格式为 {doc_id}-p{idx}，检查时按 {doc_id}-p 前缀
        查 Postgres，已存在任意父块则视为该文档已入库，跳过。

        - 每个文档先按结构感知自定义切分（标题/正文/表格）切为若干父块；
        - 父块写入 Postgres（带 doc_id）；
        - 父块切子块，子块带 doc_id 写入 Milvus + BM25。
        返回实际新增父文档数。
        """
        added = 0
        # 按前缀检查：父块 id 格式为 {doc_id}-p0, {doc_id}-p1, ...
        # 若 Postgres 中已有该前缀的 key，说明文档已入库，跳过
        new_pairs = []
        for d, i in zip(docs, ids):
            prefix = f"{i}-p"
            if not list(self.docstore.yield_keys(prefix=prefix)):
                new_pairs.append((d, i))
        if not new_pairs:
            return 0
        new_docs = [p[0] for p in new_pairs]
        new_ids = [p[1] for p in new_pairs]

        # 1) 结构感知自定义切分为父块（标题开新块、正文按行累积、表格独立）
        all_parents: list[Document] = []
        all_parent_ids: list[str] = []
        for d, i in zip(new_docs, new_ids):
            parents = self._split_to_parents(d)
            for idx, p in enumerate(parents):
                # 父块 id = 文档 id + 块序号，保证唯一
                pid = f"{i}-p{idx}"
                p.metadata["doc_id"] = pid
                p.metadata["source"] = d.metadata.get("source", "unknown")
                all_parents.append(p)
                all_parent_ids.append(pid)

        if not all_parents:
            return 0

        # 2) 写父文档到 Postgres
        pg_pairs = [
            (pid, p.__class__(page_content=p.page_content, metadata={**p.metadata}))
            for pid, p in zip(all_parent_ids, all_parents)
        ]
        self.docstore.mset(pg_pairs)

        # 3) 切子块写入 Milvus + BM25
        new_children: list[Document] = []
        texts: list[str] = []
        metadatas: list[dict] = []
        for p in all_parents:
            for child in self.child_splitter.split_documents([p]):
                c = child.__class__(
                    page_content=child.page_content,
                    metadata={**child.metadata, "doc_id": p.metadata["doc_id"]},
                )
                new_children.append(c)
                texts.append(c.page_content)
                metadatas.append(c.metadata)
        if texts:
            self._add_texts_with_retry(texts, metadatas)
        self._children.extend(new_children)
        self._bm25 = self._build_bm25(self._children)
        # 新增表格父块并入表号索引（避免入库后检索读不到新表）
        new_tables = self._build_table_index(all_parents)
        for num, ids in new_tables.items():
            existing = set(self._table_by_num.setdefault(num, []))
            for i in ids:
                if i not in existing:
                    self._table_by_num[num].append(i)
        added = len(all_parents)
        return added

    # ---- 检索 ----
    def _dense_search(self, query: str) -> list[Document]:
        """稠密检索：Milvus 向量相似度 top DENSE_K。"""
        try:
            return self.vectorstore.similarity_search(query, k=DENSE_K)
        except Exception:
            return []

    def _sparse_search(self, query: str) -> list[Document]:
        """关键词检索：BM25 top SPARSE_K。"""
        if self._bm25 is None:
            return []
        return self._bm25.invoke(query)[:SPARSE_K]

    def _rrf_fuse(
        self, dense: list[Document], sparse: list[Document]
    ) -> list[Document]:
        """RRF 倒数排名融合：score = Σ 1/(RRF_K + rank)。"""
        scores: dict[str, float] = {}
        doc_by_key: dict[str, Document] = {}
        for rank, d in enumerate(dense, start=1):
            key = _child_key(d)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            doc_by_key[key] = d
        for rank, d in enumerate(sparse, start=1):
            key = _child_key(d)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            doc_by_key.setdefault(key, d)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [doc_by_key[k] for k, _ in ranked]

    def search(self, query: str, final_k: int = FINAL_K) -> list[Document]:
        """完整混合检索管线，返回 final_k 个父文档（去重）。"""
        import time

        t0 = time.perf_counter()
        dense = self._dense_search(query)
        t1 = time.perf_counter()
        sparse = self._sparse_search(query)
        t2 = time.perf_counter()
        fused = self._rrf_fuse(dense, sparse)
        if not fused:
            return []
        candidates = fused[:RERANK_CANDIDATES]
        try:
            ranked_children = list(
                self.reranker.compress_documents(candidates, query)
            )
        except Exception:
            ranked_children = candidates
        t3 = time.perf_counter()
        # 映射回父文档并去重
        seen: set[str] = set()
        parents: list[Document] = []
        ordered_ids: list[str] = []
        for c in ranked_children:
            pid = c.metadata.get("doc_id")
            if pid and pid not in seen:
                seen.add(pid)
                ordered_ids.append(pid)
        if ordered_ids:
            got = self.docstore.mget(ordered_ids)
            for pid, p in zip(ordered_ids, got):
                if p is not None:
                    p = p.__class__(
                        page_content=p.page_content,
                        metadata={**p.metadata, "doc_id": pid},
                    )
                    parents.append(p)
                if len(parents) >= final_k:
                    break
        t4 = time.perf_counter()
        # 表号对齐（意图门控）：查询为“点名查表”（表一/表1/table1/TABLE I 归一化）
        # 则把编号匹配的表格块硬前置；仅为长篇顺带提编号时不硬排、只保底召回。
        qn = _extract_table_number(query)
        if qn is not None and qn in self._table_by_num:
            t_ids = self._table_by_num[qn]
            if _is_table_lookup_query(query):
                # 点名查表（如“表1/TABLE I/WFormer的表1”）：表格硬前置
                front: list = []
                rest: list = []
                front_ids: set[str] = set()
                for p in parents:
                    if p.metadata.get("doc_id") in t_ids:
                        front.append(p)
                        front_ids.add(p.metadata.get("doc_id"))
                    else:
                        rest.append(p)
                missing = [i for i in t_ids if i not in front_ids]
                if missing:
                    got = self.docstore.mget(missing)
                    for pid, p in zip(missing, got):
                        if p is not None:
                            front.append(
                                p.__class__(
                                    page_content=p.page_content,
                                    metadata={**p.metadata, "doc_id": pid},
                                )
                            )
                parents = (front + rest)[:final_k]
            else:
                # 顺带提编号（长篇语义查询）：不硬排，仅保底召回缺失的表块，
                # 追加到末尾防止丢失，顺序仍以 reranker 语义为准。
                have = {p.metadata.get("doc_id") for p in parents}
                missing = [i for i in t_ids if i not in have]
                if missing:
                    got = self.docstore.mget(missing)
                    for pid, p in zip(missing, got):
                        if p is not None:
                            parents.append(
                                p.__class__(
                                    page_content=p.page_content,
                                    metadata={**p.metadata, "doc_id": pid},
                                )
                            )
                parents = parents[:final_k]
        print(
            f"[retrieval] 查询耗时明细：稠密 {t1 - t0:.1f}s ｜ 稀疏 "
            f"{t2 - t1:.1f}s ｜ 重排 {t3 - t2:.1f}s ｜ 父文档映射 "
            f"{t4 - t3:.1f}s ｜ 合计 {t4 - t0:.1f}s"
        )
        return parents[:final_k]

    # ---- 基线检索（仅稠密，供评测对比） ----
    def baseline_search(self, query: str, k: int = FINAL_K) -> list[Document]:
        """基线：仅稠密检索 + 父子映射。"""
        dense = self._dense_search(query)
        if not dense:
            return []
        seen: set[str] = set()
        ordered_ids: list[str] = []
        for c in dense:
            pid = c.metadata.get("doc_id")
            if pid and pid not in seen:
                seen.add(pid)
                ordered_ids.append(pid)
        parents: list[Document] = []
        if ordered_ids:
            got = self.docstore.mget(ordered_ids)
            for pid, p in zip(ordered_ids, got):
                if p is not None:
                    parents.append(
                        p.__class__(
                            page_content=p.page_content,
                            metadata={**p.metadata, "doc_id": pid},
                        )
                    )
                if len(parents) >= k:
                    break
        return parents[:k]


def build_hybrid() -> HybridRetriever:
    """构建混合检索器单例（懒加载）。"""
    global _hybrid
    if _hybrid is not None:
        return _hybrid
    with _INDEX_LOCK:
        if _hybrid is None:
            _hybrid = HybridRetriever()
    return _hybrid


def rebuild_vector_index() -> dict[str, int]:
    """运行中更换 Embedding 模型后重建向量索引（清空检索器单例 + 重新向量化）。

    返回 {parents, children} 报告。调用方随后应重建 Agent，使检索工具绑定到
    新建的检索器；失败时抛出异常，向量索引可能处于不完整状态，可重试本函数。
    """
    global _hybrid, _search_documents
    with _INDEX_LOCK:
        _hybrid = None
        _search_documents = None   # 工具单例须重建才能指向新检索器
        # 先删旧集合：否则新检索器会连到旧维度集合，写入/检索均报维度不匹配
        _drop_collection_now()
        retriever = build_hybrid()
        return retriever.rebuild_vectors()


def vector_state() -> dict[str, Any]:
    """向量索引概况（集合是否存在 / 子块数 / 向量维度），供平台展示。"""
    out: dict[str, Any] = {
        "collection": HYBRID_COLLECTION,
        "exists": False,
        "children": 0,
        "dim": None,
    }
    try:
        from pymilvus import MilvusClient

        client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
        try:
            out["exists"] = client.has_collection(HYBRID_COLLECTION)
            if out["exists"]:
                desc = client.describe_collection(HYBRID_COLLECTION)
                for field in desc.get("fields", []):
                    dim = (field.get("params") or {}).get("dim")
                    if dim:
                        out["dim"] = int(dim)
                        break
                stats = client.get_collection_stats(HYBRID_COLLECTION)
                out["children"] = int(stats.get("row_count", 0))
        finally:
            client.close()
    except Exception as e:
        out["error"] = str(e)
    return out


def _try_start_milvus_container() -> list[str]:
    """尝试用 docker 自动拉起已停止的 Milvus 相关容器，返回已启动的容器名。

    只处理名字含 "milvus" 的容器（standalone/etcd/minio 同属一次编排，拉起核心
    容器后 MilvusClient 大概率即恢复）。docker 本身不可用（如 Docker Desktop 未
    启动）时静默返回空，交由调用方打印排查指引。
    """
    import subprocess

    def run(args, timeout: int = 10):
        try:
            return subprocess.run(
                args, capture_output=True, text=True, timeout=timeout
            )
        except Exception:
            return None

    ps = run(["docker", "ps", "-a", "--format", "{{.Names}}"], timeout=10)
    if ps is None or ps.returncode != 0:
        return []
    names = [n.strip() for n in ps.stdout.splitlines() if n.strip()]
    started: list[str] = []
    for name in names:
        if "milvus" not in name.lower():
            continue
        insp = run(["docker", "inspect", "-f", "{{.State.Status}}", name])
        if insp is None or insp.stdout.strip() == "running":
            continue  # 已运行跳过；inspect 失败（容器被删）也跳过
        if run(["docker", "start", name], timeout=30) is not None:
            started.append(name)
    return started


def _ensure_milvus_available() -> bool:
    """确认 Milvus 可达；不可达时先尝试 docker 自动拉起容器，再带退避重连。

    返回 True 表示已可用；False 表示最终仍连不上，此时已打印排查指引。
    """
    import time

    def probe() -> bool:
        try:
            from pymilvus import MilvusClient

            client = MilvusClient(uri=MILVUS_URI)
            client.close()
            return True
        except Exception:
            return False

    if probe():
        return True

    started = _try_start_milvus_container()
    if started:
        print(
            "[reset] 检测到 Milvus 容器未运行，已自动启动："
            + ", ".join(started) + "，等待就绪..."
        )
    for delay in (2, 4, 6):
        time.sleep(delay)
        if probe():
            if started:
                print("[reset] Milvus 已恢复连接。")
            return True
    print(
        "[reset] Milvus 连接失败，已中止清理"
        "（避免 Postgres/Milvus 状态不一致）。请手动处理：\n"
        "  docker ps -a | findstr milvus          # 查看容器状态\n"
        "  docker start <milvus容器名>            # 启动已停止的容器\n"
        "  # 若上面命令无法执行：请先启动 Docker Desktop\n"
        "  # 若未部署：docker compose up -d（Milvus standalone 编排文件）"
    )
    return False


def reset_index() -> dict[str, int]:
    """清空混合检索库全部存储，一条命令重置知识库。

    一次清除三类数据（解析器升级、数据污染后可直接重入库）：
    1. Postgres 父文档表：TRUNCATE 所有 parent_docs 开头的表
       （含当前使用的 parent_docs_hybrid 和历史遗留的 parent_docs），
       清除父文档 page_content + metadata；
    2. Milvus 子块集合：删除所有 agent_ 开头的集合
       （含 agent_hybrid_child），清除子块向量 + 指向父 doc_id 的引用；
    3. 本地已入库记录文件：删除项目目录下的 processed.json（若存在）。

    返回:
        清理报告 {postgres_tables, milvus_collections, record_files}，
        值为各自清理的数量。
    """
    global _hybrid
    report = {"postgres_tables": 0, "milvus_collections": 0, "record_files": 0}
    with _INDEX_LOCK:
        # 0) 前置检查：先确认 Milvus 可达再动任何数据。避免只清空 Postgres、
        #    Milvus 残留旧集合的不一致状态（旧子块仍指向已删除的父 doc_id），
        #    以及随后重入库必然失败。
        if not _ensure_milvus_available():
            return report

        # 1) Postgres：动态发现并清空所有 parent_docs 开头的表
        try:
            pg = PostgresDocStore(table="parent_docs_hybrid")
            with pg._conn.cursor() as cur:
                cur.execute(
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname = 'public' AND tablename LIKE 'parent_docs%'"
                )
                tables = [t for (t,) in cur.fetchall()]
                for t in tables:
                    cur.execute(f'TRUNCATE TABLE "{t}"')
            pg.close()
            report["postgres_tables"] = len(tables)
        except Exception as e:
            print(f"[reset] Postgres 清理失败：{e}")

        # 2) Milvus：删除所有 agent_ 开头的集合
        try:
            from pymilvus import MilvusClient

            client = MilvusClient(uri=MILVUS_URI)
            for c in client.list_collections():
                if c.startswith("agent_"):
                    client.drop_collection(c)
                    report["milvus_collections"] += 1
        except Exception as e:
            print(f"[reset] Milvus 清理失败：{e}")

        # 3) 本地已入库记录文件（若存在则删除）
        from pathlib import Path

        for name in ("processed.json",):
            p = Path(name)
            if p.exists():
                p.unlink()
                report["record_files"] += 1

        _hybrid = None
    return report


def add_documents(
    texts: list[str], metadatas: Optional[list[dict]] = None
) -> int:
    """幂等地把若干段文本加入混合检索库。"""
    docs = [
        Document(page_content=t, metadata=metadatas[i] if metadatas else {})
        for i, t in enumerate(texts)
    ]
    ids = [_stable_id(d.page_content) for d in docs]
    with _INDEX_LOCK:
        return build_hybrid().add_documents(docs, ids)


def delete_by_source(source: str) -> int:
    """按 source 删除一份文档的全部索引（三处必须同时清理）。

    换解析器重入库时，若只删 Postgres 父文档而留下 Milvus 孤儿子块，
    检索会命中孤儿子块、取不到父上下文。因此需同步清理：
    1. Postgres 父文档（parent_docs_hybrid 表，按 metadata.source 查 key）；
    2. Milvus 子块（按 doc_id 字段过滤删除，doc_id 即父块 key）；
    3. 内存 BM25 索引（移除对应子块后重建）。

    参数:
        source: 文档来源标识（通常为文件名）。
    返回:
        删除的父块数。
    """
    with _INDEX_LOCK:
        r = build_hybrid()
        # 1) 找到该 source 的全部父块 key
        keys = r.docstore.keys_by_source(source)
        if not keys:
            return 0
        # 2) 删除 Postgres 父文档
        r.docstore.mdelete(keys)
        # 3) 删除 Milvus 子块（子块 metadata 的 doc_id 即父块 key）
        try:
            r.vectorstore.delete(expr=f"doc_id in {keys}")
        except Exception as e:
            print(f"[delete] Milvus 子块删除失败（可能存在孤儿子块）：{e}")
        # 4) 重建内存 BM25 索引
        removed = set(keys)
        r._children = [
            c for c in r._children if c.metadata.get("doc_id") not in removed
        ]
        r._bm25 = r._build_bm25(r._children)
        return len(keys)


# ---- Agent 检索工具 ----

def _search(query: str, final_k: int = FINAL_K) -> list[Document]:
    """混合检索内部接口，返回 final_k 个父文档。"""
    return build_hybrid().search(query, final_k=final_k)


def auto_ingest_dataset(dataset_dir: str = "dataset", force: bool = False) -> int:
    """扫描 dataset 目录，自动解析并入库新文件（幂等）。

    去重策略：先按 source（文件名）查 Postgres 确认是否已入库。
    已入库的文件直接跳过，不触发 MinerU 解析，避免重复解析浪费时间。

    参数:
        dataset_dir: dataset 目录路径，默认为当前目录下的 dataset。
        force: 为 True 时忽略去重，强制重入库（先按 source 删旧索引再解析入库），
            用于换解析器后刷新知识库。
    返回:
        本次新增的父块总数。
    """
    from pathlib import Path

    from document_parser import SUPPORTED_SUFFIXES, parse_document
    from metadata_extractor import extract_metadata, write_review_report

    dir_path = Path(dataset_dir)
    if not dir_path.exists():
        print(f"[auto_ingest] dataset 目录不存在：{dataset_dir}")
        return 0

    files = [
        p for p in sorted(dir_path.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not files:
        print("[auto_ingest] dataset 目录中无可入库文档")
        return 0

    # 预构建检索器，用于去重检查
    retriever = build_hybrid()

    total_added = 0
    for path in files:
        try:
            source = path.name
            # 已入库且非强制模式：直接跳过，不触发 MinerU 解析
            if not force and retriever.docstore.has_source(source):
                print(f"[auto_ingest] 已入库，跳过：{path.name}")
                continue
            # 强制模式：先按 source 删旧父文档 + 旧子块，避免孤儿子块
            if force:
                deleted = delete_by_source(source)
                if deleted > 0:
                    print(f"[auto_ingest] 清除旧索引：{path.name}（{deleted} 个父块）")

            print(f"[auto_ingest] 解析文档：{path.name}")
            md, meta = parse_document(str(path))
            # 入库前管道：数据清洗 + 元数据提取（规则/LLM）+ 置信度筛选
            clean_md, file_meta, review = extract_metadata(md, meta)
            if review:
                write_review_report(source, review)   # 低置信度字段复核报告，不阻塞入库
            if not clean_md.strip():
                print(f"[auto_ingest] 文档内容为空，跳过：{path.name}")
                continue
            # 完整文档级元数据入库（source + 提取字段 + parser 继承字段）
            metas = {
                "source": source,
                **{k: v for k, v in file_meta.items() if v is not None and v != ""},
            }
            added = add_documents([clean_md], metadatas=[metas])
            if added > 0:
                print(f"[auto_ingest] 入库完成：{path.name}（新增 {added} 个父块）")
            else:
                print(f"[auto_ingest] 已入库，跳过：{path.name}")
            total_added += added
        except Exception as e:
            print(f"[auto_ingest] 处理失败 {path.name}: {e}")
    print(f"[auto_ingest] 扫描完成，共新增 {total_added} 个父块")
    return total_added


# 延迟导入 tool 装饰器，避免在模块加载时就初始化 LangChain 工具注册
def _get_search_tool():
    from langchain_core.tools import tool

    @tool
    def search_documents(query: str) -> str:
        """在知识库中检索与问题相关的文档内容。

        采用混合检索：稠密检索(DashScope qwen3.7-text-embedding + Milvus) + 关键词检索(BM25)，
        各取 top10，用 RRF 倒数排名融合，再由 DashScope Reranker 重排，返回 top5 父文档。
        当用户提问涉及已入库文献（如论文方法、实验数据、定义结论等）时调用此工具。

        提示：若问题指向具体图表或表格（如"表 11"、"Table 3"），请在 query 中带上
        完整图表编号，检索会按编号精确匹配表格题注，避免张冠李戴。

        参数:
            query: 检索关键词或问题。
        返回:
            命中的父文档内容（多条用分隔线拼接），带引用序号和来源标注。
        """
        docs = _search(query)
        if not docs:
            return "知识库中未检索到相关内容。"
        blocks = []
        for i, d in enumerate(docs, 1):
            source = d.metadata.get("source", "unknown")
            blocks.append(f"[{i}] (来源:{source})\n{d.page_content}")
        return "\n--------------------\n".join(blocks)

    return search_documents


# 模块级单例：首次访问时构建
_search_documents = None


def get_search_tool():
    """返回 search_documents 工具单例（懒加载）。"""
    global _search_documents
    if _search_documents is None:
        _search_documents = _get_search_tool()
    return _search_documents
