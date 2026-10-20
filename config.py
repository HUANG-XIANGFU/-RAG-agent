"""全局配置：LLM / Embedding / Reranker 统一走阿里云百炼，只依赖 DASHSCOPE_API_KEY。

LLM=qwen3.7-flash（OpenAI 兼容接口），Embedding=qwen3.7-text-embedding-flash，
Reranker=qwen3.7-text-rerank。报 403「Free quota exhausted」即免费额度用尽，
需到百炼控制台充值或关闭「仅使用免费额度」模式。

模型名可在运行中切换（set_models）：改写模块级模型名并清空对应单例，下次
get_llm / get_embeddings / get_reranker 即按新模型重建，无需重启进程。
"""

import os

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI

# 百炼 OpenAI 兼容接口
BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.7-flash")

# DashScope 模型
DASHSCOPE_EMBEDDING_MODEL = os.getenv(
    "DASHSCOPE_EMBEDDING_MODEL", "qwen3.7-text-embedding-flash"
)
DASHSCOPE_RERANKER_MODEL = os.getenv(
    "DASHSCOPE_RERANKER_MODEL", "qwen3.7-text-rerank"
)

# 运行时可切换的模型候选（百炼平台上同类模型很多，也可在界面上填自定义名称）
MODEL_CANDIDATES: dict[str, list[str]] = {
    "llm": [
        "qwen3.7-flash", "qwen3.8-max", "qwen3.8-flash",
        "qwen3.8-max-0902", "kimi-k3", "glm-5.3", "deepseek-v4.1-flash",
    ],
    "embedding": [
        "qwen3.7-text-embedding", "qwen3.7-text-embedding-flash",
        "text-embedding-v4", "text-embedding-v3",
    ],
    "reranker": [
        "qwen3.7-text-rerank", "gte-rerank-v2", "gte-rerank",
    ],
}

# 各 Embedding 模型的单次 input 上限（未列出的模型按保守值处理，避免 400 InvalidParameter）
_EMBED_BATCH_LIMITS: dict[str, int] = {
    "qwen3.7-text-embedding": 20,
    "qwen3.7-text-embedding-flash": 20,
    "text-embedding-v4": 10,
    "text-embedding-v3": 10,
}
_DEFAULT_EMBED_BATCH = 10

# 单例缓存
_llm: ChatOpenAI | None = None
_embeddings: Embeddings | None = None
_reranker = None


def _ensure_api_key() -> str:
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise EnvironmentError(
            "未检测到 DASHSCOPE_API_KEY 环境变量，请先配置阿里云百炼 API Key。"
        )
    return key


def get_llm() -> ChatOpenAI:
    """返回通义千问 ChatOpenAI（百炼兼容接口）单例。

    走百炼 OpenAI 兼容地址，streaming=True 流式输出，
    支持一次返回多个 tool_calls（并行工具调用）与多轮串行工具调用。
    max_retries=3：openai SDK 指数退避重试，应对网络抖动与限流；
    timeout=60：单次 API 请求超时。
    凭证仅使用 DASHSCOPE_API_KEY。
    """
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=_ensure_api_key(),
            base_url=BAILIAN_BASE_URL,
            streaming=True,
            max_retries=3,
            timeout=60,
        )
    return _llm


class _BatchLimitedEmbeddings(Embeddings):
    """对底层 Embeddings 做批量分片，满足厂商单次 input 上限。

    DashScope embedding 接口对不同模型有不同的 batch size 上限
    （qwen3.7-text-embedding 为 20，text-embedding-v3 为 10），
    而 langchain_community 的 DashScopeEmbeddings 对未知模型走
    默认 25，会触发 400 InvalidParameter。用此包装器兜底，
    无论底层库版本/模型名如何变化都能正确分片。
    """

    def __init__(self, base: Embeddings, batch_size: int) -> None:
        self._base = base
        self._batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            chunk = texts[i : i + self._batch_size]
            results.extend(self._base.embed_documents(chunk))
        return results

    def embed_query(self, text: str) -> list[float]:
        return self._base.embed_query(text)


def get_embeddings() -> Embeddings:
    """返回 DashScope Embedding 单例（仅 DASHSCOPE_API_KEY）。"""
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    from langchain_community.embeddings.dashscope import DashScopeEmbeddings

    base = DashScopeEmbeddings(
        model=DASHSCOPE_EMBEDDING_MODEL, dashscope_api_key=_ensure_api_key()
    )
    # DashScope 各 Embedding 模型单次 input 上限不同（qwen3.7-text-embedding 为 20，
    # text-embedding-v3/v4 为 10），langchain_community 的 DashScopeEmbeddings 对
    # 未知模型走默认 batch_size=25，会触发 400 InvalidParameter。按模型名取上限分片。
    limit = _EMBED_BATCH_LIMITS.get(DASHSCOPE_EMBEDDING_MODEL, _DEFAULT_EMBED_BATCH)
    _embeddings = _BatchLimitedEmbeddings(base, batch_size=limit)
    return _embeddings


def get_reranker():
    """返回 DashScope Reranker 单例（仅 DASHSCOPE_API_KEY）。"""
    global _reranker
    if _reranker is not None:
        return _reranker

    from langchain_community.document_compressors import DashScopeRerank

    _reranker = DashScopeRerank(
        model=DASHSCOPE_RERANKER_MODEL,
        top_n=20,
        dashscope_api_key=_ensure_api_key(),
    )
    return _reranker


def current_models() -> dict[str, str]:
    """当前生效的模型名（Web 平台展示用）。"""
    return {
        "llm": LLM_MODEL,
        "embedding": DASHSCOPE_EMBEDDING_MODEL,
        "reranker": DASHSCOPE_RERANKER_MODEL,
    }


def set_models(
    llm: str | None = None,
    embedding: str | None = None,
    reranker: str | None = None,
) -> dict[str, str]:
    """运行中切换模型名（无需重启），返回实际变更项。

    只改写模块级模型名并清空对应单例缓存，下次 get_llm / get_embeddings /
    get_reranker 即按新模型名重建；同时写回环境变量，便于 CLI/子进程复用。
    注意：Embedding 一换，Milvus 中既有向量即失效（维度与向量空间均变），
    必须再调用 hybrid_retriever.rebuild_vector_index() 重建向量索引。
    """
    global LLM_MODEL, DASHSCOPE_EMBEDDING_MODEL, DASHSCOPE_RERANKER_MODEL
    global _llm, _embeddings, _reranker

    changed: dict[str, str] = {}
    if llm and llm != LLM_MODEL:
        LLM_MODEL = llm
        os.environ["LLM_MODEL"] = llm
        _llm = None
        changed["llm"] = llm
    if embedding and embedding != DASHSCOPE_EMBEDDING_MODEL:
        DASHSCOPE_EMBEDDING_MODEL = embedding
        os.environ["DASHSCOPE_EMBEDDING_MODEL"] = embedding
        _embeddings = None
        changed["embedding"] = embedding
    if reranker and reranker != DASHSCOPE_RERANKER_MODEL:
        DASHSCOPE_RERANKER_MODEL = reranker
        os.environ["DASHSCOPE_RERANKER_MODEL"] = reranker
        _reranker = None
        changed["reranker"] = reranker
    return changed
