"""检索质量评测：对比基线（单一稠密检索）与混合检索（稠密+BM25+RRF+Reranker）。

指标（@k=5）：Recall@5、Precision@1/5、MRR、nDCG@5、Hit@5。语料 15 篇覆盖不同主题，
评测集含精确关键词/缩写/数字型查询（BM25 优势）与语义改写型查询（稠密优势），
用以体现混合检索增益。

运行：python eval_retrieval.py
"""

import math
import sys

from hybrid_retriever import build_hybrid, reset_index

# ---- 评测语料：15 篇文档，每篇有 source 标签 ----
CORPUS = [
    ("LangGraph 是 LangChain 团队推出的 agent 编排框架，基于状态图 StateGraph 构建可控、可流式、可持久化的智能体。它将节点 node 与边 edge 组合为有向图，支持条件路由、人在回路 human-in-the-loop、检查点 checkpoint 等高级特性。通过 InMemorySaver 等检查点器可实现基于内存的对话记忆，跨轮次保留上下文。", "langgraph-intro"),
    ("LangGraph 的 ReAct agent 通过 create_react_agent 预构建组件快速搭建工具调用型智能体。模型可以一次返回多个 tool_calls，ToolNode 会并行执行这些工具调用，再将结果回填给模型进行下一轮推理，直到模型认为无需再调用工具为止。", "langgraph-react"),
    ("LangGraph 的流式输出支持 stream_mode 为 messages、updates、values、debug 等多种模式，也可同时订阅多个模式。messages 模式可逐 token 输出模型文本，updates 模式则报告节点完成时的状态变化，便于实时展示工具调用过程。", "langgraph-streaming"),
    ("父子文档检索 Parent Document Retrieval 是一种两级分块策略：用小分块做精确向量检索，用大块父文档喂给 LLM 提供完整上下文。这样既保证检索精度，又避免上下文不足导致答案片面。Milvus 作为高性能向量数据库，负责存储子块向量。", "parent-doc-retrieval"),
    ("RRF 倒数排名融合 Reciprocal Rank Fusion 是一种无权重的多路检索结果融合方法。对每路检索结果的每个文档，按其排名 rank 计算 1/(k+rank) 并累加，k 通常取 60。RRF 不依赖各路检索的原始分数，只看排名，适合融合稠密检索与关键词检索。", "rrf-fusion"),
    ("稠密检索 dense retrieval 基于文本的语义向量表示，通过余弦相似度等度量找出语义相近的文档，擅长理解同义词和语义改写；关键词检索如 BM25 则基于词频和逆文档频率，擅长精确匹配专有名词、缩写、代码标识符等。两者互补，混合检索可显著提升召回。", "dense-vs-sparse"),
    ("阿里云百炼 DashScope 提供通义千问大模型与文本向量化服务。通过环境变量 DASHSCOPE_API_KEY 即可调用 qwen 系列对话模型和 Embedding 模型。在 LangChain 中可用 ChatTongyi 与 DashScopeEmbeddings 直接接入。", "bailian-intro"),
    ("百炼的 OpenAI 兼容接口地址为 https://dashscope.aliyuncs.com/compatible-mode/v1，使用 ChatOpenAI 即可接入 qwen-plus、qwen-turbo、qwen-max 等模型。该兼容接口对并行 tool_calls 和流式 streaming 输出支持完善。", "bailian-openai-compat"),
    ("百炼的 gte-rerank 模型是通义系列文本重排模型，可对一批候选文档按 query 相关度重新排序并返回 top_n。重排 rerank 在召回之后、生成之前进行，能显著提升最终送入 LLM 的文档质量。", "bailian-rerank"),
    ("Milvus 是开源的高性能向量数据库，支持十亿级向量检索。standalone 单机模式通过 Docker Compose 一键部署，包含 etcd、minio、milvus-standalone 三个容器，默认监听 19530 端口。Windows 不支持 milvus-lite 本地文件模式。", "milvus-intro"),
    ("PostgreSQL 是功能强大的开源关系型数据库，支持 JSONB、全文检索、扩展插件。本项目用其存储父子检索中的父文档，配合 upsert 主键覆盖语义实现幂等索引，进程重启后父文档不丢失。", "postgres-intro"),
    ("get_air_quality 工具通过 Open-Meteo 空气质量 API 查询城市实时 PM2.5、PM10 和美国 AQI 指数。该 API 免费且无需 API Key，通过城市经纬度坐标查询。", "tool-air-quality"),
    ("get_current_time 工具基于 Python zoneinfo 按时区返回当前时间，支持 IANA 时区名如 Asia/Shanghai、America/New_York。", "tool-current-time"),
    ("递归字符分块 RecursiveCharacterTextSplitter 优先按段落、换行、句号、空格等分隔符递归切分文本，尽量保持语义完整。它通过 chunk_size 和 chunk_overlap 控制块大小与重叠，是 RAG 中最常用的分块器。", "recursive-splitter"),
    ("BM25 是经典的概率检索模型，结合词频 TF 与逆文档频率 IDF，并引入文档长度归一化，对查询词与文档的相关性打分。它对中文需配合分词器如 jieba，按词切分后建立倒排索引。", "bm25-intro"),
    # 语义近似的干扰文档：让稠密检索在精确关键词查询上易跑偏，凸显 BM25+Reranker 价值
    ("向量数据库是存储与检索高维向量的系统，广泛用于推荐、搜索与 RAG。市面上有 Milvus、Faiss、Qdrant、Weaviate 等多种选择，它们在索引算法、部署方式上各有差异。选择时需考虑规模、延迟和生态。", "distractor-vectordb-general"),
    ("大模型服务的鉴权通常依赖 API Key 机制，客户端在请求头中携带密钥以完成身份认证。不同厂商的密钥命名与管理方式不一，需妥善保管避免泄露。", "distractor-apikey-general"),
    ("构建智能体时常用 ReAct 范式让模型边思考边行动，框架会预置一些现成组件降低开发门槛。开源生态中有多种 agent 框架可选，设计思路相近但接口各异。", "distractor-react-general"),
    ("流式输出是把生成内容分块逐步返回给前端的技术，能改善首字延迟体验。许多大模型 SDK 都提供流式接口，实现细节涉及分块、缓冲与背压处理。", "distractor-streaming-general"),
    ("文本重排是检索增强生成中的关键环节，通过对召回候选二次排序提升相关性。学术界有 ColBERT、Cross-Encoder 等多种重排思路，工业界多用专用重排模型。", "distractor-rerank-general"),
    ("时区与时间处理是后端常见需求，常用标准库或第三方库进行本地化。日期计算需注意夏令时与历法差异，建议统一存储 UTC 再按需转换。", "distractor-time-general"),
]

# ---- 评测集：query -> 相关 source 集合（ground truth） ----
# 含精确关键词/缩写/数字型查询（BM25 优势）与语义改写型查询（稠密优势）
EVAL_SET = [
    # 精确关键词/缩写/数字/标识符型 —— BM25 易命中
    ("RRF 融合算法是什么", {"rrf-fusion"}),
    ("Reciprocal Rank Fusion", {"rrf-fusion"}),
    ("BM25 是什么", {"bm25-intro"}),
    ("gte-rerank", {"bailian-rerank"}),
    ("create_react_agent", {"langgraph-react"}),
    ("tool_calls 并行执行", {"langgraph-react"}),
    ("OpenAI 兼容接口地址", {"bailian-openai-compat"}),
    ("qwen-plus 模型", {"bailian-openai-compat"}),
    ("Milvus 默认端口", {"milvus-intro"}),
    ("19530", {"milvus-intro"}),
    ("DASHSCOPE_API_KEY", {"bailian-intro"}),
    ("PM2.5 AQI 查询", {"tool-air-quality"}),
    ("InMemorySaver", {"langgraph-intro"}),
    ("stream_mode", {"langgraph-streaming"}),
    # 语义改写型 —— 稠密优势
    ("倒数排名融合怎么计算", {"rrf-fusion"}),
    ("稠密检索和关键词检索有什么区别", {"dense-vs-sparse"}),
    ("LangGraph 怎么实现流式输出", {"langgraph-streaming"}),
    ("父子文档检索策略", {"parent-doc-retrieval"}),
    ("Postgres 存父文档", {"postgres-intro"}),
    ("递归分块器", {"recursive-splitter"}),
    ("查空气质量的工具", {"tool-air-quality"}),
    ("按时区查时间", {"tool-current-time"}),
    # 多相关文档型 —— 更考验召回完整性
    ("百炼有哪些能力", {"bailian-intro", "bailian-openai-compat", "bailian-rerank"}),
    ("检索相关技术", {"dense-vs-sparse", "bm25-intro", "rrf-fusion", "parent-doc-retrieval"}),
]

K = 5


def _index_corpus():
    """清空并重新索引评测语料，保证语料纯净唯一、无重复 source。"""
    import hashlib

    from langchain_core.documents import Document

    reset_index()  # 清空旧数据（含示例文档），避免同 source 污染指标
    h = build_hybrid()
    docs = [Document(page_content=t, metadata={"source": s}) for t, s in CORPUS]
    ids = ["doc-" + hashlib.sha1(d.page_content.encode("utf-8")).hexdigest()[:16] for d in docs]
    h.add_documents(docs, ids)
    print(f"已索引 {len(CORPUS)} 篇文档（清空后重建）")


def _sources(docs):
    return [d.metadata.get("source") for d in docs if d is not None]


def _recall(retrieved_set, relevant):
    if not relevant:
        return 1.0
    return len(retrieved_set & relevant) / len(relevant)


def _precision(retrieved_set, relevant, k):
    if k == 0:
        return 0.0
    return len(retrieved_set & relevant) / k


def _mrr(retrieved_list, relevant):
    for rank, src in enumerate(retrieved_list, start=1):
        if src in relevant:
            return 1.0 / rank
    return 0.0


def _ndcg(retrieved_list, relevant, k):
    """二值相关度的 nDCG@k：命中=1，未命中=0。"""
    dcg = 0.0
    for i, src in enumerate(retrieved_list[:k], start=1):
        rel = 1.0 if src in relevant else 0.0
        dcg += rel / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(name, search_fn):
    recalls, p1s, p5s, mrrs, ndcgs, hits = [], [], [], [], [], []
    details = []
    for query, relevant in EVAL_SET:
        docs = search_fn(query)
        src_list = _sources(docs)
        src_set = set(src_list)
        rec = _recall(src_set, relevant)
        p1 = _precision(set(src_list[:1]), relevant, min(1, len(src_list)))
        p5 = _precision(src_set, relevant, min(K, len(src_list)))
        mrr = _mrr(src_list, relevant)
        ndcg = _ndcg(src_list, relevant, K)
        hit = 1.0 if src_set & relevant else 0.0
        recalls.append(rec)
        p1s.append(p1)
        p5s.append(p5)
        mrrs.append(mrr)
        ndcgs.append(ndcg)
        hits.append(hit)
        details.append((query, relevant, src_list, rec, p1, p5, mrr, ndcg))
    n = len(EVAL_SET)
    summary = {
        "name": name,
        "Recall@5": sum(recalls) / n,
        "Precision@1": sum(p1s) / n,
        "Precision@5": sum(p5s) / n,
        "MRR": sum(mrrs) / n,
        "nDCG@5": sum(ndcgs) / n,
        "Hit@5": sum(hits) / n,
    }
    return summary, details


def main():
    _index_corpus()
    h = build_hybrid()
    print(f"\n子块数：{len(h._children)}  评测 query 数：{len(EVAL_SET)}\n")

    base_sum, base_det = evaluate("基线(仅稠密)", lambda q: h.baseline_search(q, k=K))
    hyb_sum, hyb_det = evaluate("混合(稠密+BM25+RRF+Reranker)", lambda q: h.search(q, final_k=K))

    # 明细对比
    print("=" * 116)
    header = f"{'query':<26}{'relevant':<20}{'基线返回top5':<40}{'混合返回top5':<40}"
    print(header)
    print("-" * 124)
    # 按 query 对齐输出基线与混合的 top5 + 各自 MRR
    print(f"{'query':<22}{'基线MRR':<10}{'混合MRR':<10}{'基线top5':<42}{'混合top5':<42}")
    print("-" * 124)
    for i, (q, rel) in enumerate(EVAL_SET):
        b_list = base_det[i][2]
        h_list = hyb_det[i][2]
        b_mrr = base_det[i][6]
        h_mrr = hyb_det[i][6]
        marker = " ★" if h_mrr > b_mrr else ""
        print(f"{q[:20]:<20}{b_mrr:<10.2f}{h_mrr:<10.2f}{str(b_list)[:40]:<42}{str(h_list)[:40]:<42}{marker}")
    print("=" * 124)

    # 指标汇总
    print(f"\n{'指标':<14}{'基线(仅稠密)':<22}{'混合(稠密+BM25+RRF+Reranker)':<30}{'提升':<10}")
    print("-" * 76)
    for key in ["Recall@5", "Precision@1", "Precision@5", "MRR", "nDCG@5", "Hit@5"]:
        bv = base_sum[key]
        hv = hyb_sum[key]
        delta = hv - bv
        sign = "+" if delta >= 0 else ""
        print(f"{key:<14}{bv:<22.4f}{hv:<30.4f}{sign}{delta*100:.2f}%")
    print("-" * 76)

    # 增益分析：混合召回率严格高于基线的 query
    print("\n混合检索带来增益的 query（混合召回 > 基线召回，或混合榜首命中而基线未命中）：")
    improved = False
    for i, (q, rel) in enumerate(EVAL_SET):
        b_rec = base_det[i][3]
        h_rec = hyb_det[i][3]
        b_p1 = base_det[i][4]
        h_p1 = hyb_det[i][4]
        if h_rec > b_rec or (h_p1 > b_p1):
            improved = True
            print(f"  ✓ {q}")
            print(f"      基线: recall={b_rec:.2f} p@1={b_p1:.2f}  →  混合: recall={h_rec:.2f} p@1={h_p1:.2f}")
    if not improved:
        print("  （两路指标持平；详见上方 MRR/nDCG 排序质量对比）")

    print("\n评测完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
