"""知识库入库工具：解析 PDF/Word/PPT 并入库到混合检索系统。

流程：parse_document 解析为 Markdown → metadata_extractor 清洗/提取元数据 →
hybrid_retriever 结构感知分块 + 混合索引（Postgres 父文档 / Milvus 子块 / BM25）。

去重：默认按 source（文件名）查 Postgres 去重，已入库直接跳过；--force 时先按
source 删除旧父文档（Postgres）+ 旧子块（Milvus）再解析入库（两者必须同时清理，
否则检索会命中孤儿子块）。

用法：
    python ingest.py path/to/paper.pdf           # 导入文件（已入库则跳过）
    python ingest.py path/to/docs_dir/ --force   # 强制重解析并重入库
    python ingest.py path/to/docs_dir/ --reset   # 清空全部索引后再入库
    python ingest.py --reset                     # 仅重置，不入库
"""

import argparse
import sys
from pathlib import Path

from document_parser import parse_document
from docstore import PostgresDocStore
from hybrid_retriever import (
    add_documents,
    auto_ingest_dataset,
    delete_by_source,
    reset_index,
)
from metadata_extractor import extract_metadata, write_review_report

# 与 hybrid_retriever.HybridRetriever 使用同一张父文档表
_DOCSTORE_TABLE = "parent_docs_hybrid"


def ingest_file(file_path: str, force: bool = False) -> int:
    """解析单个文档并入库，返回新增父块数。

    默认先按 source（文件名）查 Postgres 去重，已入库直接跳过，
    不触发 MinerU 解析。force=True 时先删旧索引（Postgres 父文档 +
    Milvus 子块）再重入库。
    """
    path = Path(file_path)
    source = path.name

    # 去重检查只依赖 Postgres，单独建连接即可，无需启动 Milvus
    store = PostgresDocStore(table=_DOCSTORE_TABLE)
    already = store.has_source(source)
    store.close()

    if already and not force:
        print(f"[ingest] 已入库，跳过：{source}（强制重入库请加 --force）")
        return 0
    if already and force:
        deleted = delete_by_source(source)
        print(f"[ingest] 清除旧索引：{source}（{deleted} 个父块）")

    print(f"[ingest] 解析文档：{source}")
    md, meta = parse_document(str(path))
    # 入库前管道：数据清洗 + 元数据提取（规则/LLM）+ 置信度筛选
    clean_md, file_meta, review = extract_metadata(md, meta)
    if review:
        write_review_report(source, review)   # 低置信度字段复核报告，不阻塞入库
    if not clean_md.strip():
        print(f"[ingest] 文档内容为空，跳过：{source}")
        return 0
    # 完整文档级元数据入库（source + 提取字段 + parser 继承字段）
    metas = {
        "source": source,
        **{k: v for k, v in file_meta.items() if v is not None and v != ""},
    }
    added = add_documents([clean_md], metadatas=[metas])
    print(f"[ingest] 入库完成：{source}（新增 {added} 个父块）")
    return added


def main():
    parser = argparse.ArgumentParser(description="文档知识库入库工具")
    parser.add_argument(
        "path", nargs="?", default=None,
        help="文档文件或目录路径；配合 --reset 单独使用时可省略（仅重置不入库）",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="清空全部索引：Postgres 父文档表（parent_docs 开头）+ Milvus 集合（agent_ 开头）+ 本地已入库记录文件",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重入库：先按 source 删除旧父文档（Postgres）+ 旧子块（Milvus）再解析入库",
    )
    args = parser.parse_args()

    if args.reset:
        print("[ingest] 清空全部索引（Postgres 父文档 + Milvus 子块 + 本地入库记录）...")
        report = reset_index()
        print(
            f"[ingest] 清理完成：Postgres 表 x{report['postgres_tables']}，"
            f"Milvus 集合 x{report['milvus_collections']}，"
            f"本地记录文件 x{report['record_files']}"
        )
        if not args.path:
            print("[ingest] 仅重置模式，未指定入库路径。")
            return
        print("[ingest] 开始重新入库...")

    if not args.path:
        parser.error("请提供文档文件或目录路径（或配合 --reset 仅重置）")

    target = Path(args.path)
    if not target.exists():
        print(f"[ingest] 路径不存在：{args.path}")
        sys.exit(1)

    if target.is_file():
        ingest_file(str(target), force=args.force)
    elif target.is_dir():
        auto_ingest_dataset(str(target), force=args.force)
    else:
        print(f"[ingest] 不支持的路径类型：{args.path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
