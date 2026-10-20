"""基于 Postgres 的父文档存储，持久化父子检索中的父文档（进程重启不丢失）。

表结构：parent_docs(key TEXT PRIMARY KEY, value JSONB)。mset 采用 upsert
（主键覆盖）语义，同一 doc_id 重复写入不累积，天然支持幂等索引。
"""

import os
from collections.abc import Iterator, Sequence

import psycopg
from langchain_core.documents import Document
from langchain_core.stores import BaseStore

# 连接配置：默认连本机 Docker 起的 Postgres，可用 PG_URI 覆盖
PG_URI = os.getenv("PG_URI", "postgresql://agent:agent@localhost:5432/agent_docs")
DOCSTORE_TABLE = os.getenv("PG_DOCSTORE_TABLE", "parent_docs")


class PostgresDocStore(BaseStore[str, Document]):
    """父文档存储：Postgres + JSONB，upsert 覆盖语义。

    实现 langchain_core.stores.BaseStore[str, Document] 接口，
    混合检索器通过 mset / mget / yield_keys 管理父文档。
    """

    def __init__(self, uri: str = PG_URI, table: str = DOCSTORE_TABLE) -> None:
        self.uri = uri
        self.table = table
        # autocommit=True：每条语句独立事务，避免长连接持有事务
        self._conn = psycopg.connect(uri, autocommit=True)
        self._ensure_table()

    def _ensure_table(self) -> None:
        """首次连接时建表（若已存在则跳过）。"""
        with self._conn.cursor() as cur:
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} ("
                f"  key TEXT PRIMARY KEY,"
                f"  value JSONB NOT NULL,"
                f"  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
                f")"
            )

    # ---- 序列化辅助 ----
    @staticmethod
    def _doc_to_dict(doc: Document) -> dict:
        """Document -> 可 JSON 序列化的 dict。"""
        return {"page_content": doc.page_content, "metadata": doc.metadata}

    @staticmethod
    def _row_to_doc(value) -> Document | None:
        """JSONB 行 -> Document，value 已被 psycopg 解码为 dict。"""
        if value is None:
            return None
        if isinstance(value, str):
            import json

            value = json.loads(value)
        return Document(
            page_content=value.get("page_content", ""),
            metadata=value.get("metadata", {}) or {},
        )

    # ---- BaseStore 接口 ----
    def mget(self, keys: Sequence[str]) -> list[Document | None]:
        """批量取父文档。"""
        if not keys:
            return []
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT key, value FROM {self.table} WHERE key = ANY(%s)",
                (list(keys),),
            )
            rows = {k: v for k, v in cur.fetchall()}
        return [self._row_to_doc(rows.get(k)) for k in keys]

    def mset(self, key_value_pairs: Sequence[tuple[str, Document]]) -> None:
        """批量 upsert 父文档（主键冲突则覆盖 value）。"""
        if not key_value_pairs:
            return
        from psycopg.types.json import Jsonb

        rows = [(k, Jsonb(self._doc_to_dict(v))) for k, v in key_value_pairs]
        with self._conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.table} (key, value, updated_at) "
                f"VALUES (%s, %s, NOW()) "
                f"ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, "
                f"updated_at = NOW()",
                rows,
            )

    def mdelete(self, keys: Sequence[str]) -> None:
        """批量删除父文档。"""
        if not keys:
            return
        with self._conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self.table} WHERE key = ANY(%s)", (list(keys),)
            )

    def yield_keys(self, *, prefix: str | None = None) -> Iterator[str]:
        """遍历所有 key（可按前缀过滤）。"""
        with self._conn.cursor() as cur:
            if prefix is None:
                cur.execute(f"SELECT key FROM {self.table}")
            else:
                cur.execute(
                    f"SELECT key FROM {self.table} WHERE key LIKE %s",
                    (prefix + "%",),
                )
            for (k,) in cur.fetchall():
                yield k

    def has_source(self, source: str) -> bool:
        """检查是否已存在指定 source 的文档。

        JSONB 结构为 {"page_content":..., "metadata":{"source":"xxx"}}，
        source 嵌在 metadata 内层，需用 value#>>'{metadata,source}' 查询。
        """
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT 1 FROM {self.table} "
                f"WHERE value#>>'{{metadata,source}}' = %s LIMIT 1",
                (source,),
            )
            return cur.fetchone() is not None

    def keys_by_source(self, source: str) -> list[str]:
        """返回指定 source 的所有父文档 key（含全部父块）。

        换解析器重入库前，需按 key 删除 Postgres 父文档和 Milvus 子块，
        否则检索会命中孤儿子块、取不到父上下文。
        """
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT key FROM {self.table} "
                f"WHERE value#>>'{{metadata,source}}' = %s",
                (source,),
            )
            return [k for (k,) in cur.fetchall()]

    # BaseStore 默认提供的 amget/amset/amdelete/ayield_keys 会转发到同步实现，
    # 这里无需重写。如需真正异步可自行实现，当前 agent 同步运行已足够。

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
