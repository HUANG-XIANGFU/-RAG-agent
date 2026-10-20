"""Web 平台后端（FastAPI）：复用 Agent / 混合检索 / 入库模块，提供浏览器界面所需 API。

启动：python server.py  （或 uvicorn server:app --port 8000）
访问：http://localhost:8000
"""

import json
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import psycopg
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel

from agent import AGENT_TIMEOUT, MAX_STEPS, build_agent
from config import MODEL_CANDIDATES, current_models, set_models
from docstore import PG_URI, PostgresDocStore
from hybrid_retriever import (
    MILVUS_URI,
    auto_ingest_dataset,
    build_hybrid,
    delete_by_source,
    rebuild_vector_index,
    reset_index,
    vector_state,
)
from document_parser import SUPPORTED_SUFFIXES
from ingest import ingest_file
from memory import MemoryManager

BASE_DIR = Path(__file__).resolve().parent
WEB_INDEX = BASE_DIR / "web" / "index.html"
DATASET_DIR = BASE_DIR / "dataset"
DOCSTORE_TABLE = "parent_docs_hybrid"

# 平台启动时是否自动扫描 dataset 入库（与 CLI 一致，可用 WEB_AUTO_INGEST=0 关闭）
AUTO_INGEST = os.getenv("WEB_AUTO_INGEST", "1").lower() not in {"0", "false", "no"}

_STATE_LOCK = threading.RLock()    # 保护会话（agent / memory / thread_id）切换
_CHAT_LOCK = threading.Lock()      # 串行化对话，避免同一 thread 并发写 checkpoint
_INGEST_LOCK = threading.Lock()    # 串行化入库/清库

# 进程内单会话：同一账号跨请求复用同一 thread_id（与 CLI 行为一致）
_SESSION: dict[str, Any] = {
    "thread_id": None,
    "memory": None,
    "agent": None,
    "checkpointer": None,
}

# 启动进度（写给人看，由前端轮询 /api/startup）
_STARTUP: dict[str, Any] = {"phase": "starting", "detail": "正在启动…", "started_at": time.time()}


def _bootstrap() -> None:
    """平台启动例程：依赖自检 → dataset 自动入库 → 预构建 Agent 会话。

    在后台线程执行，避免阻塞 HTTP 服务可用性（MinerU 首次解析可能较慢）；
    进度写入 _STARTUP 供前端轮询展示。
    """
    try:
        _STARTUP.update(phase="checking", detail="正在自检依赖服务…")
        try:
            from agent import _preflight_checks

            _preflight_checks()
        except Exception as e:
            print(f"[启动] 依赖自检异常：{e}")

        if AUTO_INGEST:
            _STARTUP.update(phase="ingesting", detail="正在扫描 dataset 目录并自动入库…")
            try:
                added = auto_ingest_dataset(str(DATASET_DIR))
                _STARTUP["detail"] = f"dataset 自动入库完成（新增 {added} 个父块）"
            except Exception as e:
                _STARTUP["detail"] = f"dataset 自动入库失败：{e}"
                print(f"[启动] {_STARTUP['detail']}")

        _STARTUP.update(phase="building", detail="正在构建 Agent 会话…")
        _get_session()
        _STARTUP.update(phase="ready", detail="Agent 已就绪")
        print(f"[启动] Agent 已就绪（thread_id={_SESSION['thread_id']}），平台可用。")
    except Exception as e:
        _STARTUP.update(phase="error", detail=f"启动失败：{e}")
        print(f"[启动] 失败：{e}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 平台启动即自动拉起 Agent（后台线程，不阻塞服务就绪）
    threading.Thread(target=_bootstrap, daemon=True).start()
    yield
    # 关闭：兜底保存记忆并释放连接
    with _STATE_LOCK:
        if _SESSION["agent"] is not None:
            try:
                _SESSION["memory"].remember(_SESSION["agent"], _SESSION["thread_id"])
            except Exception:
                pass
            _SESSION["memory"].close()


app = FastAPI(title="科研文档智能问答平台", version="1.0", lifespan=lifespan)


# ---------------- 会话管理 ----------------

def _build_session(thread_id: str, checkpointer=None) -> None:
    """按 thread_id 构建会话（记忆 + checkpointer + Agent）。"""
    ckpt = checkpointer if checkpointer is not None else InMemorySaver()
    mem = MemoryManager(thread_id)
    _SESSION.update(
        thread_id=thread_id, memory=mem, checkpointer=ckpt, agent=build_agent(mem, ckpt)
    )


def _get_session() -> dict[str, Any]:
    """懒加载会话；进程内只构建一次。"""
    with _STATE_LOCK:
        if _SESSION["agent"] is None:
            _build_session(MemoryManager.default_thread_id())
        return _SESSION


def _rotate_session() -> str:
    """归档当前会话记忆并开启独立新线程（永久画像保留）。"""
    with _STATE_LOCK:
        if _SESSION["memory"] is not None:
            try:
                _SESSION["memory"].remember(_SESSION["agent"], _SESSION["thread_id"])
            except Exception:
                pass
            _SESSION["memory"].close()
        tid = MemoryManager.new_thread_id()
        _build_session(tid)
        return tid


def _reuse_thread() -> None:
    """复用原 thread_id / checkpointer 重建 Agent（短期会话历史与记忆均不丢）。"""
    with _STATE_LOCK:
        tid = _SESSION["thread_id"] or MemoryManager.default_thread_id()
        ckpt = _SESSION["checkpointer"]
        if _SESSION["memory"] is not None:
            try:
                _SESSION["memory"].close()
            except Exception:
                pass
        _build_session(tid, ckpt)


def _reset_model_singletons() -> None:
    """清空模型与检索单例：下次调用按最新模型名 / API Key 重建。"""
    import config as _config
    import hybrid_retriever as _hr

    _config._llm = None           # 模型单例：否则旧对象（旧 Key / 旧模型名）被复用
    _config._embeddings = None
    _config._reranker = None
    _hr._hybrid = None            # 检索器持有 Embedding / Reranker，须一并重建
    _hr._search_documents = None  # 检索工具单例，重建以指向新检索器


def _relink_agent(new_key: str) -> None:
    """运行中更换百炼 API Key（无需重启服务）。

    写入环境变量 → 清空 config 模型单例与混合检索单例（下次调用用新 Key 重建）→
    复用原 thread_id / checkpointer / 记忆重建 Agent，短期会话历史与长期记忆均不丢。
    """
    os.environ["DASHSCOPE_API_KEY"] = new_key
    _reset_model_singletons()
    _reuse_thread()


# 模型侧报错特征串：区分「额度用尽 / Key 无效」，前端据此提示换 Key 或充值
_QUOTA_SIGNS = (
    "allocationquota", "freetieronly", "free quota", "quota", "exhausted",
    "arrearage", "insufficient", "欠费", "余额不足", "免费额度",
)
_AUTH_SIGNS = (
    "invalidapikey", "invalid_api_key", "incorrect api key", "unauthorized",
    "authentication", "401",
)


def _classify_llm_error(exc: Any) -> dict[str, str]:
    """把模型侧报错归类为额度 / 鉴权 / 其它，并给出可操作的中文提示。"""
    low = str(exc).lower()
    if any(s in low for s in _QUOTA_SIGNS):
        return {
            "kind": "quota",
            "hint": "百炼额度已用尽（403 AllocationQuota / Free quota exhausted）："
                    "可点「更换 API Key」填入新 Key 立即恢复，或到百炼控制台充值、"
                    "关闭「仅使用免费额度」模式。",
        }
    if any(s in low for s in _AUTH_SIGNS):
        return {
            "kind": "auth",
            "hint": "API Key 无效或已失效：请点「更换 API Key」填入正确的百炼 Key。",
        }
    return {"kind": "other", "hint": ""}


# ---------------- 基础端点 ----------------

@app.get("/")
def index():
    if not WEB_INDEX.exists():
        raise HTTPException(500, "web/index.html 缺失")
    return FileResponse(WEB_INDEX)


@app.get("/api/health")
def health():
    """依赖服务自检：DashScope Key / Postgres / Milvus（只报告不阻断）。"""
    out: dict[str, Any] = {
        "dashscope": bool(os.getenv("DASHSCOPE_API_KEY")),
        "postgres": False,
        "milvus": False,
    }
    try:
        conn = psycopg.connect(PG_URI, autocommit=True)
        conn.close()
        out["postgres"] = True
    except Exception as e:
        out["postgres_error"] = str(e)
    try:
        from pymilvus import connections

        connections.connect(alias="health", uri=MILVUS_URI)
        connections.disconnect("health")
        out["milvus"] = True
    except Exception as e:
        out["milvus_error"] = str(e)
    # 不触达 agent（未配置 API Key 时也能自检），仅取默认 thread_id
    out["thread_id"] = _SESSION["thread_id"] or MemoryManager.default_thread_id()
    return out


@app.get("/api/startup")
def startup_state():
    """平台启动进度：starting → checking → ingesting → building → ready（或 error）。"""
    return _STARTUP


# ---------------- 知识库管理 ----------------

@app.get("/api/documents")
def documents():
    """列出已入库文档（按 source 聚合父块数与表格块数）。"""
    store = PostgresDocStore(table=DOCSTORE_TABLE)
    try:
        with store._conn.cursor() as cur:
            cur.execute(
                f"SELECT value#>>'{{metadata,source}}' AS source, "
                f"COUNT(*) AS parents, "
                f"COUNT(*) FILTER (WHERE value#>>'{{metadata,chunk_type}}' = 'table') AS tables, "
                f"MAX(updated_at) AS updated_at "
                f"FROM {DOCSTORE_TABLE} "
                f"WHERE value#>>'{{metadata,source}}' IS NOT NULL "
                f"GROUP BY source ORDER BY MAX(updated_at) DESC"
            )
            rows = cur.fetchall()
        return {
            "documents": [
                {"source": s, "parents": p, "tables": t, "updated_at": str(u)}
                for s, p, t, u in rows
            ]
        }
    finally:
        store.close()


@app.get("/api/dataset")
def dataset_list():
    """列出 dataset 目录下的可入库文件及其入库状态。"""
    files: list[dict] = []
    if DATASET_DIR.exists():
        store = PostgresDocStore(table=DOCSTORE_TABLE)
        try:
            for p in sorted(DATASET_DIR.iterdir()):
                if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
                    files.append(
                        {
                            "name": p.name,
                            "size_kb": round(p.stat().st_size / 1024, 1),
                            "ingested": store.has_source(p.name),
                        }
                    )
        finally:
            store.close()
    return {"files": files}


@app.delete("/api/documents/{source}")
def delete_document(source: str):
    """按 source 删除索引（Postgres 父文档 + Milvus 子块 + BM25），并移除 dataset 中的原文件。"""
    with _INGEST_LOCK:
        deleted = delete_by_source(source)
        local = DATASET_DIR / os.path.basename(source)
        if local.exists():
            local.unlink()
    return {"source": source, "deleted_parents": deleted}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...), force: bool = Form(False)):
    """上传文档 → 存入 dataset → 解析 → 向量化入库。"""
    DATASET_DIR.mkdir(exist_ok=True)
    results: list[dict] = []
    for uf in files:
        name = os.path.basename(uf.filename or "")
        suffix = Path(name).suffix.lower()
        if not name or suffix not in SUPPORTED_SUFFIXES:
            results.append({"source": name, "ok": False, "message": f"不支持的格式 {suffix or '(空)'}"})
            continue
        dest = DATASET_DIR / name
        dest.write_bytes(await uf.read())
        try:
            added = await run_in_threadpool(_ingest_one, dest, force)
            results.append(
                {
                    "source": name,
                    "ok": True,
                    "added_parents": added,
                    "message": "入库完成" if added else "已入库（内容未变），跳过",
                }
            )
        except Exception as e:
            info = _classify_llm_error(e)
            results.append(
                {
                    "source": name,
                    "ok": False,
                    "message": str(e)[:400],
                    "kind": info["kind"],
                    "hint": info["hint"],
                }
            )
    return {"results": results}


class IngestRequest(BaseModel):
    source: Optional[str] = None
    force: bool = False


@app.post("/api/ingest")
async def ingest(req: IngestRequest):
    """入库 dataset 中的指定文件；source 为空时扫描整个 dataset 目录。"""
    path: Optional[Path] = None
    if req.source:
        path = DATASET_DIR / os.path.basename(req.source)
        if not path.exists():
            raise HTTPException(404, f"dataset 中不存在文件：{req.source}")
    try:
        if path is not None:
            added = await run_in_threadpool(_ingest_one, path, req.force)
            return {"ok": True, "source": req.source, "added_parents": added}
        added = await run_in_threadpool(_ingest_dataset, req.force)
        return {"ok": True, "added_parents": added}
    except Exception as e:
        info = _classify_llm_error(e)
        return {"ok": False, "message": str(e)[:400], "kind": info["kind"], "hint": info["hint"]}


@app.post("/api/reset")
async def reset():
    """清空全部索引（Postgres 父文档 + Milvus 子块 + 本地入库记录）。"""
    report = await run_in_threadpool(_reset)
    return report


def _ingest_one(path: Path, force: bool) -> int:
    with _INGEST_LOCK:
        return ingest_file(str(path), force=force)


def _ingest_dataset(force: bool) -> int:
    with _INGEST_LOCK:
        return auto_ingest_dataset(str(DATASET_DIR), force=force)


def _reset() -> dict:
    with _INGEST_LOCK:
        return reset_index()


# ---------------- 检索调试 ----------------

class SearchRequest(BaseModel):
    query: str
    k: int = 5


@app.post("/api/search")
async def search(req: SearchRequest):
    """混合检索调试：返回 top-k 父文档及其元数据。"""
    def _run():
        docs = build_hybrid().search(req.query, final_k=max(1, req.k))
        return [
            {
                "rank": i,
                "source": d.metadata.get("source", "unknown"),
                "chunk_type": d.metadata.get("chunk_type", ""),
                "section": d.metadata.get("section", ""),
                "caption": d.metadata.get("caption", ""),
                "doc_id": d.metadata.get("doc_id", ""),
                "content": d.page_content,
            }
            for i, d in enumerate(docs, 1)
        ]

    return {"results": await run_in_threadpool(_run)}


# ---------------- 运行中更换 API Key ----------------

class ApiKeyRequest(BaseModel):
    api_key: str
    verify: bool = True


@app.post("/api/settings/api-key")
async def set_api_key(req: ApiKeyRequest):
    """运行中更换百炼 API Key：更新环境变量 → 重建模型/检索器/Agent，无需重启服务。

    verify=True 时用新 Key 发一次极短请求校验，返回校验结果与可操作提示。
    """
    key = (req.api_key or "").strip()
    if not key:
        raise HTTPException(400, "API Key 不能为空")
    try:
        await run_in_threadpool(_relink_agent, key)
    except Exception as e:
        raise HTTPException(400, f"重建 Agent 失败：{e}") from e

    verified: Optional[bool] = None
    detail = "已更新（未校验）"
    if req.verify:
        def _ping() -> Any:
            from config import get_llm

            return get_llm().invoke("ping")

        try:
            await run_in_threadpool(_ping)
            verified, detail = True, "新 Key 校验通过"
        except Exception as e:
            verified = False
            info = _classify_llm_error(e)
            detail = info["hint"] or f"新 Key 校验未通过：{str(e)[:200]}"
    return {"ok": True, "verified": verified, "detail": detail, "key_tail": key[-4:]}


# ---------------- 运行中更换模型 ----------------

# Embedding 模型已切换但向量索引尚未重建（检索会退化为仅 BM25，需提示用户重建）
_VECTOR_DIRTY = False


class ModelsRequest(BaseModel):
    llm: Optional[str] = None
    embedding: Optional[str] = None
    reranker: Optional[str] = None
    verify: bool = True


def _reuse_thread_safe() -> None:
    """重建 Agent 但吞掉异常（如未配置 API Key），保证切换流程本身不崩。"""
    try:
        _reuse_thread()
    except Exception as e:
        print(f"[models] 重建 Agent 失败：{e}")


def _apply_rebuild() -> dict[str, Any]:
    """重建向量索引并用新检索器重建 Agent。"""
    global _VECTOR_DIRTY
    try:
        report = rebuild_vector_index()
    except Exception as e:
        _VECTOR_DIRTY = True
        _reuse_thread_safe()   # 检索仍可用（退化为仅 BM25），平台不中断
        info = _classify_llm_error(e)
        return {
            "ok": False,
            "message": str(e)[:400],
            "kind": info["kind"],
            "hint": info["hint"] or "向量索引重建中断，可修正后点「重建向量索引」重试。",
            "vectors": vector_state(),
            "vectors_dirty": True,
        }
    _VECTOR_DIRTY = False
    _reuse_thread_safe()
    return {
        "ok": True,
        "embedding_rebuilt": True,
        "report": report,
        "vectors": vector_state(),
        "vectors_dirty": False,
        "detail": f"向量索引已重建：{report['parents']} 个父块 → "
                  f"{report['children']} 个子块重新向量化",
    }


def _apply_models(target: dict[str, str], verify: bool) -> dict[str, Any]:
    """更换模型并热重建（无需重启进程）。

    - LLM：先校验新模型可用，失败立即回滚，避免整个 Agent 不可用；
    - Embedding：先校验新模型可用，再删 Milvus 旧集合并按 Postgres 父文档
      重新向量化（不重新解析文档）；
    - Reranker：热切换即可（重排异常时检索会退回融合顺序，不影响可用性）。
    最后复用原 thread_id / checkpointer 重建 Agent，会话历史保留。
    """
    global _VECTOR_DIRTY
    before = current_models()
    applied = set_models(
        llm=(target.get("llm") or None),
        embedding=(target.get("embedding") or None),
        reranker=(target.get("reranker") or None),
    )
    if not applied:
        return {
            "ok": True, "applied": {}, "vectors": vector_state(),
            "vectors_dirty": _VECTOR_DIRTY, "detail": "模型未变化，无需切换",
        }

    # 校验主模型：新模型名写错或不可用时整体回滚（Agent 仍用旧模型继续工作）
    if "llm" in applied and verify:
        try:
            from config import get_llm

            get_llm().invoke("ping")
        except Exception as e:
            set_models(
                llm=before["llm"],
                embedding=before["embedding"],
                reranker=before["reranker"],
            )
            _reset_model_singletons()
            info = _classify_llm_error(e)
            return {
                "ok": False, "applied": {},
                "message": f"主模型 {applied['llm']} 不可用，已回滚为 {before['llm']}",
                "kind": info["kind"],
                "hint": info["hint"] or str(e)[:200],
                "vectors": vector_state(),
                "vectors_dirty": _VECTOR_DIRTY,
            }

    # 校验向量模型：此时尚未删除任何数据，失败可安全回滚
    if "embedding" in applied:
        try:
            from config import get_embeddings

            get_embeddings().embed_query("ping")
        except Exception as e:
            set_models(embedding=before["embedding"])
            _reset_model_singletons()
            _reuse_thread_safe()
            info = _classify_llm_error(e)
            return {
                "ok": False, "applied": {},
                "message": f"向量模型 {applied['embedding']} 不可用，"
                           f"已回滚为 {before['embedding']}",
                "kind": info["kind"],
                "hint": info["hint"] or str(e)[:200],
                "vectors": vector_state(),
                "vectors_dirty": _VECTOR_DIRTY,
            }

    # 检索器持有 Embedding / Reranker：这两者变化时必须重建检索器才能生效
    # （仅换主模型则不重建检索器，避免无谓的 BM25 重切重排）
    if "embedding" in applied or "reranker" in applied:
        import hybrid_retriever as _hr

        _hr._hybrid = None
        _hr._search_documents = None

    if "embedding" in applied:
        result = _apply_rebuild()   # 内部已重建 Agent
        result["applied"] = applied
        return result

    _reuse_thread_safe()
    summary = "、".join(f"{k}={v}" for k, v in applied.items())
    return {
        "ok": True,
        "applied": applied,
        "embedding_rebuilt": False,
        "vectors": vector_state(),
        "vectors_dirty": _VECTOR_DIRTY,
        "detail": f"已热切换（{summary}），Agent 已重建，会话历史保留",
    }


@app.get("/api/settings/models")
def get_models():
    """当前模型配置 + 可选模型候选 + 向量索引状态。"""
    return {
        "models": current_models(),
        "candidates": MODEL_CANDIDATES,
        "vectors": vector_state(),
        "vectors_dirty": _VECTOR_DIRTY,
    }


@app.post("/api/settings/models")
async def apply_models(req: ModelsRequest):
    """运行中更换主模型 / 向量模型 / 重排模型（无需重启）。

    切换向量模型会自动删除 Milvus 旧集合并用 Postgres 父文档重新向量化
    （不重新 MinerU 解析），耗时较长，前端需提示用户勿关闭页面。
    """
    target = {
        k: v.strip()
        for k, v in (
            ("llm", req.llm), ("embedding", req.embedding), ("reranker", req.reranker)
        )
        if (v or "").strip()
    }
    if not target:
        raise HTTPException(400, "请至少指定一个要切换的模型")
    try:
        return await run_in_threadpool(_apply_models, target, req.verify)
    except Exception as e:
        # 兜底：Milvus/Postgres 等依赖异常也给出可操作提示，不返回裸 500
        info = _classify_llm_error(e)
        return {
            "ok": False,
            "message": str(e)[:400],
            "kind": info["kind"],
            "hint": info["hint"] or str(e)[:200],
            "vectors": vector_state(),
            "vectors_dirty": _VECTOR_DIRTY,
        }


@app.post("/api/settings/models/rebuild-vectors")
async def rebuild_vectors_endpoint():
    """手动重建向量索引（换向量模型后重建中断时可重试）。"""
    return await run_in_threadpool(_apply_rebuild)


# ---------------- 对话（SSE 流式） ----------------

class ChatRequest(BaseModel):
    message: str
    new_session: bool = False


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _produce_events(agent, inputs, config, q: queue.Queue) -> None:
    """生产者线程：迭代 agent.stream 并入队；结束/异常打哨兵。"""
    try:
        for ev in agent.stream(inputs, config=config, stream_mode=["messages", "updates"]):
            q.put(ev)
    except Exception as e:
        q.put(("__error__", e))
    finally:
        q.put(("__done__", None))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    text = (req.message or "").strip()
    if not text:
        raise HTTPException(400, "消息不能为空")

    def event_stream():
        yield _sse({"type": "start"})
        if req.new_session:
            _rotate_session()
        session = _get_session()
        mem: MemoryManager = session["memory"]

        # 显式"记住 xxx"指令：直接写用户画像，不进入对话
        try:
            if mem.remember_explicit(text):
                yield _sse({"type": "notice", "text": "已写入永久记忆（用户画像）"})
                yield _sse({"type": "done", "stats": {"elapsed": 0.0, "steps": 0, "tools": 0, "trajectory": []}})
                return
        except Exception as e:
            yield _sse({"type": "error", "message": f"记忆写入失败：{e}"})
            yield _sse({"type": "done", "stats": {"elapsed": 0.0, "steps": 0, "tools": 0, "trajectory": []}})
            return

        agent = session["agent"]
        thread_id = session["thread_id"]
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": MAX_STEPS}
        inputs = {"messages": [HumanMessage(content=text)]}

        events_q: queue.Queue = queue.Queue()
        with _CHAT_LOCK:
            threading.Thread(
                target=_produce_events, args=(agent, inputs, config, events_q), daemon=True
            ).start()

            start = time.time()
            last_node = start
            steps = 0
            tools_called = 0
            trajectory: list[str] = []
            timed_out = False
            loop_guard = False
            error_msg = ""

            while True:
                try:
                    event = events_q.get(timeout=AGENT_TIMEOUT)
                except queue.Empty:
                    timed_out = True
                    break
                if event[0] == "__done__":
                    break
                if event[0] == "__error__":
                    error_msg = str(event[1])
                    loop_guard = isinstance(event[1], GraphRecursionError)
                    break
                mode, data = event
                if mode == "messages":
                    chunk, _meta = data
                    msg_type = getattr(chunk, "type", None) or chunk.__class__.__name__
                    content = getattr(chunk, "content", "")
                    tool_calls_chunk = getattr(chunk, "tool_call_chunks", None) or getattr(chunk, "tool_calls", None)
                    if msg_type == "AIMessageChunk" and tool_calls_chunk:
                        names = [tc.get("name") for tc in tool_calls_chunk if tc.get("name")]
                        if names:
                            yield _sse({"type": "tool_call", "names": names})
                    if msg_type == "AIMessageChunk" and content:
                        yield _sse({"type": "token", "text": content})
                    elif msg_type == "ToolMessage" and content:
                        preview = content if len(content) <= 200 else content[:200] + "..."
                        yield _sse({"type": "tool_result", "preview": preview})
                elif mode == "updates":
                    now = time.time()
                    for node_name, payload in data.items():
                        msgs = payload.get("messages", []) if isinstance(payload, dict) else []
                        if node_name == "tools":
                            tools_called += len(msgs)
                        dt = now - last_node
                        trajectory.append(f"[节点: {node_name}] 消息数={len(msgs)} 耗时={dt:.2f}s")
                        yield _sse({"type": "node", "name": node_name, "msgs": len(msgs), "dt": round(dt, 2)})
                    last_node = now
                    if any(n in ("agent", "tools") for n in data):
                        steps += 1

            if timed_out:
                yield _sse({"type": "error", "message": f"本轮执行超过 {AGENT_TIMEOUT}s，已中断，可简化问题后重试。"})
            elif loop_guard:
                yield _sse({"type": "error", "message": f"已达最大步数（{MAX_STEPS}），可能陷入循环，本轮已中止。"})
            elif error_msg:
                info = _classify_llm_error(error_msg)
                yield _sse({
                    "type": "error",
                    "kind": info["kind"],
                    "hint": info["hint"],
                    "message": f"本轮执行出错：{error_msg[:300]}",
                })

            elapsed = round(time.time() - start, 1)
            yield _sse({
                "type": "done",
                "stats": {"elapsed": elapsed, "steps": steps, "tools": tools_called, "trajectory": trajectory},
            })

        # 轮次计数：达到阈值触发长期摘要（在对话锁外执行，避免长时间占用）
        try:
            mem.maybe_remember(agent, thread_id)
        except Exception:
            pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------- 会话 / 记忆 ----------------

@app.post("/api/session/new")
def new_session():
    return {"thread_id": _rotate_session()}


@app.get("/api/memory")
def memory_state():
    """当前会话的长期摘要 + 全局用户画像。"""
    session = _get_session()
    store = session["memory"]._store
    return {
        "thread_id": session["thread_id"],
        "summary": store.get_summary(session["thread_id"]),
        "facts": store.get_profile(),
    }


class RememberRequest(BaseModel):
    text: str


@app.post("/api/memory/remember")
def remember(req: RememberRequest):
    """显式写入用户画像（等价于对话中的"记住 xxx"）。"""
    session = _get_session()
    ok = session["memory"].remember_explicit(req.text)
    return {"ok": ok, "facts": session["memory"]._store.get_profile()}


@app.delete("/api/memory/{key}")
def forget(key: str):
    """删除一条用户画像。"""
    store = _get_session()["memory"]._store
    with store._conn.cursor() as cur:
        cur.execute(f"DELETE FROM {store.profile_table} WHERE key = %s", (key,))
        deleted = cur.rowcount
    return {"deleted": deleted, "facts": store.get_profile()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("WEB_PORT", "8000")))
