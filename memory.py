"""三层记忆机制：短期（会话窗口）+ 长期（跨会话摘要）+ 永久（用户画像）。

短期：agent.py 的 pre_model_hook + trim_messages，仅裁剪送入 LLM 的消息，不改 checkpoint。
长期：每 SUMMARY_EVERY_N_TURNS 轮触发一次 LLM 摘要（旧摘要 + 新对话合并压缩），
      存 Postgres 表 memory_sessions。
永久：LLM 自动抽取 + 显式"记住 xxx"，存 Postgres 表 memory_profile。
存储复用 docstore.py 的 psycopg autocommit 模式，同键 upsert 保证幂等。
"""

import json
import os
import re
import uuid

import psycopg
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm

# 记忆窗口与触发阈值
SHORT_TERM_WINDOW = 20          # 短期记忆窗口：送入 LLM 的最大消息条数
SUMMARY_EVERY_N_TURNS = 8       # 每 N 轮用户输入触发一次长期摘要
PROFILE_EXTRACT_WINDOW = 10     # 永久记忆抽取时回看的最大消息条数

# Postgres 连接配置（复用父文档存储的连接配置）
PG_URI = os.getenv("PG_URI", "postgresql://agent:agent@localhost:5432/agent_docs")
MEMORY_SESSIONS_TABLE = "memory_sessions"
MEMORY_PROFILE_TABLE = "memory_profile"

# 账号 id：一个用户一个账号，当前默认为单用户；可用环境变量 ACCOUNT_ID 覆盖
ACCOUNT_ID = os.getenv("ACCOUNT_ID", "default")

# 显式"记住"指令前缀（不区分大小写）
_REMEMBER_PREFIXES = ("记住", "请记住", "remember")

# LLM 提示词
_SUMMARIZE_PROMPT = (
    "你负责把一段科研问答对话压缩为长期记忆摘要。请保留：用户身份与背景、"
    "研究方向、讨论过的主题与结论、偏好、未完成事项。若提供了旧摘要，"
    "请将旧摘要与新对话合并、去重、去旧存新后输出。"
    "用简洁的中文要点列表输出，仅输出摘要本身，不要任何前缀说明。"
)
_EXTRACT_PROMPT = (
    "从以下对话中抽取值得长期记住的稳定用户事实（如身份、研究方向、技术栈、"
    "偏好、项目背景等）。过滤掉一次性任务、临时问题、情绪化表达。"
    '输出 JSON：{"facts": [{"key": "简短键名", "value": "事实内容", '
    '"category": "identity|research|preference|project|other"}]}。'
    '若无可抽取内容，输出 {"facts": []}。仅输出 JSON，不要其他文字。'
)


def _log_memory_llm_failure(stage: str, exc: Exception) -> None:
    """打印记忆 LLM 调用失败的友好提示；403 免费额度错误给专门指引。"""
    msg = str(exc)
    if (
        "403" in msg
        and ("allocationquota" in msg.lower() or "free quota exhausted" in msg.lower())
    ):
        print(
            f"[记忆] {stage}失败：百炼免费额度已用尽"
            f"（403 AllocationQuota.FreeTierOnly）\n"
            "      请到阿里云百炼控制台（bailian.console.aliyun.com）充值，\n"
            "      或关闭「仅使用免费额度」模式后重试；本次记忆保存已跳过，\n"
            "      不影响当前会话使用。"
        )
    else:
        print(f"[记忆] {stage}失败：{exc}")


class PostgresMemoryStore:
    """Postgres 记忆存储：memory_sessions（长期摘要）+ memory_profile（永久画像）。

    表结构：
    - memory_sessions(thread_id TEXT PRIMARY KEY, summary TEXT NOT NULL,
                      turn_count INTEGER NOT NULL DEFAULT 0, updated_at TIMESTAMPTZ)
    - memory_profile(key TEXT PRIMARY KEY, value TEXT NOT NULL,
                     category TEXT NOT NULL DEFAULT 'fact', updated_at TIMESTAMPTZ)
    """

    def __init__(
        self,
        uri: str = PG_URI,
        sessions_table: str = MEMORY_SESSIONS_TABLE,
        profile_table: str = MEMORY_PROFILE_TABLE,
    ) -> None:
        self.uri = uri
        self.sessions_table = sessions_table
        self.profile_table = profile_table
        self._conn = psycopg.connect(uri, autocommit=True)
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """首次连接时建表（已存在则跳过）。"""
        with self._conn.cursor() as cur:
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self.sessions_table} ("
                f"  thread_id TEXT PRIMARY KEY,"
                f"  summary TEXT NOT NULL,"
                f"  turn_count INTEGER NOT NULL DEFAULT 0,"
                f"  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
                f")"
            )
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self.profile_table} ("
                f"  key TEXT PRIMARY KEY,"
                f"  value TEXT NOT NULL,"
                f"  category TEXT NOT NULL DEFAULT 'fact',"
                f"  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
                f")"
            )

    # ---- 长期记忆：会话摘要 ----
    def get_summary(self, thread_id: str) -> str:
        """读取指定会话的历史摘要，无则返回空串。"""
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT summary FROM {self.sessions_table} WHERE thread_id = %s",
                (thread_id,),
            )
            row = cur.fetchone()
        return row[0] if row else ""

    def upsert_summary(self, thread_id: str, summary: str) -> None:
        """写入会话摘要（同 thread_id 覆盖旧摘要）。"""
        with self._conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self.sessions_table} "
                f"(thread_id, summary, turn_count, updated_at) "
                f"VALUES (%s, %s, 0, NOW()) "
                f"ON CONFLICT (thread_id) DO UPDATE SET "
                f"summary = EXCLUDED.summary, updated_at = NOW()",
                (thread_id, summary),
            )

    # ---- 永久记忆：用户画像 ----
    def get_profile(self) -> list[dict]:
        """读取全部用户画像条目，最新在前。"""
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT key, value, category FROM {self.profile_table} "
                f"ORDER BY updated_at DESC"
            )
            return [
                {"key": k, "value": v, "category": c}
                for k, v, c in cur.fetchall()
            ]

    def upsert_fact(self, key: str, value: str, category: str = "fact") -> None:
        """写入用户画像事实（同 key 新值覆盖旧值）。"""
        with self._conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self.profile_table} (key, value, category, updated_at) "
                f"VALUES (%s, %s, %s, NOW()) "
                f"ON CONFLICT (key) DO UPDATE SET "
                f"value = EXCLUDED.value, category = EXCLUDED.category, "
                f"updated_at = NOW()",
                (key, value, category),
            )

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


class MemoryManager:
    """三层记忆编排：注入上下文、滚动摘要、画像抽取、显式记住。

    绑定单个 thread_id（当前会话主体）；永久画像为全局共享。
    """

    def __init__(self, thread_id: str, llm=None) -> None:
        self.thread_id = thread_id
        self._llm = llm if llm is not None else get_llm()
        self._store = PostgresMemoryStore()
        self._turn_count = 0

    # ---- 记忆注入 ----
    def inject_context(self) -> str:
        """拼接长期摘要 + 用户画像，作为 System Prompt 的追加上下文。"""
        parts: list[str] = []
        summary = self._store.get_summary(self.thread_id)
        if summary:
            parts.append("【长期记忆 · 历史会话摘要】\n" + summary)
        profile = self._store.get_profile()
        if profile:
            lines = "\n".join(
                f"- {p['key']}: {p['value']}" for p in profile
            )
            parts.append("【永久记忆 · 用户画像】\n" + lines)
        if not parts:
            return ""
        return "\n\n=== 记忆上下文 ===\n" + "\n\n".join(parts) + "\n"

    # ---- 写入时机 ----
    def maybe_remember(self, agent, thread_id: str) -> None:
        """每轮对话后调用：达到阈值时触发长期摘要。"""
        self._turn_count += 1
        if self._turn_count >= SUMMARY_EVERY_N_TURNS:
            self._turn_count = 0
            self._summarize(agent, thread_id)

    def remember(self, agent, thread_id: str) -> None:
        """退出/切换会话时兜底调用：滚动摘要 + 抽取用户画像。"""
        self._summarize(agent, thread_id)
        self._extract_profile(agent, thread_id)

    def remember_explicit(self, user_input: str) -> bool:
        """处理显式"记住"指令：'记住 <键>: <值>' 或 '记住 <值>'。

        返回 True 表示已处理（该输入不再作为普通对话发送给 agent）。
        """
        text = user_input.strip()
        lowered = text.lower()
        body = ""
        for p in _REMEMBER_PREFIXES:
            if lowered.startswith(p.lower()):
                body = text[len(p):].strip(" ，,。:：")
                break
        if not body:
            return False
        m = re.match(r"^(.+?)[:：]\s*(.+)$", body)
        if m:
            key, value = m.group(1).strip(), m.group(2).strip()
        else:
            value = body
            key = body[:8]
        if not value:
            return False
        self._store.upsert_fact(key, value, "fact")
        print(f"[记忆] 已记住：{key} = {value}")
        return True

    # ---- 内部实现 ----
    def _get_history_messages(self, agent, thread_id: str) -> list:
        """从 checkpointer 读取当前会话全部历史消息（含工具消息）。"""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            checkpoint = agent.checkpointer.get(config)
        except Exception as e:
            print(f"[记忆] 读取会话历史失败：{e}")
            return []
        if checkpoint is None:
            return []
        values = checkpoint.get("channel_values", {}) if isinstance(checkpoint, dict) else {}
        return list(values.get("messages", []) or [])

    @staticmethod
    def _messages_to_text(messages: list) -> str:
        """历史消息 -> 可读文本（保留用户提问与助手回复/工具调用概要）。"""
        lines: list[str] = []
        for msg in messages:
            role = getattr(msg, "type", "") or ""
            content = getattr(msg, "content", "")
            if role == "human":
                lines.append(f"用户: {content}")
            elif role == "ai":
                tool_calls = getattr(msg, "tool_calls", None) or []
                if content:
                    lines.append(f"助手: {content}")
                if tool_calls:
                    names = [
                        tc.get("name", "") for tc in tool_calls if tc.get("name")
                    ]
                    if names:
                        lines.append(f"助手调用工具: {', '.join(names)}")
        return "\n".join(lines)

    def _summarize(self, agent, thread_id: str) -> None:
        """滚动摘要：旧摘要 + 新增对话交 LLM 压缩后 upsert。"""
        text = self._messages_to_text(self._get_history_messages(agent, thread_id))
        if not text.strip():
            return
        old = self._store.get_summary(thread_id)
        content = f"旧摘要：\n{old}\n\n对话：\n{text}" if old else f"对话：\n{text}"
        try:
            resp = self._llm.invoke(
                [SystemMessage(content=_SUMMARIZE_PROMPT), HumanMessage(content=content)]
            )
        except Exception as e:
            _log_memory_llm_failure("长期摘要生成", e)
            return
        new_summary = (getattr(resp, "content", "") or "").strip()
        if new_summary:
            self._store.upsert_summary(thread_id, new_summary)
            print(f"[记忆] 长期摘要已更新（{len(new_summary)} 字）")

    def _extract_profile(self, agent, thread_id: str) -> None:
        """从最近对话中抽取稳定用户事实写入画像；解析失败静默降级。"""
        messages = self._get_history_messages(agent, thread_id)[-PROFILE_EXTRACT_WINDOW:]
        text = self._messages_to_text(messages)
        if not text.strip():
            return
        try:
            resp = self._llm.invoke(
                [SystemMessage(content=_EXTRACT_PROMPT), HumanMessage(content=f"对话：\n{text}")]
            )
        except Exception as e:
            _log_memory_llm_failure("用户画像抽取", e)
            return
        facts = self._parse_facts(getattr(resp, "content", "") or "")
        if not facts:
            return
        saved = 0
        for f in facts:
            key = str(f.get("key", "")).strip()
            value = str(f.get("value", "")).strip()
            if not key or not value:
                continue
            category = str(f.get("category", "fact")).strip() or "fact"
            self._store.upsert_fact(key, value, category)
            saved += 1
        if saved:
            print(f"[记忆] 用户画像已更新 {saved} 条")

    @staticmethod
    def _parse_facts(raw: str) -> list[dict]:
        """从 LLM 输出中提取 facts 列表；输出非 JSON 时返回空列表。"""
        text = raw.strip()
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            text = m.group(0)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(data, dict):
            facts = data.get("facts", [])
            return facts if isinstance(facts, list) else []
        return []

    # ---- thread_id 生成 ----
    @classmethod
    def default_thread_id(cls) -> str:
        """按账号 id 生成稳定的默认会话 id。

        一个用户一个账号：同一账号跨进程/跨启动始终复用同一 thread_id，
        长期记忆（对话摘要）自动按账号累积，无需本地文件记录。
        """
        return f"chat-{ACCOUNT_ID}"

    @classmethod
    def new_thread_id(cls) -> str:
        """为"新会话"命令生成独立的会话 id（仍归属当前账号）。"""
        return f"chat-{ACCOUNT_ID}-{uuid.uuid4().hex[:8]}"

    def close(self) -> None:
        self._store.close()
