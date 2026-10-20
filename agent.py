"""科研文档智能问答 Agent（LangGraph ReAct）。

特性：三层记忆（短期窗口 / 长期摘要 / 永久画像）、工具（search_documents 文献检索、
get_air_quality 空气质量、get_current_time 时间）、流式输出、System Prompt 强制引用
来源 [n] 并列出参考文献、循环干预（相同工具+参数连续 3 次注入系统干预）与 120s
单轮超时。启动时自动扫描 dataset/ 目录入库新文档。

运行：python agent.py
"""

import json
import os
import queue
import threading
import time
import warnings

import psycopg

# 抑制 langgraph InMemorySaver 的 allowed_objects 弃用警告
warnings.filterwarnings("ignore", message=".*allowed_objects.*")

from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from config import get_llm
from docstore import PG_URI
from hybrid_retriever import MILVUS_URI, auto_ingest_dataset, get_search_tool
from memory import MemoryManager, SHORT_TERM_WINDOW
from tools import TOOLS

# ---- 健壮性参数 ----
MAX_STEPS = 15             # 单轮最大执行步数（agent/tool 交替计，langgraph 默认 25 偏松）
MAX_REPEAT_CALLS = 3       # 相同工具+相同参数连续调用达到该次数即判定循环并干预
AGENT_TIMEOUT = 120        # 单轮对话整体超时（秒），超时中断本轮

# 交互命令帮助
_HELP_TEXT = (
    "可用命令：\n"
    "  exit / quit / 退出  —— 退出并保存记忆\n"
    "  new / 新会话       —— 开启独立新线程\n"
    "  记住 <内容>         —— 写入用户画像（如：记住 我偏好中文回答）\n"
    "  help / ? / 帮助    —— 显示本帮助\n"
    "其余输入作为对话问题发送给 Agent。"
)

# 系统提示：面向科研场景的文献问答，强制引用来源、降低幻觉
SYSTEM_PROMPT = (
    "你是一名面向科研场景的文献智能问答助手，擅长基于已入库的学术文献回答问题，"
    "并提供精准的知识溯源。你具备以下工具能力：\n"
    "1) search_documents(query)：检索知识库中已入库的学术文献（PDF/Word/PPT）；\n"
    "2) get_air_quality(city)：查询城市实时空气质量（PM2.5/PM10/AQI）；\n"
    "3) get_current_time(timezone)：查询当前时间，可指定 IANA 时区。\n\n"
    "工具选择规则（必须严格遵守）：\n"
    "- 只调用回答当前问题所必需的工具，禁止为「凑齐信息」而调用无关工具："
    "用户没有明确问天气或空气质量时，禁止调用 get_air_quality；"
    "没有明确问当前时间或日期时，禁止调用 get_current_time。\n"
    "- 凡涉及知识性内容的提问（论文方法、实验数据、定义结论，"
    "以及概述、总结、比较等），默认先调用 search_documents 检索后再作答，"
    "严禁凭训练知识作答；只有纯粹的寒暄闲聊，或用户明确只问天气/时间时，"
    "才可以不调用检索工具。\n"
    "- 仅当确实需要多个工具时才并行调用（一次回复中发起多个 tool_calls）；"
    "不需要多工具时只调用一个，不足的信息留到下一轮再补，"
    "不得一次性把所有工具都发起。\n"
    "- 工具返回结果优先于你的内部知识，请基于结果组织中文回答。\n\n"
    "回答格式与引用约束（生成链路约束，降低幻觉）：\n"
    "- 只要调用了 search_documents 并拿到检索结果，就必须标注引用："
    "在相关论述后用方括号标注引用序号，如 [1]、[2]；"
    "序号对应 search_documents 返回结果中的顺序编号；\n"
    "- 回答末尾必须另起「参考文献」段落，格式为：\n"
    "  参考文献：\n"
    "  [1] 来源:xxx\n"
    "  [2] 来源:xxx\n"
    "- 不得省略引用标注或「参考文献」段落，也不得用「根据相关资料」等"
    "模糊表述代替具体出处；\n"
    "- 严禁编造文献内容或来源；若检索结果中无相关信息，应明确说明"
    "「知识库中未找到相关文献」，不得用训练知识冒充文献内容；\n"
    "- 对于文献中的公式、表格、数据，应忠实转述，不得篡改数值或结论；\n"
    "- 回答语言默认中文，专业术语可保留英文原文。\n\n"
    "记忆上下文说明：\n"
    "- 每次对话可能附带「记忆上下文」，含【长期记忆 · 历史会话摘要】与"
    "【永久记忆 · 用户画像】，应据此保持回答的连贯性与个性化；\n"
    "- 若记忆信息与当前会话内容或检索结果冲突，以当前会话内容与检索结果为准。\n\n"
    "错误处理说明：\n"
    "- 若工具返回错误或不可用信息，请向用户解释原因并给出排查建议"
    "（如检查网络/服务状态），严禁编造数据或假装工具调用成功。"
)


def _loop_warning(messages: list) -> str:
    """检测连续重复工具调用：相同工具+相同参数连续出现 MAX_REPEAT_CALLS 次。

    从最新消息向前回溯（遇 HumanMessage 停止），收集 AI 消息中 tool_calls 的
    (name, 排序后 JSON 的 args) 指纹；统计时间上最靠后的指纹的连续重复次数，
    达到阈值返回中文干预警告，否则返回空串。
    """
    recent: list[tuple[str, str]] = []
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "human":
            break
        for tc in getattr(msg, "tool_calls", None) or []:
            name = tc.get("name", "")
            args = json.dumps(
                tc.get("args", {}), sort_keys=True, ensure_ascii=False
            )
            recent.append((name, args))
    if not recent:
        return ""
    target = recent[0]
    count = 0
    for item in recent:
        if item == target:
            count += 1
        else:
            break
    if count >= MAX_REPEAT_CALLS:
        name, args = target
        return (
            f"你已连续 {count} 次以相同参数调用工具 {name}({args}) 未取得进展。"
            f"请立即停止重复该调用，更换策略（如调整参数、换一个问题角度），"
            f"或直接基于已有信息向用户回答，不得再次发起相同调用。"
        )
    return ""


def build_agent(memory_manager: MemoryManager, checkpointer=None):
    """构建带三层记忆的 ReAct agent。

    短期记忆：checkpointer=InMemorySaver 按 thread_id 累积会话消息；
    pre_model_hook 用 trim_messages 把送入 LLM 的消息裁剪为最近 SHORT_TERM_WINDOW
    条（只通过 llm_input_messages 影响 LLM 输入，checkpoint 中的完整历史保留，
    供退出时生成长期摘要）。
    长期 + 永久记忆：prompt 改为 callable，动态注入历史会话摘要与用户画像。

    多工具调用：create_react_agent 的 ToolNode 默认并行执行模型一次返回的
    多个 tool_calls；模型也可分多轮串行调用，ReAct 循环均能处理。

    可传入已有 checkpointer：重建 Agent（如运行中更换 API Key）时复用同一
    checkpointer，即可保留该线程的短期会话历史。
    """
    checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
    tools = [*TOOLS, get_search_tool()]

    def pre_model_hook(state):
        # 短期记忆窗口：仅裁剪送入 LLM 的消息，不删除 checkpoint 历史
        messages = state.get("messages") or []
        trimmed = trim_messages(
            messages,
            token_counter=lambda msgs: len(msgs),
            max_tokens=SHORT_TERM_WINDOW,
            strategy="last",
            start_on="human",
            include_system=False,
        )
        # 防循环：检测连续重复工具调用，注入系统干预消息引导 LLM 换策略
        warning = _loop_warning(messages)
        if warning:
            trimmed = [SystemMessage(content=f"[系统干预] {warning}"), *trimmed]
        return {"llm_input_messages": trimmed}

    def prompt_fn(state):
        # 长期 + 永久记忆注入：System Prompt 后追加记忆上下文
        messages = list(state.get("messages") or [])
        system = SYSTEM_PROMPT + memory_manager.inject_context()
        return [SystemMessage(content=system), *messages]

    agent = create_react_agent(
        model=get_llm(),
        tools=tools,
        prompt=prompt_fn,
        pre_model_hook=pre_model_hook,
        checkpointer=checkpointer,
    )
    return agent


def _produce_events(agent, inputs, config, events_q: queue.Queue) -> None:
    """生产者线程：迭代 agent.stream 并把事件入队；异常/结束打标。

    消费方通过哨兵事件区分：("__done__", None) 正常结束、("__error__", exc) 出错。
    """
    try:
        for ev in agent.stream(
            inputs, config=config, stream_mode=["messages", "updates"]
        ):
            events_q.put(ev)
    except Exception as e:  # 统一交给消费方做友好提示
        events_q.put(("__error__", e))
    finally:
        events_q.put(("__done__", None))


def stream_agent(
    agent, user_input: str, thread_id: str, memory_manager: MemoryManager
) -> None:
    """以流式方式运行 agent 并实时打印输出，带超时/统计/异常兜底。

    - messages：逐 token 输出 AI 文本，并标注工具调用/返回；
    - updates：节点完成时打印轨迹（agent -> tools -> agent）。
    - 防循环：config 设置 recursion_limit=MAX_STEPS，超步数抛出
      GraphRecursionError 时转为友好警告；
    - 容错：整体执行超过 AGENT_TIMEOUT 秒时中断本轮并提示，不打断主循环；
    - 统计：每轮输出耗时、步数、工具调用次数。
    对话结束后触发长期记忆检查（达到阈值时生成滚动摘要）。
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": MAX_STEPS,
    }
    inputs = {"messages": [HumanMessage(content=user_input)]}

    print(f"\n用户> {user_input}")
    print("助手> ", end="", flush=True)

    events_q: queue.Queue = queue.Queue()
    threading.Thread(
        target=_produce_events, args=(agent, inputs, config, events_q), daemon=True
    ).start()

    start_time = time.time()
    last_node_time = start_time
    steps = 0
    tool_calls = 0
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
            if isinstance(event[1], GraphRecursionError):
                loop_guard = True
            break
        mode, data = event
        if mode == "messages":
            chunk, _metadata = data
            msg_type = getattr(chunk, "type", None) or chunk.__class__.__name__
            content = getattr(chunk, "content", "")
            tool_calls_chunk = getattr(chunk, "tool_call_chunks", None) or getattr(
                chunk, "tool_calls", None
            )
            # 标注工具调用请求（可能含多个工具名，表示并行调用）。
            # 不要求 content 为空：部分模型流式输出工具调用时会附带思考文本。
            if msg_type == "AIMessageChunk" and tool_calls_chunk:
                names = [tc.get("name") for tc in tool_calls_chunk if tc.get("name")]
                if names:
                    print(f"\n[调用工具: {', '.join(names)}]", end="", flush=True)
            if msg_type == "AIMessageChunk" and content:
                print(content, end="", flush=True)
            elif msg_type == "ToolMessage" and content:
                preview = content if len(content) <= 160 else content[:160] + "..."
                print(f"\n[工具返回] {preview}\n", end="", flush=True)
        elif mode == "updates":
            now = time.time()
            for node_name, payload in data.items():
                msgs = (
                    payload.get("messages", []) if isinstance(payload, dict) else []
                )
                # 工具调用计数：以 tools 节点返回的 ToolMessage 条数为准
                # （并行调用一次会返回多条）。messages 流不保证产生
                # ToolMessage chunk，不能在那里计数。
                if node_name == "tools":
                    tool_calls += len(msgs)
                # 分步耗时：距上一个被上报节点的时间差（含流式/模型/工具实际耗时）
                dt = now - last_node_time if last_node_time is not None else 0.0
                trajectory.append(
                    f"[节点: {node_name}] 消息数={len(msgs)} 耗时={dt:.2f}s"
                )
            last_node_time = now
            # 步数只统计 agent/tools 节点（pre_model_hook 等 hook 节点不计入）
            if any(n in ("agent", "tools") for n in data):
                steps += 1

    print()  # AI 文本收尾换行
    if timed_out:
        print(f"\n[超时] 本轮执行超过 {AGENT_TIMEOUT} 秒，已中断。可简化问题后重试。")
    elif loop_guard:
        print(
            f"\n[警告] 已达到最大执行步数（{MAX_STEPS}），"
            "Agent 可能陷入循环，已中止本轮。"
        )
    elif error_msg:
        print(f"\n[错误] 本轮执行出错：{error_msg[:200]}")
        print(
            "建议检查 DASHSCOPE_API_KEY 是否有效、网络是否连通、"
            "Postgres/Milvus 服务是否正常。"
        )
    if trajectory:
        print("\n--- 执行轨迹 ---")
        for line in trajectory:
            print(line)
        print("----------------")
    elapsed = time.time() - start_time
    print(f"[统计] 耗时 {elapsed:.1f}s ｜ 步数 {steps} ｜ 工具调用 {tool_calls} 次\n")

    # 轮次计数 + 触发长期摘要（达到阈值时）
    memory_manager.maybe_remember(agent, thread_id)


def _preflight_checks() -> None:
    """启动自检：DASHSCOPE_API_KEY / Postgres / Milvus 连通性。

    只提示不阻断：失败项在相关功能调用时会通过工具错误反馈兜底。
    """
    print("[自检] 正在检查依赖服务...")
    problems: list[str] = []

    if not os.getenv("DASHSCOPE_API_KEY"):
        problems.append("未检测到 DASHSCOPE_API_KEY 环境变量（影响 LLM/Embedding/Reranker 调用）")

    try:
        conn = psycopg.connect(PG_URI, autocommit=True)
        conn.close()
        print("[自检] OK Postgres 连接正常")
    except Exception as e:
        problems.append(f"Postgres 连接失败：{e}")

    try:
        from pymilvus import connections

        connections.connect(alias="preflight", uri=MILVUS_URI)
        connections.disconnect("preflight")
        print("[自检] OK Milvus 连接正常")
    except Exception as e:
        problems.append(f"Milvus 连接失败：{e}")

    for p in problems:
        print(f"[自检] WARN {p}")
    if problems:
        print("[自检] 存在以上问题，相关功能调用时会报错并给出指引，可继续对话。")
    else:
        print("[自检] 全部通过。")


def main():
    """交互式命令行对话循环。

    - 启动时自动扫描 dataset/ 目录入库新文档（内容哈希去重），并执行依赖服务自检。
    - thread_id 由账号 id 派生（一个用户一个账号，默认单用户，可设 ACCOUNT_ID
      环境变量）：同一账号跨进程始终复用同一会话，长期记忆（对话摘要）自动
      按账号累积；输入"新会话"可开启独立新线程。
    - 输入"记住 xxx"显式写入用户画像；输入"help/?"查看命令帮助；
      退出（含异常退出）时兜底生成长期摘要并自动抽取用户画像。
    - 单轮异常被捕获并打印友好提示，不打断整个会话。
    """
    print("=" * 60)
    print("正在扫描 dataset 目录并自动入库新文档...")
    auto_ingest_dataset("dataset")
    print("=" * 60)

    _preflight_checks()

    thread_id = MemoryManager.default_thread_id()
    memory_manager = MemoryManager(thread_id)
    agent = build_agent(memory_manager)
    print(f"已启动流式对话 Agent（账号默认单用户，thread_id={thread_id}）。")
    print("命令：'exit' 退出并保存记忆｜'新会话' 开启新线程｜'记住 xxx' 写入用户画像｜'help' 查看帮助")
    print("可尝试：查北京空气质量 / 现在几点 / dataset 中文献相关问题")
    print("多工具示例：北京空气质量和现在几点？")

    try:
        while True:
            try:
                user_input = input("\n你> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break
            if not user_input:
                continue
            lowered = user_input.lower()
            if lowered in {"exit", "quit", "退出"}:
                break
            if lowered in {"help", "?", "帮助"}:
                print(_HELP_TEXT)
                continue
            if lowered in {"new", "新会话"}:
                # 先归档当前会话记忆，再开启新线程（短期记忆清空、长期按新 id 从零开始）
                memory_manager.remember(agent, thread_id)
                memory_manager.close()
                thread_id = MemoryManager.new_thread_id()
                memory_manager = MemoryManager(thread_id)
                agent = build_agent(memory_manager)
                print(f"已开启新会话（thread_id={thread_id}），永久记忆（用户画像）仍保留。")
                continue
            # 显式"记住"指令：直接写入用户画像，不进入对话
            if memory_manager.remember_explicit(user_input):
                continue
            try:
                stream_agent(agent, user_input, thread_id, memory_manager)
            except Exception as e:  # 最后防线：单轮异常不打断会话
                print(f"\n[错误] 本轮对话处理异常：{e}")
                print("可检查 API Key / 网络 / Postgres / Milvus 服务状态后重试。")
    finally:
        # 兜底：正常/异常退出都先写记忆再关闭连接
        try:
            memory_manager.remember(agent, thread_id)
        except Exception as e:
            print(f"[记忆] 退出时保存记忆失败：{e}")
        memory_manager.close()


if __name__ == "__main__":
    main()
