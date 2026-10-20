# 科研文档智能问答平台

基于 **LangGraph ReAct Agent + 混合检索（稠密 + BM25 → RRF → Reranker）** 的学术文献问答系统。
提供浏览器操作平台（上传文档→解析→向量化入库、流式对话、检索调试、记忆管理）与命令行入口。

模型层（LLM / Embedding / Reranker）统一走阿里云百炼，只需一个 `DASHSCOPE_API_KEY`。

## 功能

- **文档入库**：上传或扫描 `dataset/` 目录，自动解析（PDF/Word/PPT → Markdown）、清洗、提取元数据、结构感知分块，写入 Postgres + Milvus + BM25。
- **流式对话**：LangGraph ReAct Agent，逐 token 输出，展示工具调用与节点轨迹；强制引用来源 `[n]`，降低幻觉。
- **混合检索**：稠密检索（DashScope Embedding + Milvus）+ 关键词检索（BM25，jieba 分词），RRF 倒数排名融合后 Reranker 重排，返回 top5 父文档。
- **三层记忆**：短期窗口（最近 20 条消息）、长期摘要（跨会话，Postgres）、永久画像（显式"记住"或自动抽取）。
- **健壮性**：单轮 120s 超时、`recursion_limit=15`、相同工具+参数连续 3 次注入系统干预；Milvus 连接失败自动重连并尝试拉起容器。
- **知识库管理**：查看已入库文档、按来源删除、一键清空索引、检索结果调试。
- **运行中更换 API Key**：额度用尽或 Key 失效时页面给出可点击提示，填入新 Key 即刻重建模型 / 检索器 / Agent，**无需重启服务**，会话与短期历史不丢。
- **运行中更换模型**：主模型（LLM）/ 向量模型（Embedding）/ 重排模型（Reranker）均可在页面上切换，**无需重启服务**；换向量模型时自动删除 Milvus 旧集合并按 Postgres 父文档重新向量化（不重新解析文档）。

## 架构

```
浏览器单页 (web/index.html)  ── HTTP / SSE ──▶  server.py (FastAPI)
                                                    │
        ┌───────────────────────────────────────────┼──────────────────────────┐
        │                                           │                          │
   agent.py                                     ingest.py            hybrid_retriever.py
   LangGraph ReAct + 三层记忆                   解析→清洗→元数据→分块      稠密+BM25→RRF→Reranker
        │                                           │                          │
        │                                     ┌─────┴─────┐              ┌─────┼─────┐
        └─ search_documents 工具 ─────────────┤document_parser│         │     │     │
                                              │metadata_extract│   Postgres Milvus BM25
                                              └───────────────┘    父文档  子块向量 关键词
```

- 父文档（大块）存 Postgres 表 `parent_docs_hybrid`，供 LLM 完整上下文；子块（小块）向量存 Milvus 集合 `agent_hybrid_child`、全文进 BM25。
- 记忆存 Postgres 表 `memory_sessions`（长期摘要）与 `memory_profile`（用户画像）。

## 目录结构

```
.
├── config.py              # LLM / Embedding / Reranker（百炼）
├── agent.py               # ReAct Agent（CLI 入口）
├── server.py              # FastAPI 后端（Web 入口）
├── hybrid_retriever.py    # 混合检索 + 分块 + 父子存储
├── ingest.py              # 入库工具（CLI）
├── document_parser.py     # MinerU 解析（轻量回退）
├── metadata_extractor.py  # 清洗 + 元数据提取 + 复核报告
├── memory.py              # 三层记忆
├── docstore.py            # Postgres 父文档存储
├── tools.py               # 示例工具（空气质量 / 时间）
├── eval_retrieval.py      # 检索质量评测
├── web/index.html         # 浏览器单页界面（零构建）
├── dataset/               # 待入库文档目录
├── docker-compose.yml     # Milvus standalone 编排
└── requirements.txt
```

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 Milvus（Docker）

```bash
docker compose up -d
```

### 3. 启动 Postgres

```bash
docker run -d --name pg-agent \
  -e POSTGRES_USER=agent -e POSTGRES_PASSWORD=agent -e POSTGRES_DB=agent_docs \
  -p 5432:5432 postgres:16
```

### 4. 配置百炼 API Key

```powershell
# Windows PowerShell（当前会话生效）
$env:DASHSCOPE_API_KEY = "sk-你的百炼Key"
```

> 若报 403 `AllocationQuota.FreeTierOnly`，说明免费额度用尽，请到百炼控制台充值或关闭「仅使用免费额度」。

### 可选环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DASHSCOPE_API_KEY` | 无（必填） | 百炼 API Key |
| `MILVUS_URI` | `http://localhost:19530` | Milvus 地址 |
| `PG_URI` | `postgresql://agent:agent@localhost:5432/agent_docs` | Postgres 地址 |
| `ACCOUNT_ID` | `default` | 会话账号 id（决定默认 thread_id） |
| `LLM_MODEL` / `DASHSCOPE_EMBEDDING_MODEL` / `DASHSCOPE_RERANKER_MODEL` | `qwen3.7-flash` / `qwen3.7-text-embedding-flash` / `qwen3.7-text-rerank` | 模型名（亦可在页面「设置 → 模型设置」运行中切换） |
| `WEB_PORT` | `8000` | Web 服务端口 |
| `WEB_AUTO_INGEST` | `1` | 启动时是否自动扫描 `dataset/` 入库（`0` 关闭） |

## 快速开始

### Web 平台

```bash
python server.py
# 浏览器打开 http://localhost:8000
```

### 命令行

```bash
python agent.py                 # 交互式问答（启动时自动扫描 dataset 入库）
python ingest.py path/paper.pdf # 入库单个文件（已入库则跳过）
python ingest.py path/dir --force   # 强制重解析并重入库
python ingest.py --reset        # 清空全部索引（Postgres + Milvus + 本地记录）
python eval_retrieval.py        # 检索质量评测
```

## 浏览器平台

启动 `server.py` 后平台会**自动拉起 Agent**：依赖自检 → 扫描 `dataset/` 自动入库 → 预构建 Agent 会话，进度实时显示在页面顶部（可用 `WEB_AUTO_INGEST=0` 关闭自动入库）。顶部同时显示依赖自检状态（API Key / Postgres / Milvus）。四个面板：

| 面板 | 功能 |
|---|---|
| **对话** | 流式问答；展示工具调用、节点耗时与统计；「新会话」开启独立线程（永久画像保留）；输入「记住 xxx」写入用户画像 |
| **知识库** | 上传文件并解析入库；查看 `dataset/` 文件与入库状态（可单个入库 / 强制重入库）；查看已入库文档并删除；一键清空索引 |
| **检索** | 输入问题调试混合检索，展示 top-k 父文档及来源/章节/题注等元数据 |
| **设置** | 模型设置（运行中切换主模型 / 向量模型 / 重排模型，无需重启）；查看当前会话长期摘要与用户画像，写入或删除画像条目 |

## HTTP API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 依赖自检（API Key / Postgres / Milvus） |
| GET | `/api/startup` | 平台启动进度（自检 → 自动入库 → 构建 Agent） |
| POST | `/api/settings/api-key` | 运行中更换百炼 API Key（重建模型 / 检索器 / Agent，无需重启） |
| GET | `/api/settings/models` | 当前模型配置、可选模型候选、向量索引状态（集合 / 子块数 / 维度） |
| POST | `/api/settings/models` | 运行中切换主模型 / 向量模型 / 重排模型（换向量模型会自动重建向量索引） |
| POST | `/api/settings/models/rebuild-vectors` | 手动重建向量索引（换向量模型后重建中断时可重试） |
| POST | `/api/upload` | 上传文件 → 解析 → 入库（multipart，`force` 可选） |
| POST | `/api/ingest` | 入库 `dataset` 指定文件或整个目录 |
| GET | `/api/dataset` | 列出 `dataset` 文件与入库状态 |
| GET | `/api/documents` | 列出已入库文档（父块 / 表格块统计） |
| DELETE | `/api/documents/{source}` | 删除某文档全部索引 |
| POST | `/api/reset` | 清空全部索引 |
| POST | `/api/search` | 混合检索调试 |
| POST | `/api/chat` | 流式对话（SSE，`new_session` 可选） |
| POST | `/api/session/new` | 开启新会话 |
| GET | `/api/memory` | 长期摘要 + 用户画像 |
| POST | `/api/memory/remember` | 写入用户画像 |
| DELETE | `/api/memory/{key}` | 删除画像条目 |

## 检索原理

1. **分块**：按 Markdown 结构切父块（标题开新块、正文累积超 1500 字符切块、表格独立成块并附着题注`表 11 / Table XIII`）。
2. **索引**：父块 → Postgres；父块切子块 → 向量入 Milvus、全文入 BM25。
3. **召回**：稠密 top10 + BM25 top10。
4. **融合**：RRF `score = Σ 1/(60 + rank)`。
5. **重排**：DashScope Reranker 对融合结果重排，映射回父文档返回 top5。
6. **表号优先**：查询为纯表号引用（如「表 3」）时，对应编号的表格块硬前置；长句顺带提编号仅做召回兜底，不覆盖语义排序。

## 上传到 GitHub

仓库已通过 `.gitignore` 排除 `volumes/`、`__pycache__/`、`processed.json`、`.env` 等运行时产物。

```bash
git init
git add .
git commit -m "科研文档智能问答平台（Web + CLI）"
git branch -M main
git remote add origin https://github.com/<你的账号>/<仓库名>.git
git push -u origin main
```

## 额度用尽 / API Key 失效

对话或入库遇到 `403 AllocationQuota.FreeTierOnly`、`Free quota exhausted`、`InvalidApiKey` 等错误时，平台会：

1. 在对话流或上传结果中给出中文原因与处理建议；
2. 提供「更换 API Key」按钮（页面顶部亦有常驻入口）。

点击后填入新 Key：后端写入环境变量并清空模型 / Embedding / Reranker / 检索单例，复用原 `thread_id` 与 checkpointer 重建 Agent，**无需重启服务**，当前会话与短期历史保留。保存时会用新 Key 发一次极短请求校验，界面提示校验结果；若仍报额度错误，请到百炼控制台充值或关闭「仅使用免费额度」。

若是**单个模型的额度**用尽（换 Key 无用），可改为切换到其它同类模型，见下节。

## 运行中更换模型

百炼平台上同类模型很多，某个模型额度用尽时可换成其它模型（如 `qwen3.7-text-embedding` ⇄ `qwen3.7-text-embedding-flash`），在页面「设置 → 模型设置」或调用 API 完成，**无需重启服务**：

| 类型 | 切换代价 | 说明 |
|---|---|---|
| 主模型 LLM | 即时 | 清空单例 → 重建 Agent；切换前发一次极短请求校验，不可用则自动回滚 |
| 重排模型 Reranker | 即时 | 检索器重建后生效（重排报错时检索退回融合顺序，不影响可用性） |
| 向量模型 Embedding | 需重建向量索引 | 向量维度与向量空间都随模型变化，Milvus 中旧向量全部失效 |

**换向量模型的处理**：先校验新模型可用（失败即回滚、不动任何数据），再删除 Milvus 集合 `agent_hybrid_child`，按 Postgres 中已有父文档重新切子块并用新模型重新向量化（**不重新 MinerU 解析**），最后重建 Agent。期间请勿关闭页面；若中途失败（如额度用尽），界面会给出提示并保留「重建向量索引」按钮，修正后可重试（重试期间检索退化为仅 BM25，其余功能不受影响）。

模型候选取自后端 `MODEL_CANDIDATES`（界面下拉可选，也可直接填自定义模型名）；切换结果写入当前进程的环境变量，仅在本次运行内有效，重启后仍以启动时的环境变量 / 默认值为准（Milvus 中已重建的向量与模型一致，无需再重建）。

## 常见问题

- **Milvus 连不上**：容器冷启动时 Proxy 需数秒就绪，程序会自动退避重连并尝试 `docker start`；仍失败请先启动 Docker Desktop 并确认 `docker compose up -d`。
- **入库很慢**：MinerU 首次运行需从 modelscope 下载模型（自动设置国内源），后续走缓存。
- **检索不到表格数据**：确认文档解析产出为 HTML 表格（MinerU 主路径），系统会转纯文本并按行切分。
- **免费额度耗尽**：到百炼控制台充值或关闭「仅使用免费额度」，不影响已入库数据。
- **回答没有出处 `[n]` / 参考文献**：出处只来自 `search_documents` 的返回（含 `[n] (来源:文件名)`）加 System Prompt 的引用约束。若该轮没调检索工具（模型判定"不必检索"而直接凭记忆作答），就不会有出处。System Prompt 已强制"知识性提问默认先检索、命中结果必须标注引用并列参考文献"；若仍偶发，多为所用主模型指令遵循能力偏弱（如 flash 档或切换过的其它模型），可在「设置 → 模型设置」换用指令遵循更好的模型，并在对话轨迹里确认该轮是否调用了 `search_documents`。
- **不需要天气/时间却调用了 get_air_quality / get_current_time**：这两个是示例工具，System Prompt 与工具描述均已声明"仅当用户明确询问时调用"。若仍被误调，同样属于模型侧工具选择问题，建议更换主模型；不需要这两个能力时也可从 `tools.py` 末尾的 `TOOLS` 列表中移除。
