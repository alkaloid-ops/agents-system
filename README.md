# 多智能体系统（Multi-Agents System）

基于 OpenAI Agents SDK / LangGraph 构建的多智能体系统，集成飞书机器人与 Open WebUI，支持流式对话、图像理解、RAG 检索及多轮记忆。

飞书端视频demo（BV1wV546JELV）：
 - https://www.bilibili.com/video/BV1wV546JELV/?vd_source=aff3b25fd44c72d5fa43b58ed82ddbe4
网页端视频demo（BV1QjLw6PEDk）：
 - https://www.bilibili.com/video/BV1QjLw6PEDk/?vd_source=aff3b25fd44c72d5fa43b58ed82ddbe4

## ✨ 主要特性

- **多智能体协作**  
  路由智能体（Router）自动将用户请求分发至：
  - **即时响应助手** – 处理通用聊天、简单问答
  - **领域专家** – 处理需要深度分析的复杂问题
  - **检索助手** – 基于向量数据库的知识库检索（RAG）
- **双运行时支持**  
  可选基于 `openai-agents SDK` 或 `LangGraph` 实现，适应不同技术偏好。
- **多端交互**  
  - **飞书 App** – 群聊/私聊机器人，支持文本、图片、流式卡片回复
  - **Open WebUI** – 兼容 OpenAI API 的 `/chat/completions` 流式接口
- **智能检索**  
  - 使用 `BAAI/bge-m3` 稠密+稀疏混合检索
  - 结合 `bge-reranker-v2-m3` 重排与温度采样，提升召回质量
  - 支持 Milvus 向量数据库
- **对话记忆**  
  - OpenAI Agents 版本使用 SQLiteSession 持久化会话
  - LangGraph 版本使用 InMemorySaver（可替换为 Redis/PostgreSQL）
- **异步高性能**  
  全异步设计，飞书端使用消息缓冲与防抖，避免频繁触发


## 🚀 快速开始

### 1. 环境要求

- 已安装Docker、Docker compose、Git

### 2. 启动APP与WebUI双端智能体系统（一行命令即可打包镜像、拉取镜像并后台运行）

```bash
git clone git@github.com:alkaloid-ops/agents-system.git
docker compose up -d --build

# 此时milvus向量数据库的空的，没有知识库；本仓库提供了knowledge_docs示例文件作为知识库（shanghai disney 相关文档）
# 下列命令将示例文档进行embedding并存储到milvus向量数据库，用于RAG检索。
cd app/
pip install -r requirements.txt
python3 /app/rag_ingestion/ingestion.py
```

### 3. 配置环境变量

创建 `.env` 文件（保持和docker-compose.yml同级目录下即可）：

```ini
OPENAI_API_KEY="大模型API KEY（本地部署可随便填一个字符串）"
BASE_URL = "大模型服务方的URL或本地部署局域网或云端公网地址与端口"

FEISHU_APP_ID = "飞书创建机器人后的APP ID"
FEISHU_APP_SECRET = "飞书创建机器人后的密钥"
FEISHU_ROBOT_NAME = "飞书创建机器人的名称"

ROUTER_AGENT_MODEL = "路由智能体模型名称"
INSTANT_AGENT_MODEL = "快速响应智能体模型名称"
EXPERT_AGENT_MODEL = "专家推理智能体模型名称"
RETRIEVE_AGENT_MODEL = "RAG检索智能体模型名称"
RETRIEVE_EVALUATOR_MODEL = "RAG检索结果评估智能体模型名称"

EMBEDDING_MODEL = "/app/models/bge-m3"
RERANK_MODEL = "/app/models/bge-reranker-v2-m3"

MILVUS_HOST = "milvus"
MILVUS_PORT = "19530
```

### 4. 后续配置

- 飞书智能体在docker启动后无后续配置
- OpenWebUI智能体后续配置：
- 1. 游览器打开http://localhost:3000
  2. 创建管理员账号并登陆；
  3. 网页右上角点击头像 > Setting > Connections > OpenAI > Manage > ➕ Add New Connection
  4. URL地址输入http://openwebui-agent:8000
  5. 密钥设置为无即可使用

## 🧪 扩展与定制

### 添加新工具

在 `openai_agents_tools.py` 或 `langgraph_agents_tools.py` 中定义函数，并绑定到对应智能体。

### 切换大模型

修改配置文件中 `model` 字段（如 `gpt-4o`、`qwen-max` 等），需兼容 OpenAI 接口格式。

### 持久化记忆

- OpenAI Agents 版本默认使用 SQLite，修改 `db_path` 参数
- LangGraph 版本可将 `InMemorySaver` 替换为 `PostgresSaver` 或 `RedisSaver`

## 📄 日志与监控（Docker挂载卷持久化存储）

- 系统记录：`/app/logs/openai_agents_system_logs.jsonl` 或 `langgraph_agents_system_logs.jsonl`
- 检索日志：`/app/logs/openai_agents_retrieve_logs.jsonl` 或 `/app/logs/langgraph_agents_retrieve_logs.jsonl`
- 飞书 SDK 日志：控制台 `DEBUG` 级别（可调整）

## 🤝 贡献

欢迎提交 Issue 或 Pull Request。
