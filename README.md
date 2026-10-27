# 企业级 RAG 智能知识库问答系统

基于 `FastAPI + LangChain + ChromaDB + BM25 + Ollama/DeepSeek` 构建的私有化 RAG（检索增强生成）问答系统。专为工业手册、长篇文档等复杂场景设计，彻底解决传统 RAG 的“跨页截断”、“目录污染”与“溯源错位”问题。

##  核心亮点
- **企业级架构**：FastAPI + Pydantic 数据契约，模块化分层（api/core/schemas）。
- **极高精准度的溯源**：独创**“物理页与印刷页动态对齐”**机制，确保返回给前端的页码（如 21 页）与 PDF 手册上印的页码 100% 匹配。
- **强抗干扰能力**：引入正则清洗与动态权重打分算法，自动拦截 PDF 目录页（满屏点号）及封底联系方式等“毒药”噪音，保障召回纯度。
- **双路召回融合**：向量语义检索 + BM25 关键词检索，手写修复 RRF 融合算法去重 Bug。
- **防幻觉与截断治理**：基于段落锚点切分，解决长文本跨页截断；强力 Prompt 约束 + 兜底清洗，杜绝小模型幻觉。
- **高并发异步处理**：全面使用 `run_in_threadpool` 解决同步 LLM 推理阻塞事件循环的问题。

## 快速启动
1. 克隆仓库：`git clone https://github.com/qq-exe/enterprise-rag-system.git`
2. 安装依赖：`pip install -r requirements.txt`
3. 启动服务：`uvicorn app.main:app --reload`
4. 访问 API 文档：`http://127.0.0.1:8000/docs`
