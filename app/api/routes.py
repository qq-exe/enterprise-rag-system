import uuid
import os
import time
import re
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool #[核心]引入线程池，解决同步阻塞
import ollama
from app.core.rag_service import rag_service
from app.core.retriever import get_retrieval_context
from app.schemas import UploadResponse, ChatResponse, HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload_pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
     # ✅ 企业级修复：使用 uuid 生成安全的临时文件名，保留原后缀
    file_extension = os.path.splitext(file.filename)[1] or ".pdf"
    temp_path = f"./tmp_{uuid.uuid4().hex}{file_extension}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    try:
        # 【工程考量】PyMuPDFLoader 和 Chroma 都是同步阻塞操作，
        # 必须用run_in_threadpool 丢进线程池，否则会卡死FastAPI的事件循环，导致并发能力极差。
        chunk_count = await run_in_threadpool(rag_service.ingest_pdf, temp_path)
        return UploadResponse(msg="文档上传成功", chunk_count=chunk_count)
    finally:
        #无论成功与否，都删除临时文件，防止磁盘爆满
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.get("/rag_chat", response_model=ChatResponse)
async def rag_chat(question: str):
    """RAG问答接口：检索+生成+溯源"""
    if not rag_service.vector_db:
        raise HTTPException(status_code=400, detail="知识库为空，请先上传PDF!")
    logger.info(f"用户提问：{question}")
    # 检索召回
    raw_docs = await run_in_threadpool(get_retrieval_context, question, rag_service)
    context_text = "\n\n".join([d.page_content for d in raw_docs])

    # [Prompt优化] 软拒答引导+禁止编造+禁止重复输出
    prompt = f"""你是一个严谨的企业设备手册助手。请严格基于下面提供的知识库内容回答用户问题。
如果知识库内容中找不到答案，必须直接回答“根据已有知识库无法回答该问题”，绝对严禁自己编造任何步骤！
如果找到了答案，请在你的回答最后，单独加上一行，格式必须一字不差地为：【参考页码：第X页】

知识库内容：
{context_text}

用户问题：{question}
"""


    # [参数调优]调用本地Ollama模型，加入repeat_penalty 解决“复读机”问题
    resp = await run_in_threadpool(
        ollama.chat,
        model="qwen2.5:3b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2, "repeat_penalty": 1.15, "num_predict": 1000}
    )  

    # 【溯源信息组装 - 终极物理+印刷页对齐版】
    answer_content = resp["message"]["content"]
    references = []
    
    if raw_docs:
        clean_question = re.sub(r'[^\w\u4e00-\u9fa5]', '', question)
        keywords = [w for w in re.findall(r'[\u4e00-\u9fa5]{2,}', clean_question)]
        if not keywords:
            keywords = [clean_question]
            
        scored_docs = []
        for doc in raw_docs:
            content = doc.page_content
            if re.search(r'[\.\。]{5,}', content):
                continue
            if "电话" in content and "邮箱" in content:
                continue
            if "邮编" in content and "Email" in content:
                continue
                
            score = sum(content.count(kw) for kw in keywords)
            
            # 1. 先抓 PDF 物理页码
            page_match = re.search(r'【第\s*(\d+)\s*页】', content)
            physical_page = page_match.group(1) if page_match else "未知"
            
            # 2. 尝试抓取文档里印着的真实页码（比如 `- 21 -` 或者 `第 21 页`）
            printed_page_match = re.search(r'-\s*(\d+)\s*-', content)
            if not printed_page_match:
                printed_page_match = re.search(r'第\s*(\d+)\s*页', content)
            
            # 3. 如果找到了印刷页码，就用印刷页码；没找到就用物理页码
            final_page_num = printed_page_match.group(1) if printed_page_match else physical_page
            
            scored_docs.append({"score": score, "doc": doc, "page_num": final_page_num})
            
        # 4. 按得分降序排，只取第一名！
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        
        # 只要得分最高的那 1 个，如果得分大于 0，就绝不返回第二个干扰项
        for item in scored_docs[:1]:  
            content = item["doc"].page_content
            references.append({
                "source_file": item["doc"].metadata.get("source", "").split("/")[-1],
                "page_number": item["page_num"],
                "chunk_text": re.sub(r'[\.\。]{5,}', '...', content[:250]).replace("\n", " ").replace("\\n", " ")
            })

    # 清洗换行符，并强行抹除大模型自己瞎编的页码字样
    clean_answer = answer_content.replace("\\n", "\n").replace("\n\n", "\n")
    clean_answer = re.sub(r'【参考页码：第\s*\d+\s*页】', '', clean_answer).strip()
    
    return ChatResponse(answer=clean_answer, references=references)

@router.get("/health", response_model=HealthResponse)
async def health():
    """健康检查接口，用于运维监控"""
    return HealthResponse(status="running", time=time.time())