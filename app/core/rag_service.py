import re
import logging
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.config import settings

logger = logging.getLogger(__name__)

#全局预加载Embedding模型，避免每次请求加载，极大节省时间
# 使用HuggingFaceEmbeddings替代弃用的HuggingFaceBgeEmbeddings
embedding_model = HuggingFaceEmbeddings(
    model_name=settings.EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": "True"},
)

class RAGService:
    """RAG核心服务类：封装文档解析、清洗、切分、入库逻辑"""
    def __init__(self):
        # 用类属性代替全局变量，避免多线程下的状态污染
        self.vector_db = None
        self.bm25_retriever = None

    def _clean_text(self, text: str) -> str:
        # PDF解析出的数据含有大量无关内容，清晰避免污染向量空间
        # 过滤HTML实体，PDF排版残留
        text = text.replace("\u003c","<").replace("\u003e",">")
        # 过滤【1082】这种学术参考文献编号
        text = re.sub(r'\[\d+\]', '', text)
        # 过滤URL链接
        text = re.sub(r'https?://\S+', '', text)
        return text
    
    def ingest_pdf(self, file_path: str) -> int:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        
        final_splits = []
        # 不搞合并了，直接遍历每一页
        for idx, doc in enumerate(docs):
            cleaned = self._clean_text(doc.page_content)
            # 如果这页全是垃圾，跳过
            if not cleaned.strip(): 
                continue
            
            # 把每页当成一个独立的块，强制打上页码标签
            page_doc = Document(
                page_content=f"【第{idx+1}页】\n{cleaned}",
                metadata={"source": doc.metadata.get("source", "未知文件")}
            )
            final_splits.append(page_doc)

        # 如果某一页特别大（比如好几十页），我们再用 splitter 把它切小，但不需要重叠了
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
        )
        final_splits = splitter.split_documents(final_splits)
        final_splits = [s for s in final_splits if len(s.page_content.strip()) > 20]

        self.vector_db = Chroma.from_documents(
            documents=final_splits,
            embedding=embedding_model,
            persist_directory=settings.CHROMA_PERSIST_DIR
        )
        # ... 后面的 BM25 代码保持不变 ...

        # 构建BM25关键词检索器
        self.bm25_retriever = BM25Retriever.from_documents(final_splits)
        self.bm25_retriever.k = settings.TOP_K

        return len(final_splits)

# 单条实例，全局唯一，避免重复初始化
rag_service = RAGService()
