from typing import List
from langchain_core.documents import Document
from app.config import settings

def rrf_merge(docs_list: List[List[Document]], k: int=60) -> List[Document]:
    # RRF(Reciprocal Rank Fusion)融合算法：将向量检索和BM25检索结果合并排序
    score_map = {}
    for doc_list in docs_list:
        # 绝不能用id(doc)做唯一标识，因为向量检索和BM25返回的文档对象内存地址不同
        #即使内容一样，id()也不同，会导致RRF无法去重。
        #改用基于文本内容的hash做唯一标识，精准去重
        for rank, doc in enumerate(doc_list):
            doc_id = hash(doc.page_content.strip())
            if doc_id not in score_map:
                score_map[doc_id] = {"doc": doc, "score": 0.0}
            # RRF核心公式1/(k+rank)
            score_map[doc_id]["score"] += 1.0 / (k + rank +1)

    # 按得分降序排序，并截取前TOP_K 个文档
    sorted_items = sorted(score_map.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_items][:settings.TOP_K]

def get_retrieval_context(question: str, rag_service) -> List[Document]:
    # 执行双路召回（向量+BM25），并返回RRF融合后的结果
    # 向量检索：带相似度分数，并做阈值过滤
    vector_docs_with_score = rag_service.vector_db.similarity_search_with_relevance_scores(
        question,k=15)
    vector_docs = [doc for doc, score in vector_docs_with_score if score > 0.15]
    # BM25关键词检索
    bm25_docs = rag_service.bm25_retriever.invoke(question)
    # RRF 融合后，把 limit 也调大，暂不截断
    merged = rrf_merge([vector_docs, bm25_docs])
    
    # 这里直接返回合并后的前 10 个
    return merged[:10]