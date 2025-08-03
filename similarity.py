# similarity.py

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class KnowledgeBoundaryAwareSimilarity:
    def __init__(self,
                 docs_dir: str,
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # 加载与切分文档
        loader = DirectoryLoader(
            docs_dir,
            glob="**/*.txt",
            loader_cls=lambda p: TextLoader(p, encoding="utf-8")
        )
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)

        # 初始化嵌入模型
        self.emb = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.doc_texts = [d.page_content for d in splits]
        self.doc_sources = [d.metadata["source"] for d in splits]
        self.vectors = self.emb.embed_documents(self.doc_texts)

        # 建立 source -> slices 的映射
        self.doc_index = {}
        for i, src in enumerate(self.doc_sources):
            self.doc_index.setdefault(src, []).append(i)

    def compute_max_similarity(self, src1: str, src2: str) -> float:
        """基于知识边界感知计算最大余弦相似度"""
        idxs1 = self.doc_index.get(src1, [])
        idxs2 = self.doc_index.get(src2, [])
        if not idxs1 or not idxs2:
            raise ValueError(f"未找到文档 {src1} 或 {src2} 的切片")

        vecs1 = np.array([self.vectors[i] for i in idxs1])
        vecs2 = np.array([self.vectors[i] for i in idxs2])

        # 计算 pairwise cosine similarity
        norms = np.linalg.norm(vecs1, axis=1)[:, None] * np.linalg.norm(vecs2, axis=1)[None, :]
        sim_matrix = (vecs1 @ vecs2.T) / norms
        return float(np.max(sim_matrix))

    def rank_similar_documents(self, target_src: str, top_k=5):
        """返回与目标文档最相似的前K个文档源路径（用于SRT过滤前排序）"""
        if target_src not in self.doc_index:
            raise ValueError(f"目标文档 {target_src} 不存在")

        similarities = []
        for other_src in set(self.doc_sources):
            if other_src == target_src:
                continue
            sim = self.compute_max_similarity(target_src, other_src)
            similarities.append((other_src, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def list_documents(self):
        return sorted(set(self.doc_sources))
