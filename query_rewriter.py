# query_rewriter.py

import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader


class QueryRewriter:
    def __init__(self,
                 api_key: str,
                 model_name: str,
                 base_url: str,
                 docs_dir: str):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            max_tokens=2048,
        )

        # 构建向量库
        loader = DirectoryLoader(
            docs_dir,
            glob="**/*.txt",
            loader_cls=lambda p: TextLoader(p, encoding="utf-8"),
        )
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_documents(splits, embeddings)

    def rewrite_query_with_hyde(self,
                                original_query: str,
                                k: int = 3,
                                max_length: int = 512) -> str:
        """使用 HyDE 方法重写查询。"""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        initial_docs = retriever.invoke(original_query)[:k]
        initial_context = "\n\n---\n\n".join(d.page_content for d in initial_docs)

        hyde_prompt = (
            "请根据以下检索到的文档片段，以及用户的原始问题，"
            "撰写一段更详尽、信息丰富的“假设文档”来回答该问题：\n\n"
            f"【检索上下文】\n{initial_context}\n\n"
            f"【原始问题】\n{original_query}\n\n"
            "请以段落形式输出，不要包含“假设”二字，也不输出多余说明。"
        )
        response = self.llm.invoke([{"role": "user", "content": hyde_prompt}],
                                   max_tokens=max_length)
        hypothetical_doc = response.content.strip()

        rewritten_query = (
            f"{hypothetical_doc}\n\n"
            f"基于上述内容，请对“{original_query}”做更准确的回答。"
        )
        return rewritten_query
