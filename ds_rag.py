# ds_rag.py

import os
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import RetrievalQA

# 1. 环境 & 模型初始化
API_KEY = "your-api"
BASE_URL = "https://your/url/v1"

# BASE_URL = "https://openrouter.ai/api/v1"

LLM = ChatOpenAI(
    model="deepseek/deepseek-r1-0528:free",
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.0,
)

# 2. 加载文档
loader = DirectoryLoader(
    "rag_word/",  # 你放 .txt 文档的目录
    glob="**/*.txt",
    loader_cls=lambda path: TextLoader(path, encoding='utf-8')  # 指定 UTF-8 编码
)
docs = loader.load()

# 3. 切分文档片段
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)
print("📄 文档切片数:", len(splits))
if splits:
    print("示例片段内容：\n", splits[0].page_content)
input("✅ 文档切片完成，按回车继续")

# 4. 构建向量数据库（使用本地 HuggingFace Embedding）
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(splits, embeddings)
print("✅ FAISS 向量库构建完成")
input("按回车继续")

# 5. 构建 RAG 检索链
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
rag_chain = RetrievalQA.from_chain_type(
    llm=LLM,
    chain_type="stuff",  # 简单拼接检索到的片段
    retriever=retriever,
    return_source_documents=True,
)

# 6. 比较：直接 LLM vs RAG
question = "中国最著名的城市有哪些，举五个？不用具体介绍"

# 6a. 原始 LLM 回答
baseline_response = LLM.invoke([
    {"role": "system", "content": "你是一个知识渊博的中文助手"},
    {"role": "user", "content": question},
])
print("────────────────────────")
print("✨ Baseline 回答（不带检索）：")
print(baseline_response.content)
print()

# 6b. RAG 检索增强回答
rag_result = rag_chain({"query": question})
print("────────────────────────")
print("🚀 RAG 回答（带文档检索）：")
print(rag_result["result"])
print()

print("📚 使用到的文档片段：")
for doc in rag_result["source_documents"]:
    print("--------")
    print(doc.metadata.get("source", ""), ":\n", doc.page_content[:200].replace("\n", " ") + "…")
