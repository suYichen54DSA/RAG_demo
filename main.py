# main.py

import os
from langchain_openai import ChatOpenAI
from query_rewriter import QueryRewriter
from similarity import KnowledgeBoundaryAwareSimilarity
from agent import web_research_agent_research

def main():
    # === 一、初始化 ===
    API_KEY    = "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"
    MODEL_NAME = "qwen/qwen2.5-vl-32b-instruct:free"
    BASE_URL   = "https://openrouter.ai/api/v1"
    DOCS_DIR   = "rag_word/"

    IMAGE_URL = (
        "https://bkimg.cdn.bcebos.com/pic/"
        "d50735fae6cd7b899e51ed97e67c55a7d933c895a0d4"
        "?x-bce-process=image/format,f_auto/quality,Q_70"
    )
    user_query = "帮我基于这张图片，写一段微信公众号的介绍文案"

    # 0) 网络研究Agent - 自动爬取相关资料
    print("=== 开始网络研究，自动爬取相关资料 ===")
    try:
        saved_files = web_research_agent_research(
            query=user_query,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            base_url=BASE_URL,
            docs_dir=DOCS_DIR
        )
        print(f"成功爬取并保存了 {len(saved_files)} 份文档")
        for file_path in saved_files:
            print(f"  - {os.path.basename(file_path)}")
    except Exception as e:
        print(f"网络研究过程中出现错误: {e}")
        print("继续执行原有逻辑...")

    # 1) HyDE 查询重写
    rewriter = QueryRewriter(
        api_key=API_KEY,
        model_name=MODEL_NAME,
        base_url=BASE_URL,
        docs_dir=DOCS_DIR
    )
    hyde_q = rewriter.rewrite_query_with_hyde(user_query, k=3)

    # 2) 多模态 RAG LLM 客户端
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.0,
        max_tokens=2048,
    )

    # === Baseline ===
    baseline = llm.invoke([{
        "role": "user",
        "content": [
            {"type": "text",      "text": user_query},
            {"type": "image_url", "image_url": {"url": IMAGE_URL}}
        ]
    }])
    result_dir = r"D:\AAAAA 安克创新\RAG\result"
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "baseline_answer.txt"), "w", encoding="utf-8") as f:
        f.write("── Baseline 回答 ──\n")
        f.write(baseline.content + "\n")

    # === HyDE + RAG with SRT & MCT ===

    # SRT 阶段：检索更多候选文档
    retriever = rewriter.vectorstore.as_retriever(search_kwargs={"k": 5})
    candidate_docs = retriever.invoke(hyde_q)

    # 初始化相似度计算器
    sim_calc = KnowledgeBoundaryAwareSimilarity(docs_dir=DOCS_DIR)

    # 选取第一个候选作为基准源
    base_src = candidate_docs[0].metadata["source"]

    # 排序并取 Top-3
    ranked = sim_calc.rank_similar_documents(base_src, top_k=3)
    top3_srcs = [src for src, _ in ranked]
    srt_docs = [d for d in candidate_docs if d.metadata["source"] in top3_srcs]

    # MCT 阶段：剔除冗余与高度相似的文档
    filtered = []
    for doc in srt_docs:
        src = doc.metadata["source"]
        if all(sim_calc.compute_max_similarity(src, kept.metadata["source"]) < 0.95
               for kept in filtered):
            filtered.append(doc)

    # 拼接上下文：先是过滤后文档，再加上 Hyde 生成的 query
    context = "\n\n---\n\n".join(d.page_content for d in filtered)
    final_query = context + "\n\n" + hyde_q

    # 最终 RAG 调用
    rag_input = [
        {"role": "system",
         "content": "你是一个熟练的公众号写手，你将根据检索到的信息，请结合以下要求和图片生成一段微信公众号文案，注意以下三点：1.文辞恰当，逻辑严密，不能出现冗余片段；2.模仿人类口吻生成，注意生成的文字切合人类逻辑；3.不能使用任何攻击、对立、政治敏感等措辞，写这篇公众号。"},
        {"role": "user",
         "content": [
             {"type": "text",      "text": final_query},
             {"type": "image_url", "image_url": {"url": IMAGE_URL}}
         ]}
    ]
    rag_resp = llm.invoke(rag_input)
    with open(os.path.join(result_dir, "hyde_rag_answer.txt"), "w", encoding="utf-8") as f:
        f.write("── HyDE+RAG（SRT+MCT）回答 ──\n")
        f.write(rag_resp.content + "\n")

if __name__ == "__main__":
    main()
