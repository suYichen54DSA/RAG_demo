# test_agent.py

import os
from agent import web_research_agent_research

def test_agent():
    """测试网络研究Agent功能"""
    
    # 测试配置
    API_KEY = "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"
    MODEL_NAME = "qwen/qwen2.5-vl-32b-instruct:free"
    BASE_URL = "https://openrouter.ai/api/v1"
    DOCS_DIR = "rag_word"
    
    # 测试查询
    test_query = "故宫博物院的历史和文化价值"
    
    print("=== 开始测试网络研究Agent ===")
    print(f"测试查询: {test_query}")
    print(f"文档保存目录: {DOCS_DIR}")
    
    try:
        # 执行网络研究
        saved_files = web_research_agent_research(
            query=test_query,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            base_url=BASE_URL,
            docs_dir=DOCS_DIR
        )
        
        print(f"\n=== 测试结果 ===")
        print(f"成功保存了 {len(saved_files)} 个文件:")
        
        for i, file_path in enumerate(saved_files, 1):
            print(f"  {i}. {os.path.basename(file_path)}")
            
            # 显示文件内容预览
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    preview = content[:200] + "..." if len(content) > 200 else content
                    print(f"     预览: {preview}")
            except Exception as e:
                print(f"     读取文件失败: {e}")
        
        print("\n测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent() 