# debug_search.py

from ddgs import DDGS

def debug_search():
    query = "介绍一下这张图片的景点和它的历史渊源"
    
    with DDGS() as ddgs:
        results = []
        for r in ddgs.text(query, max_results=5):
            results.append(r)
            print(f"结果: {r}")
            print(f"  标题: {r.get('title', 'N/A')}")
            print(f"  URL: {r.get('link', 'N/A')}")
            print(f"  摘要: {r.get('body', 'N/A')[:100]}...")
            print("-" * 50)

if __name__ == "__main__":
    debug_search() 