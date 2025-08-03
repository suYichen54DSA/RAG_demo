# agent.py

import os
import re
import time
import requests
from typing import List, Dict, Any
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool
from langchain.schema import BaseMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from ddgs import DDGS
import json


class WebScrapingTool(BaseTool):
    """网页抓取工具"""
    name: str = "web_scraping"
    description: str = "抓取指定URL的网页内容，提取文本信息"
    
    def _run(self, url: str) -> str:
        """执行网页抓取"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 移除脚本和样式标签
            for script in soup(["script", "style"]):
                script.decompose()
            
            # 提取文本内容
            text = soup.get_text()
            
            # 清理文本
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # 限制长度
            if len(text) > 5000:
                text = text[:5000] + "..."
                
            return text
            
        except Exception as e:
            return f"抓取失败: {str(e)}"


class ContentAnalysisTool(BaseTool):
    """内容分析工具"""
    name: str = "content_analysis"
    description: str = "分析网页内容的相关性和质量"
    
    def _run(self, content: str, query: str) -> str:
        """分析内容与查询的相关性"""
        # 简单的关键词匹配分析
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        # 计算重叠度
        overlap = len(query_words.intersection(content_words))
        relevance_score = overlap / len(query_words) if query_words else 0
        
        # 内容长度评分
        length_score = min(len(content) / 1000, 1.0)
        
        # 综合评分
        total_score = (relevance_score * 0.7 + length_score * 0.3)
        
        return json.dumps({
            "relevance_score": round(relevance_score, 3),
            "length_score": round(length_score, 3),
            "total_score": round(total_score, 3),
            "content_length": len(content)
        })


class WebResearchAgent:
    """基于LangChain的网络研究Agent"""
    
    def __init__(self, api_key: str, model_name: str, base_url: str):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            max_tokens=2048,
        )
        
        # 初始化搜索工具
        self.ddgs = DDGS()
        self.scraping_tool = WebScrapingTool()
        self.analysis_tool = ContentAnalysisTool()
    
    def research_and_save(self, query: str, output_dir: str = "rag_word", max_docs: int = 10) -> List[str]:
        """执行研究并保存文档"""
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"开始搜索相关网页: {query}")
        
        # 1. 使用搜索工具搜索相关网页
        try:
            search_results = []
            with self.ddgs as ddgs:
                for r in ddgs.text(query, max_results=20):
                    search_results.append({
                        'title': r.get('title', ''),
                        'url': r.get('link', ''),
                        'snippet': r.get('body', '')
                    })
            print(f"搜索完成，找到 {len(search_results)} 个结果")
        except Exception as e:
            print(f"搜索失败: {e}")
            return []
        
        # 2. 提取URL
        urls = [result['url'] for result in search_results if result['url']]
        print(f"找到 {len(urls)} 个有效URL")
        
        # 3. 抓取网页内容
        scraped_contents = []
        for i, url in enumerate(urls[:max_docs*2]):  # 抓取更多URL以便筛选
            try:
                print(f"抓取第 {i+1} 个网页: {url}")
                content = self.scraping_tool.run(url)
                if content and not content.startswith("抓取失败"):
                    # 分析内容相关性
                    analysis = self.analysis_tool.run(content, query)
                    scraped_contents.append({
                        'url': url,
                        'content': content,
                        'analysis': analysis
                    })
            except Exception as e:
                print(f"抓取失败 {url}: {e}")
                continue
        
        # 4. 按相关性排序并保存
        saved_files = self._save_top_contents(scraped_contents, output_dir, max_docs)
        
        return saved_files
    

    
    def _save_top_contents(self, scraped_contents: List[Dict], output_dir: str, max_docs: int) -> List[str]:
        """保存最相关的内容"""
        saved_files = []
        
        # 按相关性排序
        def get_relevance_score(item):
            try:
                analysis = json.loads(item['analysis'])
                return analysis.get('total_score', 0)
            except:
                return 0
        
        sorted_contents = sorted(scraped_contents, key=get_relevance_score, reverse=True)
        
        # 保存前max_docs个
        for i, item in enumerate(sorted_contents[:max_docs], 1):
            try:
                filename = f"web_research_{i}.txt"
                filepath = os.path.join(output_dir, filename)
                
                # 生成摘要
                summary_prompt = f"请为以下内容生成一个简短的摘要（50字以内）：\n\n{item['content'][:1000]}"
                try:
                    summary_response = self.llm.invoke([{"role": "user", "content": summary_prompt}])
                    summary = summary_response.content.strip()
                except:
                    summary = "内容摘要生成失败"
                
                # 生成标题
                title_prompt = f"请为以下内容生成一个简短的标题（20字以内）：\n\n{item['content'][:500]}"
                try:
                    title_response = self.llm.invoke([{"role": "user", "content": title_prompt}])
                    title = title_response.content.strip()
                except:
                    title = f"网页内容 {i}"
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"标题：{title}\n")
                    f.write(f"来源：{item['url']}\n")
                    f.write(f"摘要：{summary}\n")
                    f.write(f"抓取时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"相关性评分：{get_relevance_score(item):.3f}\n")
                    f.write("-" * 50 + "\n")
                    f.write(item['content'][:3000])  # 限制内容长度
                
                saved_files.append(filepath)
                print(f"已保存：{filename}")
                
            except Exception as e:
                print(f"保存文件 {i} 时出错：{e}")
                continue
        
        return saved_files


def web_research_agent_research(query: str, api_key: str, model_name: str, base_url: str, docs_dir: str = "rag_word") -> List[str]:
    """
    执行网络研究并保存文档的主函数
    
    Args:
        query: 研究查询
        api_key: API密钥
        model_name: 模型名称
        base_url: API基础URL
        docs_dir: 文档保存目录
    
    Returns:
        保存的文件路径列表
    """
    agent = WebResearchAgent(api_key, model_name, base_url)
    return agent.research_and_save(query, docs_dir, max_docs=10) 