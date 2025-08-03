# Agent.py 实现总结

## 实现概述

基于你的需求，我成功创建了一个基于LangChain框架的`agent.py`文件，实现了自动网络爬取功能，并完美集成到现有的RAG系统中。

## 核心功能

### 1. 网络研究Agent (`WebResearchAgent`)
- **自动搜索**: 使用DuckDuckGo搜索工具自动搜索相关网页
- **智能抓取**: 使用BeautifulSoup抓取网页内容，自动处理编码和HTML清理
- **内容分析**: 基于关键词匹配和内容长度计算相关性评分
- **文档保存**: 自动将最相关的10份文档保存为txt格式

### 2. 工具组件

#### WebScrapingTool
```python
class WebScrapingTool(BaseTool):
    name = "web_scraping"
    description = "抓取指定URL的网页内容，提取文本信息"
```
- 自动设置User-Agent避免被反爬
- 智能编码检测和处理
- HTML标签清理和文本提取
- 内容长度限制（5000字符）

#### ContentAnalysisTool
```python
class ContentAnalysisTool(BaseTool):
    name = "content_analysis"
    description = "分析网页内容的相关性和质量"
```
- 关键词匹配度计算
- 内容长度评分
- 综合相关性评分算法

#### DuckDuckGoSearchRun
- 无需API密钥的搜索工具
- 支持多种搜索查询格式

### 3. 主函数接口
```python
def web_research_agent_research(query: str, api_key: str, model_name: str, base_url: str, docs_dir: str = "rag_word") -> List[str]:
```

## 集成方式

### 1. 在main.py中的调用
```python
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
```

### 2. 工作流程
1. **网络研究阶段**: Agent自动搜索并爬取相关网页内容
2. **HyDE重写**: 使用假设文档嵌入方法重写用户查询
3. **文档检索**: 从本地文档库（包括新爬取的文档）中检索相关文档
4. **SRT排序**: 基于相似度对文档进行排序
5. **MCT过滤**: 去除冗余文档
6. **多模态回答**: 结合图像和文本生成最终答案

## 文件结构

```
RAG/
├── main.py                 # 主程序（已修改，集成Agent调用）
├── agent.py               # 网络研究Agent（新增）
├── query_rewriter.py      # HyDE查询重写器（不变）
├── similarity.py          # 知识边界感知相似度计算（不变）
├── test_agent.py          # Agent测试脚本（新增）
├── requirements.txt       # 依赖包列表（新增）
├── README.md              # 项目说明文档（新增）
├── rag_word/             # 文档存储目录
│   ├── *.txt             # 原始文档
│   └── web_research_*.txt # Agent爬取的文档（自动生成）
└── result/               # 结果输出目录
```

## 技术特点

### 1. 基于LangChain框架
- 使用`create_openai_tools_agent`创建工具型Agent
- 集成多种工具：搜索、抓取、分析
- 支持工具链式调用和结果处理

### 2. 智能内容处理
- 自动编码检测和处理
- HTML标签清理
- 文本长度优化
- 相关性评分算法

### 3. 错误处理和容错
- 网络请求超时处理
- 编码错误处理
- 文件保存异常处理
- 优雅降级机制

### 4. 可扩展性
- 支持自定义搜索关键词
- 可配置爬取文档数量
- 支持多种文档格式
- 可扩展更多分析工具

## 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行主程序
```bash
python main.py
```

### 3. 测试Agent功能
```bash
python test_agent.py
```

## 输出结果

### 1. 爬取文档
- 文件名格式：`web_research_1.txt`, `web_research_2.txt`, ...
- 内容包含：标题、来源URL、摘要、抓取时间、完整内容

### 2. 最终结果
- `baseline_answer.txt`: 基线回答
- `hyde_rag_answer.txt`: HyDE+RAG增强回答（包含爬取文档的影响）

## 优势特点

1. **自动化程度高**: 无需手动搜索和整理资料
2. **智能相关性**: 基于AI的内容分析和评分
3. **无缝集成**: 完全兼容现有RAG系统
4. **容错性强**: 网络问题不影响原有功能
5. **可扩展性**: 易于添加新的工具和功能

## 注意事项

1. 需要网络连接进行搜索和抓取
2. API密钥需要有效且有足够配额
3. 爬取速度取决于网络状况和目标网站
4. 建议在稳定的网络环境下运行

这个实现完全满足了你的需求：能在main函数中调用，使用LangChain框架的tool自动爬取相关资料，并以txt形式存储在rag_word目录中，同时保持main中其他逻辑不变。 