# 增强型RAG系统 - 带网络研究Agent

这是一个基于LangChain框架的增强型RAG（检索增强生成）系统，集成了网络研究Agent功能。

## 功能特性

### 1. 网络研究Agent
- **自动网络爬取**: 基于用户查询自动在网络上搜索相关资料
- **智能内容分析**: 使用LangChain工具分析网页内容的相关性和质量
- **文档自动保存**: 将最相关的10份文档保存为txt格式到`rag_word`目录

### 2. 原有RAG功能
- **HyDE查询重写**: 使用假设文档嵌入方法重写用户查询
- **SRT文档排序**: 基于知识边界感知的相似度计算进行文档排序
- **MCT冗余过滤**: 多候选阈值过滤，去除冗余和高度相似的文档
- **多模态支持**: 支持文本和图像的多模态问答

## 项目结构

```
RAG/
├── main.py                 # 主程序入口
├── agent.py               # 网络研究Agent
├── query_rewriter.py      # HyDE查询重写器
├── similarity.py          # 知识边界感知相似度计算
├── rag_word/             # 文档存储目录
│   ├── *.txt             # 原始文档
│   └── web_research_*.txt # Agent爬取的文档
├── result/               # 结果输出目录
└── requirements.txt      # 依赖包列表
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 基本使用
直接运行main.py：
```bash
python main.py
```

### 2. 自定义查询
修改main.py中的`user_query`变量：
```python
user_query = "你的查询内容"
```

### 3. 配置API
在main.py中配置你的API信息：
```python
API_KEY = "你的API密钥"
MODEL_NAME = "qwen/qwen2.5-vl-32b-instruct:free"
BASE_URL = "https://openrouter.ai/api/v1"
```

## 工作流程

1. **网络研究阶段**: Agent自动搜索并爬取相关网页内容
2. **HyDE重写**: 使用假设文档嵌入方法重写用户查询
3. **文档检索**: 从本地文档库中检索相关文档
4. **SRT排序**: 基于相似度对文档进行排序
5. **MCT过滤**: 去除冗余文档
6. **多模态回答**: 结合图像和文本生成最终答案

## Agent工具说明

### WebScrapingTool
- 功能：抓取指定URL的网页内容
- 特点：自动处理编码、清理HTML标签、提取纯文本

### ContentAnalysisTool
- 功能：分析网页内容的相关性和质量
- 评分：基于关键词匹配和内容长度计算综合评分

### DuckDuckGoSearchRun
- 功能：执行网络搜索
- 特点：无需API密钥，支持多种搜索查询

## 输出结果

系统会在`result`目录下生成两个文件：
- `baseline_answer.txt`: 基线回答（仅使用图像）
- `hyde_rag_answer.txt`: HyDE+RAG增强回答

## 注意事项

1. 确保网络连接正常，Agent需要访问网络进行搜索
2. API密钥需要有效且有足够的配额
3. 爬取的文档会自动保存到`rag_word`目录
4. 系统会自动处理编码和格式问题

## 扩展功能

- 支持自定义搜索关键词
- 可配置爬取文档数量
- 支持多种文档格式
- 可扩展更多分析工具 