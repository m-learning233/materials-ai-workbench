# 🧪 材料AI研发工作台 (Materials AI Workbench)

这是一个面向材料科学研发场景的AI工具，旨在结合大语言模型（LLM）的文献理解能力与机器学习（ML）的预测能力，加速材料配方设计与筛选。

##  核心功能

- ** 智能对话助手**：支持多轮对话、流式输出（打字机效果），可选择 Qwen 或 DeepSeek 模型。
- ** 文献数据挖掘**：上传 PDF 文献（支持多选），自动切分长文本，调用大模型提取结构化数据（如改性剂、配方、性能提升），一键导出 CSV。
- ** 材料性能预测**：基于随机森林回归模型，用户输入特征参数即可实时预测目标性能（如拉伸强度）。

##  技术栈

- **UI 框架**: Gradio
- **大模型 API**: SiliconFlow (Qwen2.5-7B-Instruct, deepseek-ai/DeepSeek-V4-Flash)
- **机器学习**: scikit-learn (Random Forest), joblib, pandas, numpy
- **PDF 解析**: pypdf
- **工程化**: 模块化拆分 (pdf_tools.py, ml_tools.py, chat.py), Git 版本控制

##  快速开始

1. **克隆项目**
   ```bash
   git clone https://github.com/m-learning233/materials-ai-workbench.git
   cd materials-ai-workbench
2. **安装依赖**
   ```bash
   pip install -r requirements.txt
3. **配置api密钥**
   ```bash
   在根目录新建.env文件，填入你的密钥:SILICONFLOW_API_KEY=你的密钥
4. **准备模型和数据**
   将你的训练数据存为 material_data.csv(特征在前，目标值在最后一列)。
   运行python train_model.py 生成模型文件和特征配置。
5. **启动应用**
   ```bash
   python chat.py 在浏览器中打开即可使用。

##  项目结构
materials-ai-workbench/
├── chat.py              # 主入口，UI 界面与对话逻辑
├── pdf_tools.py         # PDF 解析、分块、大模型提取逻辑
├── ml_tools.py          # 模型加载与预测逻辑
├── train_model.py       # 模型训练脚本
├── requirements.txt     # 依赖清单
└── .gitignore           # Git 忽略规则（保护密钥）

##  作者
 孟令超
 Github：@m-learning233

##  许可证
MIT License