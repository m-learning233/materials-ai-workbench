import os
os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.getcwd(), "gradio_temp")
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True) # 自动创建临时文件夹

import json
import re
from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr
import time
import csv

#导入工具包
from pdf_tools import process_multiple_pdfs
from ml_tools import predict_features, feature_names, target_name

load_dotenv()
api_key = os.getenv("SILICONFLOW_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1"
)

# 对话核心函数
def chat_with_ai(user_input, history, model_name):
    messages = [{"role": "system", "content": "你是材料研发助手，熟悉高分子、金属、无机非金属等材料的配方设计与性能分析"}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})

    stream = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.3,
        stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

# 用户发送消息（UI辅助函数）
def user_msg(message, history):
    return "", history + [{"role": "user", "content": message}]

# 模型回复
def bot_response(history, model_name):
    if not history:
        return history
    user_input = history[-1]["content"]
    prev_history = history[:-1]

    history.append({"role": "assistant", "content": ""})
    yield history

    t0 = time.time()
    for delta in chat_with_ai(user_input, prev_history, model_name):
        history[-1]["content"] += delta
        yield history
    
    time_cost = time.time() - t0
    history[-1]["content"] += f"\n\n耗时 {time_cost:.2f} 秒"
    yield history

def export_to_csv(data):
    if data is None or data.empty:
        return None
    #生成一个带时间戳的文件名，防止覆盖
    file_path = "extraction_result.csv"
    #写入csv
    with open(file_path, mode='w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        if hasattr(data, 'columns') and hasattr(data , 'values'):
           writer.writerow(data.columns.tolist())
           for row in data.values.tolist():
               writer.writerow(row)
        else :
           writer.writerow(["改性剂名称","添加量","基体配方","性能提升幅度","测试标准","实验条件"])
           for row in data:
               writer.writerow(row)
    return file_path #返回文件路径给DownloadButton
# ==========================================
# 2. 搭建网页界面
# ==========================================
with gr.Blocks(title="材料AI研发工作台") as demo:
    gr.Markdown("## 🧪 材料研发工作台")
    gr.Markdown("支持多轮对话，流式输出 | 模型：Qwen2.5-7B-Instruct, deepseek-ai/DeepSeek-V4-Flash")
    gr.Markdown("作者：孟令超")

    # ------------------ 房间1：智能对话助手 ------------------
    with gr.Tab(" 智能对话助手"):
        chatbot = gr.Chatbot(height=400)
        model_dd = gr.Dropdown(
            choices=["Qwen/Qwen2.5-7B-Instruct", "deepseek-ai/DeepSeek-V4-Flash"],
            value="Qwen/Qwen2.5-7B-Instruct",
            label="选择大模型"
        )
        msg = gr.Textbox(label="输入你的问题", placeholder="例如：TDI和MDI有什么区别？")
        clear_btn = gr.Button("清空对话")

        # 事件绑定
        msg.submit(user_msg, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response, [chatbot, model_dd], chatbot
        )
        clear_btn.click(lambda: None, None, chatbot, queue=False)

    # ------------------ 房间2：文献改性剂挖掘 ------------------
    with gr.Tab(" 文献数据挖掘助手"):
        gr.Markdown("### 上传文献PDF，AI自动提取改性剂配方与性能数据")
        
        with gr.Row():
            file_input = gr.File(label="上传PDF文献(支持多选)", file_types=[".pdf"],file_count="multiple",type="binary")
            extract_model_dd = gr.Dropdown(
                choices=["Qwen/Qwen2.5-7B-Instruct",  "deepseek-ai/DeepSeek-V4-Flash"],
                value="Qwen/Qwen2.5-7B-Instruct", # 默认用模型
                label="选择提取模型"
    )
            extract_btn = gr.Button("开始AI提取", variant="primary")
            
        result_table = gr.Dataframe(
            headers=["改性剂名称", "添加量", "基体配方", "性能提升幅度", "测试标准", "实验条件"],
            label="提取结果"
        )
        download_btn = gr.DownloadButton("导出为CSV", visible=False)#初始隐藏
        # 事件绑定
        extract_btn.click(process_multiple_pdfs, inputs=[file_input, extract_model_dd], outputs=[result_table]).then(
            lambda: gr.update(visible=True), #提取成功后，让下载按钮出现
            outputs=[download_btn]
        )
        download_btn.click(
            export_to_csv,
            inputs=[result_table],
            outputs=[download_btn]
        )


    # ------------------ 房间3：材料性能预测 ------------------
    with gr.Tab(" 材料性能预测"):

        #导入配置中的名称，动态生成界面提示
        if feature_names:
            feature_desc = " , ".join(feature_names) 
        else :
            "特证1 , 特证2,,'''"

        gr.Markdown(f"### 输入材料特征参数，预测**{target_name}**")
        gr.Markdown(f"严格按照以下顺序输入（用逗号分隔）:**\n`{feature_desc}`")

    
        feature_input = gr.Textbox(
            label="特征参数（逗号分隔）", 
            placeholder=f"例如:{ ' , '.join(['30']* len(feature_names)) if feature_names else '30, 120'}"
        )
    
        predict_btn = gr.Button("开始预测", variant="primary")
        predict_output = gr.Textbox(label="预测结果", interactive=False)

        predict_btn.click(
            predict_features,
            inputs=[feature_input],
            outputs=[predict_output]
        )

demo.launch()
