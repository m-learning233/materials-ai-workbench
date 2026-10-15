import os
import json
import re
import io
import time
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# 加载密钥
load_dotenv()
api_key = os.getenv("SILICONFLOW_API_KEY")
if not api_key:
    raise ValueError("未找到 SILICONFLOW_API_KEY，请检查 .env")

# 初始化客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1"
)

# 滑动窗口确保文献阅读完全
def split_text(text, chunk_size=3000, overlap=300):
    """将长文本切为待有重叠的块，保证提取数据不丢失"""
    if not text:
        return[]
    chunks = []
    start = 0
    text_length = len(text)

    while   start < text_length:
        end = min(start + chunk_size , text_length)
        chunks.append(text[start:end])

        if end == text_length:
            break
        start = end - overlap #滑动窗口逻辑：往前倒推overlap个字符
    return chunks

# 核心：让大模型把文本变成表格
def extract_info_from_text(text, model_name):
    if not text:
        return "[]"
    
    prompt = f"""
    你是一个材料科学的数据提取专家。请从以下相关文献片段中，提取相关实验数据。
    需要提取的字段包括：
    1. 改性剂名称
    2. 改性剂添加量
    3. 基体的配方
    4. 性能提升幅度（必须是极简模式，例如：拉伸强度：300→500 Mpa）
    5. 测试标准（例如：ASTM D638）
    6. 实验条件（例如：温度、反应时间等）
    
     严格格式要求：
    - 必须严格返回 JSON 数组格式！即最外层必须是中括号 [ ]！
    - 即使只提取到一条数据，也请用 [ {{...}} ] 包裹！
    - 绝对不要输出任何解释文字，不要加 ```json 标记！
    - JSON 字段名称必须是：modifier_name, dosage, base_formula, performance_improvement, test_standard, experimental_conditions。
    - 如果某字段在文中确实找不到，才填写“未提及”。

    文献内容如下：
    ---
    {text}
    ---
    """
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        
        raw_content = resp.choices[0].message.content

        if not raw_content:
            return "[]" 

        #  1. 暴力清洗：去掉所有 Markdown 代码块标记
        raw_content = raw_content.replace("```json", "").replace("```", "").strip()

        #  2. 智能匹配：不管模型吐的是 [] 还是 {}，我们都尽量匹配
        match_array = re.search(r'\[.*\]', raw_content, re.DOTALL)
        match_object = re.search(r'\{.*\}', raw_content, re.DOTALL)
        
        if match_array:
            return match_array.group(0)
        elif match_object:
            #  3. 如果模型只返回了单个对象 {}，我们强行给它套上 [] 变成数组
            clean_obj = match_object.group(0)
            return f"[{clean_obj}]"
        else:
            print(f"  模型完全没有输出 JSON 格式: {raw_content[:100]}")
            return "[]"
            
    except Exception as e:
        return "[]"

def process_multiple_pdfs(files, model_name):
    if not files:
        return [["请先上传文件", "", "", "", "", ""]]

    all_rows = [] # 用来存放所有文件提取出的数据
    
    # 遍历每一个上传的文件
    for i, file_obj in enumerate(files):
        file_name = f"文件{i+1}"
        print(f"\n 开始处理 {file_name}...")
        
        try:
            # ==========================================
            # 1. 绝对防御：强制拿到 PDF 的字节数据
            # ==========================================
            pdf_bytes = b""
            if isinstance(file_obj, bytes):
                pdf_bytes = file_obj
            elif isinstance(file_obj, dict):
                with open(file_obj.get("path", file_obj.get("name")), 'rb') as f:
                    pdf_bytes = f.read()
            else:
                with open(file_obj, 'rb') as f:
                    pdf_bytes = f.read()

            # ==========================================
            # 2. 解析 PDF，提取全文本（不截断）
            # ==========================================
            import io
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            # 防御空文本（扫描件情况）
            if not text.strip():
                all_rows.append([f" {file_name}: PDF内容为空或为扫描件", "", "", "", "", ""])
                continue

            # ==========================================
            # 3. 核心：分块处理（滑动窗口）
            # ==========================================
            chunks = split_text(text, chunk_size=3000, overlap=300)
            print(f" {file_name} 共拆分为 {len(chunks)} 个文本块，开始分批提取...")
            
            file_data = [] # 存放当前文件提取出的所有数据
            
            for j, chunk in enumerate(chunks):
                print(f"   正在处理 第 {j+1}/{len(chunks)} 块, 使用模型：{model_name}...")
                
                # 调用大模型提取
                json_str = extract_info_from_text(chunk , model_name)
                
                try:
                    data = json.loads(json_str)
                    if isinstance(data, list):
                        file_data.extend(data)
                    else:
                        print(f"   第 {j+1} 块返回的不是列表格式，已跳过")
                except Exception as e:
                    print(f"   第 {j+1} 块 JSON 解析失败: {str(e)}")
                
                #  防限流：每调用一次 API，强制休息 1.5 秒
                time.sleep(1.5)

            # ==========================================
            # 4. 数据去重（基于改性剂名称）
            # ==========================================
            seen_modifiers = set()
            for item in file_data:
                mod_name = item.get("modifier_name", "").strip()
                
                # 过滤掉空名称和"未提及"
                if mod_name and mod_name != "未提及" and mod_name not in seen_modifiers:
                    seen_modifiers.add(mod_name)
                    all_rows.append([
                        mod_name,
                        item.get("dosage", "未提及"),
                        item.get("base_formula", "未提及"),
                        item.get("performance_improvement", "未提及"),
                        item.get("test_standard", "未提及"),
                        item.get("experimental_conditions", "未提及")
                    ])
            
            # 如果提取完发现一条有效数据都没有，给个提示
            if not seen_modifiers:
                all_rows.append([f" {file_name}: 未能提取到有效改性剂数据", "", "", "", "", ""])

        except Exception as e:
            print(f" {file_name} 核心报错: {str(e)}")
            all_rows.append([f" {file_name} 解析失败", f"原因: {str(e)}", "", "", "", ""])
    
    return all_rows
