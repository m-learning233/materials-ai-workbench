import os
import json
import joblib
import numpy as np

# 获取当前脚本绝对目录，防止路径错乱
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 加载模型
try:
    ml_model = joblib.load(os.path.join(BASE_DIR, 'material_model.pkl'))
    print(" 机器学习模型加载成功！")
except Exception as e:
    ml_model = None
    print(f" 模型加载失败: {e}")

# 2. 加载特征配置
feature_names = []
target_name = "目标值"
config_path = os.path.join(BASE_DIR, 'feature_config.json')

try:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        # 只要 JSON 里有 features 就取，没有就给空列表，防止 KeyError
        feature_names = config.get("features", [])
        target_name = config.get("target", "目标值")
    print(f" 特征配置加载成功，包含特征：{feature_names}")
except Exception as e:
    print(f" 特征配置加载失败，具体原因: {e}")

# 3. 定义预测函数（这是 chat.py 需要导入的核心函数）
def predict_features(feature_str):
    if ml_model is None:
        return " 模型未加载，请检查终端。"
    if not feature_names:
        return " 特征配置未加载，请先运行 train_model.py。"
        
    try:
        # 解析输入
        raw_list = [float(x.strip()) for x in feature_str.split(',')]
        
        # 数量校验
        if len(raw_list) != len(feature_names):
            return f" 特征数量错误！期望 {len(feature_names)} 个，但您输入了 {len(raw_list)} 个。"
            
        # 预测
        input_data = np.array([raw_list])
        prediction = ml_model.predict(input_data)[0]
        return f" 预测 {target_name}：{prediction:.4f}"
    except Exception as e:
        return f" 输入格式有误，报错：{str(e)}"