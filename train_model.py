import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import json

# 1. 读取你的真实数据
df = pd.read_csv("data.csv", encoding="utf-8-sig")

#抓取列名
feature_names = df.columns[:-1].tolist() #除去最后一列，前面所有列的列名
target_name = df.columns[-1] #最后一列是目标列名

# 2. 自动识别：取所有列（除了最后一列）作为特征，取最后一列作为目标
X = df.iloc[:, :-1]  # 前面所有列全是特征
y = df.iloc[:, -1]   # 最后一列是目标值

# 3. 自动处理缺失值
X = X.fillna(X.mean(numeric_only=True))
y = y.fillna(y.mean())

# 4. 划分训练集和测试集：
X = X.values
y = y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 训练模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. 评估并保存
y_pred = model.predict(X_test)
print(f" 模型训练完成！特征数量：{X.shape[1]}，R2 = {r2_score(y_test, y_pred):.4f}")
joblib.dump(model, 'material_model.pkl')

#保存配置文件
config = {
    "features": feature_names,
    "target": target_name
}
with open('feature_config.json','w',encoding='utf-8')as f:
    json.dump(config, f, ensure_ascii=False, indent=4)
print("特征配置文件已保存！")