import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # 모델 저장

# 📂 데이터 파일 경로
data_path = r"C:\Users\miju\Desktop\me\raw.data\sevteen_final_dataset"
train_file = os.path.join(data_path, "train_data.csv")
test_file = os.path.join(data_path, "test_data.csv")

# ✅ CSV 파일 로드
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# 📌 특징(Feature)과 라벨(Label) 분리
X_train = train_df.iloc[:, :-1].values  # 마지막 열(Label) 제외
y_train = train_df.iloc[:, -1].values   # 마지막 열(Label)만 선택

X_test = test_df.iloc[:, :-1].values  # 마지막 열(Label) 제외
y_test = test_df.iloc[:, -1].values   # 마지막 열(Label)만 선택

# ✅ Random Forest 모델 생성 및 학습
rf_model = RandomForestClassifier(n_estimators=300)
rf_model.fit(X_train, y_train)

# ✅ 예측 수행
y_pred = rf_model.predict(X_test)

# ✅ 성능 평가
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\n🎯 **Random Forest 모델 평가 결과**")
print(f"✅ 정확도(Accuracy): {accuracy:.4f}")
print(f"\n📊 **분류 보고서(Classification Report):**\n{report}")
print(f"\n📌 **혼동 행렬(Confusion Matrix):**\n{conf_matrix}")

# ✅ 모델 저장
model_path = os.path.join(data_path, "sevteen_random_forest_model.pkl")
joblib.dump(rf_model, model_path)
print(f"\n💾 모델이 저장되었습니다: {model_path}")
