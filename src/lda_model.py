import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import joblib  # dùng để lưu mô hình

BASE_DIR = r"D:\Capstone_TinTuc"

def train_and_predict_lda(csv_path):  # output_csv="labeled_data.csv
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)
     # Bước 3: Tạo pipeline: Vector hóa + Naive Bayes
    clf = make_pipeline(CountVectorizer(), MultinomialNB())
     # Bước 4: Train mô hình
    clf.fit(X_train, y_train)
    # Bước 5: Dự đoán
    y_pred = clf.predict(X_test)
    # Bước 6: Báo cáo kết quả
    report = classification_report(y_test, y_pred, output_dict=True)
     # Bước 7: Lưu mô hình vào file
    save_path = os.path.join(BASE_DIR, "saved_models", "lda_model.joblib")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(clf, save_path)
    print(" Mô hình đã được train và lưu tại: saved_models/lda_model.joblib")
    return report

csv_path = os.path.join(BASE_DIR, "data", "labeled_data.csv")
report = train_and_predict_lda(csv_path)

def predict_with_lda(model_path, new_texts):
    clf = joblib.load(model_path)
    return clf.predict(new_texts)