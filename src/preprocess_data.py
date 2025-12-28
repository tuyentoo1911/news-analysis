import os
import json
import pandas as pd
import re
import string
from tqdm import tqdm
from underthesea import word_tokenize

# Lấy thư mục gốc dự án (capstone)
BASE_DIR = r"D:\Capstone_TinTuc"

# Load stopwords (chỉ load 1 lần)
STOPWORDS_PATH = os.path.join(BASE_DIR, "vietnamese_stopwords.txt")

with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
    stopwords = set(f.read().splitlines())

# --- Hàm làm sạch chung ---
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-ZÀ-Ỵà-ỵ0-9\s.,!?]", "", text)
    text = text.strip()
    return word_tokenize(text, format="text")

# --- Hàm dùng để xử lý toàn bộ folder JSON để train và lưu file CSV ---
def process_json_folder_for_training(folder_path, output_csv=None):
    
    if output_csv is None:
        output_csv = os.path.join(BASE_DIR, "data", "labeled_data.csv")


    documents = []
    filenames = []
    labels = []

    for file in tqdm(os.listdir(folder_path)):
        if file.endswith(".json"):
            label = file.replace(".json", "").replace("vnexpress_", "")
            filepath = os.path.join(folder_path, file)

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

                for article in data:
                    title = article.get("title", "").strip()
                    content = article.get("content", "").strip()
                    full_text = f"{title}. {content}"

                    if full_text.strip():
                        documents.append(clean_text(full_text))
                        filenames.append(file)
                        labels.append(label)

    df = pd.DataFrame({
        "filename": filenames,
        "text": documents,
        "label": labels
    })

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")  #  sử dụng biến output_csv
    print(f" Đã xử lý xong. Tổng số bài báo: {len(df)}. File lưu: {output_csv}")
    return df




