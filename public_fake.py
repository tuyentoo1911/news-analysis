import pandas as pd
import random

# Bước 1: Đọc dữ liệu
df = pd.read_csv("public_train.csv")
print("Các cột có trong file:", df.columns)

# Bước 2: Lọc 100 dòng dữ liệu thật (label == 0)
df_real = df[df['label'] == 0].copy()
fake_sample = df_real.sample(100, random_state=42)

# Bước 3: Tạo nội dung giả từ post_message
def make_fake(text):
    if isinstance(text, str):
        return text + " (Thông tin này chưa được kiểm chứng và có thể là tin giả)."
    return text

fake_sample['label'] = 1
fake_sample['post_message'] = fake_sample['post_message'].apply(make_fake)

# Bước 4: Gộp dữ liệu gốc và dữ liệu giả
df_final = pd.concat([df, fake_sample], ignore_index=True)

# Bước 5: Lưu ra file mới
df_final.to_csv("public_fake_news.csv", index=False)

print("✅ Đã lưu file mới: public_fake_news.csv")
