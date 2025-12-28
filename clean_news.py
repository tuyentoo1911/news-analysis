import pandas as pd

# Đọc file CSV gốc
df = pd.read_csv("clean_news.csv")

# Chuyển clean_text thành kiểu chuỗi và loại bỏ khoảng trắng 2 đầu
df["clean_text"] = df["clean_text"].astype(str).str.strip()

# Loại bỏ các dòng có clean_text là NaN, "nan", hoặc chuỗi rỗng ""
df_cleaned = df[
    (df["clean_text"].notna()) &
    (df["clean_text"].str.lower() != "nan") &
    (df["clean_text"] != "")
]

# Ghi lại dữ liệu sạch ra file mới
df_cleaned.to_csv("clean_news_fixed.csv", index=False)

print(f"Đã lưu dữ liệu sạch vào 'clean_news_fixed.csv' với {len(df_cleaned)} dòng.")
