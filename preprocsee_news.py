import pandas as pd
import re
from pyvi import ViTokenizer  # dùng pyvi thay underthesea

# Đọc file tổng hợp đã có
df = pd.read_csv('all_news.csv')

# Hàm làm sạch văn bản
def clean_text(text):
    text = str(text)
    text = re.sub(r'\n|\r', ' ', text)          # Bỏ xuống dòng
    text = re.sub(r'https?://\S+', '', text)    # Bỏ link
    text = re.sub(r'[^0-9a-zA-ZÀ-ỹ\s]', '', text)  # Bỏ ký tự lạ
    return ViTokenizer.tokenize(text)           # Tách từ tiếng Việt

# Thêm cột mới clean_text đã làm sạch
df['clean_text'] = df['text'].apply(clean_text)

# Lưu file mới
df.to_csv('clean_news.csv', index=False, encoding='utf-8-sig')
print("Đã làm sạch và lưu file clean_news.csv")
