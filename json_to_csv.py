import os
import json
import pandas as pd

folder = "./du_lieu_json"
all_data = []

for filename in os.listdir(folder):
    if filename.endswith(".json"):
        with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Dữ liệu json bạn là dạng list []
            for article in data:
                all_data.append({
                    'text': article.get('title', '') + ". " + article.get('content', ''),
                    'label': 0  # Dữ liệu thật
                })

df = pd.DataFrame(all_data)
df.to_csv('vnexpress_news.csv', index=False, encoding='utf-8-sig')
print(" Đã xuất file vnexpress_news.csv")
