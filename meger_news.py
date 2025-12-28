import pandas as pd


df_real = pd.read_csv('vnexpress_news.csv')
df_fake = pd.read_csv('public_fake_news.csv')

# Gộp 2 file 
df = pd.concat([df_real, df_fake], ignore_index=True)


df = df.sample(frac=1).reset_index(drop=True)


df.to_csv('all_news.csv', index=False, encoding='utf-8-sig')
print(" Đã gộp xong file all_news.csv")