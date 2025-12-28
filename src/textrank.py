# textrank.py
import spacy
import networkx as nx
from typing import List
import re
from numpy import dot
from numpy.linalg import norm

nlp = spacy.load("vi_core_news_lg")

def clean_text_textrank(text: str) -> str:
    # Bỏ ký tự đặc biệt và xuống dòng
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,]', '', text)  # Giữ dấu chấm, phẩy để chia câu
    return text.lower()

def cosine_sim(v1, v2):
    return dot(v1, v2) / (norm(v1) * norm(v2) + 1e-10)  # +1e-10 tránh chia 0

def textrank_summarize(text: str, num_sentences: int = 3) -> str:
    cleaned_text = clean_text_textrank(text)
    doc = nlp(cleaned_text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if len(sentences) <= num_sentences:
        return text

    # Chuyển câu thành vector
    sentence_vectors = []
    for sent in sentences:
        if nlp(sent).has_vector:
            sentence_vectors.append(nlp(sent).vector)
        else:
            sentence_vectors.append(nlp("văn bản").vector)  # fallback nếu không có vector

    similarity_matrix = nx.Graph()
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = cosine_sim(sentence_vectors[i], sentence_vectors[j])
            similarity_matrix.add_edge(i, j, weight=sim)

    scores = nx.pagerank(similarity_matrix)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    selected = [s for _, s in ranked_sentences[:num_sentences]]
    # Sắp xếp theo thứ tự gốc trong văn bản
    ordered_summary = [s for s in sentences if s in selected]
    return " ".join(ordered_summary)
