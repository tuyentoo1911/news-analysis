# -*- coding: utf-8 -*-
"""
Utilities for the Streamlit App
===============================
Gồm: CSS giao diện, biểu đồ Plotly, mô phỏng crawl URL, và helpers.
"""

from __future__ import annotations

import base64
import os
import time
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


def load_custom_css() -> None:
    """Inject CSS để có giao diện hiện đại, nhất quán."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        .stApp { font-family: 'Inter', sans-serif !important; }
        .stApp { background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important; }
        #MainMenu, footer { display: none !important; }

        .analysis-card {
            background: #ffffffee; backdrop-filter: blur(8px);
            border-radius: 16px; padding: 1.25rem; border: 1px solid #e5e7eb;
            box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        }
        .result-card { color: #fff; border-radius: 14px; padding: 1rem; text-align: center; }
        .result-positive { background: linear-gradient(135deg, #10b981, #34d399); }
        .result-negative { background: linear-gradient(135deg, #ef4444, #f97316); }
        .result-neutral  { background: linear-gradient(135deg, #3b82f6, #8b5cf6); }
        .result-fake     { background: linear-gradient(135deg, #db2777, #f43f5e); }
        .result-real     { background: linear-gradient(135deg, #059669, #10b981); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_bootstrap(theme: str = "dark") -> None:
    """Nhúng Bootstrap 5 từ CDN và thiết lập style chủ đạo (dark/light)."""
    st.markdown(
        """
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-YvpcrYf0tY3lHB60NNkmXc5s9fDVZLESaAA55NDzOxhy9GkcIdslK1eN7N6jIeHz" crossorigin="anonymous"></script>
        <style>
        body, .stApp { background: #0b1220; }
        .navbar-custom { background: rgba(13, 20, 36, 0.7); backdrop-filter: blur(8px); }
        .pill-container { background:#0f1a33; border-radius: 40px; padding: 6px; box-shadow: 0 10px 30px rgba(0,0,0,.25); }
        .pill-container .nav-link { color: #c7d2fe; padding: 10px 22px; border-radius: 24px; font-weight: 600; }
        .pill-container .nav-link.active { background:#1e293b; color:#fff; }
        .hero-title { font-size: 3.2rem; font-weight: 800; color: #e5e7eb; line-height: 1.15; }
        .ai-grad { background: linear-gradient(45deg,#8ab4ff,#c7b8ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
        .hero-sub { color:#9aa4b2; text-transform: uppercase; letter-spacing: .2rem; font-weight:700; }
        .hero-desc { color:#cbd5e1; max-width: 560px; }
        .btn-next { display:inline-flex; align-items:center; gap:.6rem; background:#2b3a67; color:#e5e7eb; border-radius: 40px; padding:.8rem 1.2rem; border:1px solid #3b4a78; text-decoration:none; }
        .btn-next:hover { background:#364b8a; color:#fff; }
        .round-arrow { width:72px; height:72px; border-radius:50%; background:#0f1a33; display:flex; align-items:center; justify-content:center; box-shadow: 0 10px 30px rgba(0,0,0,.35); }
        .round-arrow span { color:#e5e7eb; font-size:1.6rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_robot_image_base64(prefer_file: str = "image5.png") -> str | None:
    """Đọc ảnh từ thư mục media và trả về chuỗi base64 để nhúng vào HTML img."""
    media_dir = "Blue and White Modern Artificial Intelligence Presentation_media"
    path = None
    if os.path.exists(os.path.join(media_dir, prefer_file)):
        path = os.path.join(media_dir, prefer_file)
    else:
        # chọn file png kích thước lớn nhất
        if os.path.exists(media_dir):
            pngs = [f for f in os.listdir(media_dir) if f.lower().endswith('.png')]
            if pngs:
                pngs.sort(key=lambda f: os.path.getsize(os.path.join(media_dir, f)), reverse=True)
                path = os.path.join(media_dir, pngs[0])
    if not path or not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/png;base64,{b64}"


def create_sentiment_chart(sentiment_result: dict) -> go.Figure:
    probs = sentiment_result.get('probabilities', {})
    colors = {'Tích cực': '#10b981', 'Tiêu cực': '#ef4444', 'Trung tính': '#3b82f6'}
    fig = go.Figure(
        data=[
            go.Bar(
                x=list(probs.keys()),
                y=list(probs.values()),
                marker_color=[colors.get(k, '#64748b') for k in probs.keys()],
                text=[f"{v:.0%}" for v in probs.values()], textposition='auto',
            )
        ]
    )
    fig.update_layout(
        title='Phân tích cảm xúc', height=350, showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_yaxis(tickformat='.0%')
    return fig


def create_fake_news_gauge(fake_result: dict) -> go.Figure:
    risk = float(fake_result.get('risk_score', 0.5))
    value = (1 - risk) * 100.0
    color = 'green' if risk < 0.35 else 'orange' if risk < 0.65 else 'red'
    fig = go.Figure(go.Indicator(
        mode='gauge+number', value=value, title={'text': 'Độ tin cậy (%)'},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}}
    ))
    fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)')
    return fig


def create_topic_chart(topic_result: dict) -> go.Figure:
    probs = topic_result.get('probabilities', {})
    top_items = dict(list(probs.items())[:6])
    fig = go.Figure(
        data=[go.Bar(
            x=list(top_items.values()), y=list(top_items.keys()), orientation='h',
            marker_color=px.colors.qualitative.Set3[: len(top_items)],
            text=[f"{v:.0%}" for v in top_items.values()], textposition='auto',
        )]
    )
    fig.update_layout(
        title='Xác suất chủ đề', height=350, showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_xaxis(tickformat='.0%')
    return fig


def simulate_url_crawl(url: str) -> str:
    """Giả lập crawl nội dung với progress bar thân thiện."""
    progress = st.progress(0)
    status = st.empty()
    for i in range(100):
        progress.progress(i + 1)
        if i < 30:
            status.info('Đang kết nối...')
        elif i < 70:
            status.info('Đang tải nội dung...')
        else:
            status.info('Đang xử lý dữ liệu...')
        time.sleep(0.01)
    progress.empty(); status.empty()

    lower = url.lower()
    if 'vnexpress' in lower:
        return (
            'Công nghệ AI đang phát triển mạnh mẽ tại Việt Nam. Doanh nghiệp đầu tư trí tuệ nhân tạo '
            'để cải thiện hiệu quả và trải nghiệm khách hàng. Thị trường dự kiến tăng trưởng nhanh.'
        )
    if any(k in lower for k in ['bbc', 'cnn']):
        return (
            'Global technology companies invest heavily in artificial intelligence. Recent breakthroughs '
            'enable improvements in healthcare, education and business automation.'
        )
    if any(k in lower for k in ['social', 'facebook']):
        return (
            'CHẤN ĐỘNG!!! Bí mật KHÔNG AI BIẾT!!! Bạn sẽ không tin được điều này!!! Chia sẻ ngay!!!'
        )
    return (
        'Đây là nội dung mẫu được tạo từ URL để demo các chức năng phân tích AI gồm cảm xúc, tin giả '
        'và chủ đề. Nội dung mang tính minh hoạ.'
    )


def get_emoji_for_sentiment(label: str) -> str:
    return {'Tích cực': '😊', 'Tiêu cực': '😞', 'Trung tính': '😐'}.get(label, '🤔')


def get_emoji_for_topic(topic: str) -> str:
    mapping = {
        'Công nghệ': '💻', 'Kinh tế': '💰', 'Sức khỏe': '🏥', 'Giáo dục': '📚',
        'Thể thao': '⚽', 'Giải trí': '🎭', 'Chính trị': '🏛️', 'Xã hội': '👥'
    }
    return mapping.get(topic, '📰')


