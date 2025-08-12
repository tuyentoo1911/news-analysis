# -*- coding: utf-8 -*-
"""
AI Text Analysis Web App (Streamlit)
===================================
Yêu cầu: 3 cách nhập văn bản; 3 mô hình AI (cảm xúc, tin giả, chủ đề);
Giao diện hiện đại có sidebar và biểu đồ trực quan.
"""

from __future__ import annotations

import time
import streamlit as st
import pandas as pd

from models import analyze_text
from utils import (
    load_custom_css,
    inject_bootstrap,
    create_sentiment_chart,
    create_fake_news_gauge,
    create_topic_chart,
    simulate_url_crawl,
    get_emoji_for_sentiment,
    get_emoji_for_topic,
    get_robot_image_base64,
)


# ---------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------
st.set_page_config(
    page_title='AI Text Analysis Platform', page_icon='🤖', layout='wide', initial_sidebar_state='expanded'
)
load_custom_css()
inject_bootstrap()


# ---------------------------------------------------------------
# Session State
# ---------------------------------------------------------------
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'current_text' not in st.session_state:
    st.session_state.current_text = ''


# ---------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------
with st.sidebar:
    st.title('🤖 AI Analysis')
    page = st.radio('Điều hướng', ['🏠 Trang chủ', '🔮 Dự đoán', '📊 Dữ liệu', '👥 Giới thiệu'], index=0)
    st.markdown('---')
    if st.session_state.analysis_history:
        st.caption('Tổng số phân tích:')
        st.metric('Lượt', len(st.session_state.analysis_history))


# ---------------------------------------------------------------
# Pages
# ---------------------------------------------------------------
if page == '🏠 Trang chủ':
    # Navbar pills mock
    st.markdown(
        """
        <div class="container py-4">
          <div class="d-flex align-items-center justify-content-between mb-4">
            <div class="text-uppercase fw-bold text-secondary">THYNK<br/>UNLIMITED.</div>
            <ul class="nav nav-pills pill-container">
              <li class="nav-item"><a class="nav-link active" href="#">HOME</a></li>
              <li class="nav-item"><a class="nav-link" href="#">ABOUT</a></li>
              <li class="nav-item"><a class="nav-link" href="#">CONTENT</a></li>
              <li class="nav-item"><a class="nav-link" href="#">OTHERS</a></li>
            </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    img_b64 = get_robot_image_base64()
    img_html = f'<img src="{img_b64}" class="img-fluid" style="max-width:520px;border-radius:18px;box-shadow:0 30px 60px rgba(0,0,0,.35);"/>' if img_b64 else '<div class="display-3">🤖</div>'

    st.markdown(
        f"""
        <div class="container py-2">
          <div class="row align-items-center g-4" style="min-height:70vh;">
            <div class="col-lg-6">
              <h1 class="hero-title">The Business of<br/><span class="ai-grad">Artificial<br/>Intelligence</span></h1>
              <div class="hero-sub mt-3">WHERE INNOVATION MEETS OPPORTUNITY</div>
              <p class="hero-desc mt-3">Artificial Intelligence (AI) is no longer just a futuristic idea—it's a powerful tool shaping businesses today. From customer service to logistics, AI is driving efficiency, innovation, and new market opportunities across industries.</p>
              <a class="btn-next mt-3" href="#predict">Next Slide <span>›</span></a>
            </div>
            <div class="col-lg-6 text-center">{img_html}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == '🔮 Dự đoán':
    st.subheader('Phân tích văn bản với 3 mô hình AI')

    st.markdown('### 📝 Chọn cách nhập văn bản')
    method = st.radio('Cách nhập', ['Dán văn bản', 'Tải file .txt', 'Nhập URL'], horizontal=True)

    text_to_analyze = ''

    if method == 'Dán văn bản':
        text_to_analyze = st.text_area('Nội dung bài báo', value=st.session_state.current_text, height=200)

    elif method == 'Tải file .txt':
        up = st.file_uploader('Chọn file .txt', type=['txt'])
        if up is not None:
            try:
                text_to_analyze = up.read().decode('utf-8')
                st.success(f'Đã tải: {up.name}')
                with st.expander('Xem trước'):
                    st.text(text_to_analyze[:800] + ('...' if len(text_to_analyze) > 800 else ''))
            except Exception as e:
                st.error(f'Lỗi đọc file: {e}')

    else:  # Nhập URL
        url = st.text_input('Nhập URL bài báo', placeholder='https://example.com/article')
        if url and st.button('🌐 Giả lập crawl nội dung'):
            text_to_analyze = simulate_url_crawl(url)
            st.success('Đã lấy nội dung mẫu từ URL')
            with st.expander('Nội dung đã crawl'):
                st.text(text_to_analyze)

    st.markdown('---')
    col_a, col_b = st.columns([3, 1])
    with col_a:
        run = st.button('🚀 Phân tích', type='primary', use_container_width=True)
    with col_b:
        clear = st.button('🧹 Xóa', use_container_width=True)

    if clear:
        st.session_state.current_text = ''
        st.session_state.analysis_results = None
        st.rerun()

    if run:
        # Hỗ trợ tự động crawl khi người dùng chọn Nhập URL
        if method == 'Nhập URL':
            if not url:
                st.warning('Vui lòng nhập URL.')
            else:
                # Nếu chưa có nội dung thì tự crawl nội dung mẫu từ URL
                if not text_to_analyze or len(text_to_analyze.strip()) < 10:
                    text_to_analyze = simulate_url_crawl(url)
                    with st.expander('Nội dung đã crawl'):
                        st.text(text_to_analyze)
                if not text_to_analyze or len(text_to_analyze.strip()) < 10:
                    st.warning('Không lấy được nội dung từ URL. Vui lòng kiểm tra lại.')
                else:
                    with st.spinner('🤖 Đang phân tích...'):
                        prog = st.progress(0)
                        for i in [30, 65, 100]:
                            time.sleep(0.15)
                            prog.progress(i)
                        prog.empty()
                        results = analyze_text(text_to_analyze)
                        st.session_state.analysis_results = results
                        st.session_state.analysis_history.append(results)
                        st.success('Hoàn tất!')
        else:
            if not text_to_analyze or len(text_to_analyze.strip()) < 10:
                st.warning('Vui lòng nhập tối thiểu 10 ký tự.')
            else:
                with st.spinner('🤖 Đang phân tích...'):
                    # Hiệu ứng tiến trình nhỏ
                    prog = st.progress(0)
                    for i in [30, 65, 100]:
                        time.sleep(0.15)
                        prog.progress(i)
                    prog.empty()
                    results = analyze_text(text_to_analyze)
                    st.session_state.analysis_results = results
                    st.session_state.analysis_history.append(results)
                    st.success('Hoàn tất!')

    # Hiển thị kết quả
    if st.session_state.analysis_results:
        res = st.session_state.analysis_results

        c1, c2, c3 = st.columns(3)
        sent = res['sentiment']
        fake = res['fake_news']
        topic = res['topic']

        with c1:
            cls = 'result-positive' if sent['sentiment'] == 'Tích cực' else (
                'result-negative' if sent['sentiment'] == 'Tiêu cực' else 'result-neutral')
            st.markdown(f"<div class='result-card {cls}'><h4>🎭 Cảm xúc</h4><h3>{sent['sentiment']}</h3><p>Độ tin cậy: {sent['confidence']:.0%}</p></div>", unsafe_allow_html=True)
            if sent['keywords']:
                st.caption('Từ khóa: ' + ', '.join(sent['keywords']))

        with c2:
            cls = 'result-fake' if fake.get('is_fake') else ('result-real' if fake.get('is_fake') is False else 'result-neutral')
            st.markdown(f"<div class='result-card {cls}'><h4>🔍 Tin giả</h4><h3>{fake['prediction']}</h3><p>Độ tin cậy: {fake['confidence']:.0%}</p></div>", unsafe_allow_html=True)
            if fake['indicators']['fake_signals']:
                st.caption('Dấu hiệu: ' + ', '.join(fake['indicators']['fake_signals']))

        with c3:
            st.markdown("<div class='result-card result-neutral'><h4>📂 Chủ đề</h4>" f"<h3>{topic['topic']}</h3><p>Độ tin cậy: {topic['confidence']:.0%}</p></div>", unsafe_allow_html=True)
            if topic['keywords']:
                st.caption('Từ khóa: ' + ', '.join(topic['keywords']))

        st.markdown('---')
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(create_sentiment_chart(sent), use_container_width=True)
            st.plotly_chart(create_topic_chart(topic), use_container_width=True)
        with g2:
            st.plotly_chart(create_fake_news_gauge(fake), use_container_width=True)
            st.metric('Số từ', res['word_count'])
            st.metric('Độ dài (kí tự)', res['text_length'])

elif page == '📊 Dữ liệu':
    st.subheader('Lịch sử phân tích')
    hist = st.session_state.analysis_history
    if not hist:
        st.info('Chưa có dữ liệu. Hãy phân tích ở tab "🔮 Dự đoán".')
    else:
        rows = []
        for i, r in enumerate(hist, start=1):
            rows.append({
                'STT': i,
                'Cảm xúc': r['sentiment']['sentiment'],
                'Tin giả': r['fake_news']['prediction'],
                'Chủ đề': r['topic']['topic'],
                'Số từ': r['word_count'],
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.download_button('💾 Tải CSV', df.to_csv(index=False), file_name='analysis_history.csv', mime='text/csv')

else:  # 👥 Giới thiệu
    st.subheader('Giới thiệu')
    st.write('- Công nghệ: Streamlit, Plotly, Python')
    st.write('- Tác vụ: Phân tích cảm xúc, phát hiện tin giả, phân loại chủ đề')
    st.write('- Thiết kế: CSS tuỳ chỉnh, biểu đồ trực quan, responsive')


