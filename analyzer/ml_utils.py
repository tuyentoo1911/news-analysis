import re
import string
import numpy as np
from pathlib import Path
import os
import math
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def preprocess_text(text):
    """
    Tiền xử lý văn bản
    """
    if not text or not isinstance(text, str):
        return ""

    # Chuyển về chữ thường
    text = text.lower()

    # Loại bỏ dấu câu
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Loại bỏ số
    text = re.sub(r'\d+', '', text)

    # Loại bỏ khoảng trắng thừa
    text = ' '.join(text.split())

    return text


def predict_fake_news(text):
    """
    THUẬT TOÁN PHÂN TÍCH TIN GIẢ ĐƯỢC CẢI TIẾN
    - Giảm bias, confidence thực tế hơn
    - Thêm nhiều yếu tố phân tích
    - Logic cân bằng hơn
    """
    try:
        # Kiểm tra đầu vào
        if not text or not isinstance(text, str):
            return {
                'is_fake': False,
                'confidence': 0.5,
                'message': 'Không thể phân tích văn bản trống',
                'processed_text': '',
                'analysis_details': {}
            }

        processed_text = preprocess_text(text)

        if not processed_text or len(processed_text.strip()) < 10:
            return {
                'is_fake': False,
                'confidence': 0.6,
                'message': 'Văn bản quá ngắn để phân tích chính xác',
                'processed_text': processed_text,
                'analysis_details': {'reason': 'text_too_short'}
            }

        # =================================
        # CÁC CHỈ SỐ PHÂN TÍCH ĐƯỢC CẢI TIẾN
        # =================================

        analysis_details = {}

        # 1. PHÂN TÍCH TỪ KHÓA ĐÁNG NGỜ (Giảm trọng số)
        suspicious_words = [
            # Clickbait words
            'giật gân', 'sốc', 'không thể tin được', 'bí mật', 'khủng khiếp',
            'nóng hổi', 'bom tấn', 'độc quyền', 'lạ lùng', 'kinh hoàng',
            'bất ngờ', 'chấn động', 'rúng động', 'gây sốt', 'viral',
            # Thêm từ tiếng Việt
            'kinh dị', 'rợn người', 'choáng váng', 'thót tim', 'đỉnh cao',
            'tuyệt đối', 'nhất định', '100%', 'chắc chắn', 'không ai biết'
        ]

        suspicious_found = [word for word in suspicious_words if word in processed_text]
        suspicious_score = len(suspicious_found) * 0.08  # GIẢM từ 0.15 xuống 0.08
        analysis_details['suspicious_words'] = suspicious_found

        # 2. PHÂN TÍCH CHỈ BÁO TIN CẬY (Tăng trọng số)
        reliable_indicators = [
            # Nguồn tin chính thức
            'theo báo', 'nguồn tin', 'chính thức', 'thông cáo', 'báo cáo',
            'nghiên cứu', 'chuyên gia', 'phát ngôn viên', 'cơ quan chức năng',
            # Thêm nguồn cụ thể
            'vnexpress', 'tuổi trẻ', 'thanh niên', 'vietnamnet', 'dantri',
            'bộ y tế', 'bộ giáo dục', 'thủ tướng', 'chính phủ',
            'đại học', 'tiến sĩ', 'giáo sư', 'bác sĩ', 'luật sư'
        ]

        reliable_found = [word for word in reliable_indicators if word in processed_text]
        reliable_score = len(reliable_found) * 0.12  # TĂNG từ 0.1 lên 0.12
        analysis_details['reliable_indicators'] = reliable_found

        # 3. PHÂN TÍCH CẤU TRÚC VĂN BẢN
        structure_score = 0

        # Độ dài văn bản
        text_length = len(processed_text)
        if text_length < 50:
            structure_score += 0.15  # Quá ngắn
        elif text_length < 100:
            structure_score += 0.08  # Hơi ngắn
        elif text_length > 2000:
            structure_score -= 0.05  # Dài = tốt

        # Tỷ lệ dấu chấm than
        exclamation_count = text.count('!')
        exclamation_ratio = exclamation_count / len(text) if text else 0
        if exclamation_ratio > 0.02:  # >2% là quá nhiều
            structure_score += min(exclamation_ratio * 5, 0.2)

        # Tỷ lệ chữ in hoa
        upper_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if upper_ratio > 0.05:  # >5% là quá nhiều
            structure_score += min(upper_ratio * 2, 0.15)

        analysis_details['structure'] = {
            'length': text_length,
            'exclamation_ratio': round(exclamation_ratio, 4),
            'upper_ratio': round(upper_ratio, 4)
        }

        # 4. PHÂN TÍCH NGÔN NGỮ TÌNH CẢM
        emotional_words = [
            'yêu', 'ghét', 'tức giận', 'phẫn nộ', 'căm thù', 'khiếp sợ',
            'lo lắng', 'hoảng sợ', 'phấn khích', 'hào hứng'
        ]
        emotional_score = len([w for w in emotional_words if w in processed_text]) * 0.05

        # 5. KIỂM TRA SỐ LIỆU VÀ THÔNG TIN CỤ THỂ
        has_numbers = bool(re.search(r'\d', text))
        has_dates = bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text))
        has_specific_info = any(indicator in text.lower() for indicator in [
            'nghiên cứu cho thấy', 'theo thống kê', 'dữ liệu', 'phần trăm', '%'
        ])

        specificity_score = 0
        if has_numbers:
            specificity_score -= 0.05
        if has_dates:
            specificity_score -= 0.08
        if has_specific_info:
            specificity_score -= 0.1

        analysis_details['specificity'] = {
            'has_numbers': has_numbers,
            'has_dates': has_dates,
            'has_specific_info': has_specific_info
        }

        # =================================
        # TÍNH ĐIỂM TỔNG HỢP (CẢI TIẾN)
        # =================================

        # Base score bắt đầu từ 0.4 thay vì 0
        base_score = 0.4

        # Tổng hợp các điểm
        total_fake_score = (
            base_score +
            suspicious_score +
            structure_score +
            emotional_score +
            specificity_score -
            reliable_score
        )

        # Thêm yếu tố ngẫu nhiên NHỎ HƠN
        import random
        random_factor = random.uniform(-0.05, 0.05)  # GIẢM từ ±0.1 xuống ±0.05
        total_fake_score += random_factor

        # Chuẩn hóa điểm số với hàm sigmoid để tránh cực trị
        normalized_score = 1 / (1 + math.exp(-(total_fake_score - 0.5) * 6))

        # =================================
        # QUYẾT ĐỊNH KẾT QUẢ
        # =================================

        is_fake = normalized_score > 0.55  # Tăng ngưỡng lên 0.55

        # TÍNH CONFIDENCE THỰC TẾ HƠN
        if is_fake:
            # Confidence cho tin giả
            raw_confidence = normalized_score
            if raw_confidence > 0.9:
                confidence = 0.75 + (raw_confidence - 0.9) * 1.5  # Max ~0.9
            elif raw_confidence > 0.7:
                confidence = 0.65 + (raw_confidence - 0.7) * 0.5  # 0.65-0.75
            else:
                confidence = 0.55 + (raw_confidence - 0.55) * 0.67  # 0.55-0.65
        else:
            # Confidence cho tin thật
            raw_confidence = 1 - normalized_score
            if raw_confidence > 0.8:
                confidence = 0.7 + (raw_confidence - 0.8) * 1.0   # Max ~0.9
            elif raw_confidence > 0.6:
                confidence = 0.6 + (raw_confidence - 0.6) * 0.5   # 0.6-0.7
            else:
                confidence = 0.5 + raw_confidence * 0.2            # 0.5-0.6

        # Cap confidence tối đa 0.89
        confidence = min(confidence, 0.89)

        # Đặc biệt: Nếu không có đủ thông tin, giảm confidence
        if len(processed_text) < 100 and not reliable_found and not suspicious_found:
            confidence = max(confidence * 0.7, 0.5)  # Giảm xuống, tối thiểu 0.5

        analysis_details['scoring'] = {
            'suspicious_score': round(suspicious_score, 3),
            'reliable_score': round(reliable_score, 3),
            'structure_score': round(structure_score, 3),
            'emotional_score': round(emotional_score, 3),
            'specificity_score': round(specificity_score, 3),
            'total_fake_score': round(total_fake_score, 3),
            'normalized_score': round(normalized_score, 3),
            'random_factor': round(random_factor, 3)
        }

        # Thông điệp dựa trên kết quả
        if confidence < 0.6:
            message = "Độ tin cậy phân tích thấp - cần thêm thông tin"
        elif is_fake and confidence > 0.8:
            message = "Có dấu hiệu mạnh của tin giả"
        elif is_fake:
            message = "Có một số dấu hiệu đáng ngờ"
        elif confidence > 0.8:
            message = "Có vẻ là tin đáng tin cậy"
        else:
            message = "Phân tích hoàn tất"

        return {
            'is_fake': is_fake,
            'confidence': round(confidence, 3),
            'message': message,
            'processed_text': processed_text,
            'analysis_details': analysis_details,
            # Backward compatibility
            'suspicious_words_found': suspicious_found,
            'reliable_indicators_found': reliable_found,
            # Thêm thông tin debug (có thể bỏ trong production)
            'debug_info': {
                'text_length': len(text),
                'processed_length': len(processed_text),
                'decision_threshold': 0.55,
                'algorithm_version': '2.1_improved'
            }
        }

    except Exception as e:
        return {
            'is_fake': False,
            'confidence': 0.5,
            'message': f'Lỗi khi phân tích: {str(e)}',
            'processed_text': '',
            'analysis_details': {'error': str(e)}
        }


def summarize_text(text):
    """
    Tóm tắt văn bản tối ưu - gọn gàng và hiệu quả
    """
    try:
        if not text or len(text.strip()) < 80:
            return {
                'summary': text.strip(),
                'compression_ratio': 1.0,
                'message': 'Văn bản quá ngắn'
            }

        # Tách câu nhanh
        sentences = [s.strip() for s in re.split(r'[.!?]+', text.strip()) if s.strip()]

        if len(sentences) <= 2:
            return {
                'summary': text.strip(),
                'compression_ratio': 1.0,
                'message': 'Đã gọn'
            }

        # Từ khóa quan trọng tối ưu
        key_words = ['quan trọng', 'chính', 'kết quả', 'quyết định', 'thông báo', 'cho biết']

        # Tính điểm nhanh cho từng câu
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            words = sentence.lower().split()
            score = 0

            # Điểm vị trí (câu đầu và cuối quan trọng)
            if i == 0:
                score += 2
            elif i == len(sentences) - 1:
                score += 1
            elif i < len(sentences) * 0.3:
                score += 0.5

            # Điểm độ dài (câu 8-20 từ tối ưu)
            if 8 <= len(words) <= 20:
                score += 1
            elif len(words) < 4:
                score -= 1

            # Điểm từ khóa
            score += sum(0.3 for word in key_words if word in sentence.lower())

            scored_sentences.append((sentence, score))

        # Chọn top 30-50% câu tốt nhất
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        target_count = max(1, min(3, len(sentences) // 3))

        selected = scored_sentences[:target_count]
        selected.sort(key=lambda x: sentences.index(x[0]))  # Giữ thứ tự gốc

        # Tạo tóm tắt
        summary = '. '.join([s[0] for s in selected]) + '.'
        compression_ratio = len(summary) / len(text)

        return {
            'summary': summary,
            'compression_ratio': round(compression_ratio, 3),
            'message': f'Rút gọn {len(sentences)}→{len(selected)} câu'
        }

    except Exception as e:
        # Fallback đơn giản
        words = text.split()
        if len(words) > 50:
            summary = ' '.join(words[:30]) + '...'
            return {
                'summary': summary,
                'compression_ratio': 0.6,
                'message': 'Tóm tắt cơ bản'
            }
        return {
            'summary': text,
            'compression_ratio': 1.0,
            'message': 'Lỗi nhỏ, giữ nguyên'
        }


def analyze_topic(text):
    """
    Phân tích chủ đề của văn bản sử dụng LDA model
    """
    try:
        import joblib
        import numpy as np
        import os
        import json

        # Đường dẫn đến model
        model_dir = os.path.join('ml_models', 'topic_model')
        lda_model_path = os.path.join(model_dir, 'lda_model.joblib')
        vectorizer_path = os.path.join(model_dir, 'vectorizer_bow.joblib')
        topics_path = os.path.join(model_dir, 'topics.json')

        # Kiểm tra file tồn tại
        if not all(os.path.exists(path) for path in [lda_model_path, vectorizer_path, topics_path]):
            return fallback_topic_analysis(text)

        # Load model và vectorizer
        lda_model = joblib.load(lda_model_path)
        vectorizer = joblib.load(vectorizer_path)

        # Load topic mapping
        with open(topics_path, 'r', encoding='utf-8') as f:
            topic_keywords = json.load(f)

        # Preprocess text
        processed_text = preprocess_text(text)

        # Transform text to vector
        text_vector = vectorizer.transform([processed_text])

        # Predict topic probabilities
        topic_probs = lda_model.transform(text_vector)[0]

        # Lấy topic có xác suất cao nhất
        dominant_topic = np.argmax(topic_probs)
        confidence = topic_probs[dominant_topic]

        # Map topic số thành tên chủ đề dựa trên keywords
        topic_names = {
            '0': 'Sức khỏe',      # bệnh, thể, cơ, bác sĩ
            '1': 'Chính trị',     # mỹ, ông, trump, thống
            '2': 'Giáo dục',      # án, đề, học
            '3': 'Công nghệ',     # công, ai, dụng, năng
            '4': 'Thể thao',      # trận, đội, bóng, thủ
            '5': 'Đời sống',      # tôi, người, con, nhà
            '6': 'Kinh tế',       # giá, nước, đồng, bản
            '7': 'Giáo dục',      # học, sinh, thi, trường
            '8': 'Giao thông',    # xe, người, đường, an
            '9': 'Kinh tế',       # công, doanh, đầu, kinh
            '10': 'Du lịch',      # khách, du, ảnh, diễn
            '11': 'Thời sự'       # tỉnh, thành, công, bay
        }

        topic_name = topic_names.get(str(dominant_topic), 'Không xác định')

        return {
            'topic': topic_name,
            'confidence': float(confidence),
            'topic_id': dominant_topic,
            'message': f'Phân loại thành công với độ tin cậy {confidence*100:.1f}%'
        }

    except Exception as e:
        print(f"Error in LDA topic analysis: {e}")
        return fallback_topic_analysis(text)


def fallback_topic_analysis(text):
    """
    Phân tích chủ đề dự phòng bằng keyword matching
    """
    try:
        processed_text = preprocess_text(text)

        # Từ khóa cho các chủ đề
        topic_keywords = {
            'Chính trị': ['chính phủ', 'bộ trưởng', 'quốc hội', 'thủ tướng', 'chủ tịch', 'đảng', 'chính trị', 'mỹ', 'trump'],
            'Kinh tế': ['kinh tế', 'thị trường', 'đầu tư', 'chứng khoán', 'ngân hàng', 'tài chính', 'doanh nghiệp', 'giá', 'đồng'],
            'Thể thao': ['bóng đá', 'thể thao', 'vận động viên', 'world cup', 'sea games', 'olympic', 'trận', 'đội', 'thủ'],
            'Giải trí': ['nghệ sĩ', 'ca sĩ', 'diễn viên', 'phim', 'âm nhạc', 'showbiz'],
            'Công nghệ': ['công nghệ', 'smartphone', 'internet', 'ai', 'robot', 'ứng dụng'],
            'Sức khỏe': ['sức khỏe', 'bệnh viện', 'bác sĩ', 'thuốc', 'y tế', 'covid', 'bệnh'],
            'Giáo dục': ['học', 'sinh', 'thi', 'trường', 'điểm', 'đại học', 'giáo dục'],
            'Giao thông': ['xe', 'đường', 'tai nạn', 'giao thông', 'máy'],
            'Du lịch': ['du lịch', 'khách', 'điểm đến', 'tour'],
            'Đời sống': ['gia đình', 'con', 'nhà', 'cuộc sống']
        }

        topic_scores = {}

        for topic, keywords in topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in processed_text)
            if score > 0:
                topic_scores[topic] = score

        if topic_scores:
            best_topic = max(topic_scores, key=topic_scores.get)
            max_score = topic_scores[best_topic]
            confidence = min(max_score / 10.0, 1.0)  # Normalize confidence

            return {
                'topic': best_topic,
                'confidence': confidence,
                'topic_id': -1,
                'message': f'Phân loại bằng keywords với {max_score} từ khóa khớp'
            }
        else:
            return {
                'topic': 'Không xác định',
                'confidence': 0.0,
                'topic_id': -1,
                'message': 'Không tìm thấy từ khóa phù hợp'
            }

    except Exception as e:
        return {
            'topic': 'Không xác định',
            'confidence': 0.0,
            'topic_id': -1,
            'message': f'Lỗi phân tích: {str(e)}'
        }


def analyze_sentiment(text):
    """
    Phân tích cảm xúc của văn bản sử dụng RoBERTa model
    """
    try:
        # Đường dẫn đến model
        model_dir = os.path.join('ml_models', 'news_sentiment_pol3')

        # Kiểm tra model tồn tại
        if not os.path.exists(model_dir):
            return fallback_sentiment_analysis(text)

        # Load tokenizer và model
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        # Tiền xử lý văn bản
        processed_text = text.strip()
        if len(processed_text) > 512:
            # Cắt văn bản nếu quá dài (RoBERTa có giới hạn 512 tokens)
            processed_text = processed_text[:512]

        # Tokenize
        inputs = tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()

        # Mapping labels
        label_mapping = {
            0: 'tiêu cực',    # neg
            1: 'trung tính',   # neu
            2: 'tích cực'      # pos
        }

        sentiment_label = label_mapping.get(predicted_class, 'không xác định')

        # Emoji mapping
        emoji_mapping = {
            'tiêu cực': '😢',
            'trung tính': '😐',
            'tích cực': '😊'
        }

        return {
            'sentiment': sentiment_label,
            'confidence': float(confidence),
            'sentiment_id': predicted_class,
            'emoji': emoji_mapping.get(sentiment_label, '❓'),
            'message': f'Phân tích cảm xúc hoàn tất với độ tin cậy {confidence*100:.1f}%'
        }

    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return fallback_sentiment_analysis(text)


def fallback_sentiment_analysis(text):
    """
    Phân tích cảm xúc dự phòng bằng keyword matching
    """
    try:
        processed_text = text.lower()

        # Từ khóa cảm xúc
        positive_words = [
            'tốt', 'tuyệt vời', 'xuất sắc', 'hạnh phúc', 'vui', 'thành công',
            'tích cực', 'hoàn hảo', 'ấn tượng', 'hài lòng', 'thắng lợi',
            'khuyến khích', 'hy vọng', 'lạc quan', 'phát triển', 'tiến bộ'
        ]

        negative_words = [
            'xấu', 'tệ', 'khủng khiếp', 'buồn', 'thất bại', 'tiêu cực',
            'thất vọng', 'lo lắng', 'khó khăn', 'vấn đề', 'tai nạn',
            'bệnh', 'chết', 'mất', 'thiệt hại', 'nguy hiểm', 'khủng hoảng'
        ]

        positive_score = sum(1 for word in positive_words if word in processed_text)
        negative_score = sum(1 for word in negative_words if word in processed_text)

        if positive_score > negative_score:
            sentiment = 'tích cực'
            confidence = min(0.6 + (positive_score - negative_score) * 0.1, 0.85)
            sentiment_id = 2
            emoji = '😊'
        elif negative_score > positive_score:
            sentiment = 'tiêu cực'
            confidence = min(0.6 + (negative_score - positive_score) * 0.1, 0.85)
            sentiment_id = 0
            emoji = '😢'
        else:
            sentiment = 'trung tính'
            confidence = 0.5
            sentiment_id = 1
            emoji = '😐'

        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'sentiment_id': sentiment_id,
            'emoji': emoji,
            'message': f'Phân tích bằng keywords với {positive_score + negative_score} từ khóa khớp'
        }

    except Exception as e:
        return {
            'sentiment': 'không xác định',
            'confidence': 0.0,
            'sentiment_id': -1,
            'emoji': '❓',
            'message': f'Lỗi phân tích cảm xúc: {str(e)}'
        }

