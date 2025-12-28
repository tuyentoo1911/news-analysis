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

    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = re.sub(r'\d+', '', text)

    text = ' '.join(text.split())

    return text


def predict_fake_news(text):
    """
    Phân tích tin giả sử dụng RoBERTa model được fine-tuned
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

        # Đường dẫn đến model
        model_dir = os.path.join('ml_models', 'fake_real_model')

        # Kiểm tra model tồn tại
        if not os.path.exists(model_dir):
            return fallback_fake_news_analysis(text)

        # Load tokenizer và model
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        # Tiền xử lý văn bản
        processed_text = text.strip()
        if len(processed_text) > 512:
            # Cắt văn bản nếu quá dài (RoBERTa có giới hạn 512 tokens)
            processed_text = processed_text[:512]

        if len(processed_text) < 10:
            return {
                'is_fake': False,
                'confidence': 0.6,
                'message': 'Văn bản quá ngắn để phân tích chính xác',
                'processed_text': processed_text,
                'analysis_details': {'reason': 'text_too_short'}
            }

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

            # Lấy probabilities của cả 2 class
            prob_class_0 = predictions[0][0].item()
            prob_class_1 = predictions[0][1].item()

        # Debug info
        print(f" AI Model: Class 0={prob_class_0:.3f}, Class 1={prob_class_1:.3f} → Predicted: {predicted_class}")

        # KIỂM TRA MODEL BIAS hoặc CONFIDENCE THẤP
        use_hybrid = False

        if confidence > 0.95 and predicted_class == 0:
            print(" Model bias cao, sử dụng hybrid approach")
            use_hybrid = True
        elif confidence < 0.65:
            print("  Confidence thấp, sử dụng hybrid approach")
            use_hybrid = True

        if use_hybrid:
            # Lấy kết quả từ thuật toán keyword
            fallback_result = fallback_fake_news_analysis(text)

            ai_fake_score = prob_class_1  # Xác suất model cho FAKE
            keyword_fake_score = fallback_result['confidence'] if fallback_result['is_fake'] else (1 - fallback_result['confidence'])

            combined_fake_score = (ai_fake_score * 0.35) + (keyword_fake_score * 0.65)

            # Giới hạn confidence tối đa 99%
            confidence = min(combined_fake_score if combined_fake_score > 0.5 else (1 - combined_fake_score), 0.99)

            # Format % không hiển thị 100
            ai_pct = min(ai_fake_score * 100, 99)
            keyword_pct = min(keyword_fake_score * 100, 99)

            if combined_fake_score > 0.5:
                is_fake = True
                message = f"Hybrid: Phát hiện tin giả (AI {ai_pct:.0f}% + Keyword {keyword_pct:.0f}%)"
            else:
                is_fake = False
                ai_real_pct = min((1-ai_fake_score) * 100, 99)
                keyword_real_pct = min((1-keyword_fake_score) * 100, 99)
                message = f"Hybrid: Xác nhận tin thật (AI {ai_real_pct:.0f}% + Keyword {keyword_real_pct:.0f}%)"

            print(f"   → Hybrid result: is_fake={is_fake}, confidence={confidence:.3f}")
        else:
            display_confidence = min(confidence * 100, 99)

            if predicted_class == 0:
                is_fake = False  # Class 0 = Real news
                message = f"AI xác nhận tin thật với độ tin cậy {display_confidence:.1f}%"
            else:
                is_fake = True   # Class 1 = Fake news
                message = f"AI phát hiện tin giả với độ tin cậy {display_confidence:.1f}%"

            # Giới hạn confidence lưu vào database
            confidence = min(confidence, 0.99)

        return {
            'is_fake': is_fake,
            'confidence': float(confidence),
            'message': message,
            'processed_text': processed_text,
            'analysis_details': {
                'model_prediction': predicted_class,
                'model_confidence': float(confidence),
                'model_version': 'RoBERTa_fine_tuned'
            },
            # Backward compatibility
            'suspicious_words_found': [],
            'reliable_indicators_found': [],
            # Debug info
            'debug_info': {
                'text_length': len(text),
                'processed_length': len(processed_text),
                'model_type': 'transformer',
                'algorithm_version': '5.4_roberta_final_0real_1fake'
            }
        }

    except Exception as e:
        print(f"Error in fake news analysis: {e}")
        return fallback_fake_news_analysis(text)


def fallback_fake_news_analysis(text):

    try:
        processed_text = preprocess_text(text)

        if not processed_text or len(processed_text.strip()) < 10:
            return {
                'is_fake': False,
                'confidence': 0.6,
                'message': 'Văn bản quá ngắn để phân tích chính xác',
                'processed_text': processed_text,
                'analysis_details': {'reason': 'text_too_short'}
            }

        suspicious_words = [
            # Clickbait tiếng Việt
            'giật gân', 'sốc', 'shock', 'không thể tin được', 'bí mật', 'khủng khiếp',
            'nóng hổi', 'bom tấn', 'độc quyền', 'lạ lùng', 'kinh hoàng',
            'bất ngờ', 'chấn động', 'rúng động', 'gây sốt', 'viral',
            'kinh dị', 'rợn người', 'choáng váng', 'thót tim',
            'phát hiện', 'cách kiếm tiền', 'tỷ đồng', 'giảm cân', 'thần tốc',
            'tuyệt đối', '100%', 'chắc chắn', 'không ai biết', 'bí quyết',
            'giật mình', 'choáng ngợp', 'không ngờ', 'thần kỳ',
            # Siêu nhiên / Giả khoa học
            'siêu năng lực', 'năng lực siêu nhiên', 'hiện tượng lạ', 'khoa học chưa giải thích',
            'chưa lý giải', 'bí ẩn', 'thần bí', 'kỳ lạ', 'phi thường',
            'phép màu', 'kỳ diệu', 'ma thuật', 'ảo thuật', 'thần thánh',
            'ngoài sức tưởng tượng', 'không ai tin', 'triệu lượt xem',
            'cư dân mạng xôn xao', 'lan truyền chóng mặt', 'gây tranh cãi'
        ]

        # Chỉ báo tin cậy - cải thiện
        reliable_indicators = [
            'nguon tin', 'chính thức', 'thông cáo', 'báo cáo',
            'nghiên cứu', 'chuyên gia', 'phát ngôn viên', 'cơ quan chức năng',
            'vnexpress', 'tuoi tre', 'thanh nien', 'vietnamnet', 'dantri',
            'bo y te', 'bo giao duc', 'thu tuong', 'chinh phu',
            # Thêm các chỉ báo tin cậy
            'cap nhat', 'xac nhan',
            'bac si', 'giao su', 'tien si', 'chuyen gia'
        ]

        suspicious_found = [word for word in suspicious_words if word in processed_text]
        reliable_found = [word for word in reliable_indicators if word in processed_text]

        # Tính điểm với trọng số cải thiện MẠNH
        suspicious_score = len(suspicious_found) * 0.25   # TĂNG MẠNH trọng số từ khóa đáng ngờ
        reliable_score = len(reliable_found) * 0.30      # Tăng trọng số tin cậy

        # Phân tích cấu trúc bổ sung
        structure_bonus = 0
        text_upper = text.upper()

        # Kiểm tra dấu chấm than nhiều (clickbait)
        if text.count('!') > 2:
            structure_bonus += 0.2

        # Kiểm tra chữ in hoa nhiều
        if len([c for c in text if c.isupper()]) / len(text) > 0.1:
            structure_bonus += 0.2

        if text.count('"') > 3 or text.count('"') > 3:
            structure_bonus += 0.15

        if 'triệu lượt' in text.lower() or 'tỷ lượt' in text.lower():
            structure_bonus += 0.25  # TĂNG MẠNH vì rất dấu hiệu tin giả

        # Tính điểm tổng
        base_score = 0.2  # Giảm base score
        total_score = base_score + suspicious_score + structure_bonus - reliable_score

        # Debug
        print(f"Keyword Score: suspicious={len(suspicious_found)} ({suspicious_score:.2f}), structure={structure_bonus:.2f}, reliable={len(reliable_found)} (-{reliable_score:.2f}) → total={total_score:.2f}")

        if total_score > 0.9:
            is_fake = True
            confidence = min(0.85 + (total_score - 0.9) * 0.1, 0.98)
            message = "Phat hien cuc ky nhieu dau hieu tin gia"
        elif total_score > 0.7:
            is_fake = True
            confidence = 0.75 + (total_score - 0.7) * 0.25
            message = "Phat hien nhieu dau hieu tin gia ro rang"
        elif total_score > 0.5:
            is_fake = True
            confidence = 0.65 + (total_score - 0.5) * 0.25
            message = "Phat hien nhieu dau hieu tin gia"
        elif total_score > 0.35:
            is_fake = True
            confidence = 0.60 + (total_score - 0.35) * 0.25
            message = "Co dau hieu tin gia"
        elif total_score < 0.1 and reliable_score > 0.2:
            is_fake = False
            confidence = min(0.75 + reliable_score * 0.15, 0.92)
            message = "Co nhieu dau hieu tin cay"
        elif total_score < 0.25:
            is_fake = False
            confidence = 0.65 + (0.25 - total_score) * 0.25
            message = "It dau hieu dang ngo"
        else:
            is_fake = True
            confidence = 0.58 + (total_score - 0.25) * 0.15
            message = "Khong ro rang - phan loai than trong la tin gia"

        return {
            'is_fake': is_fake,
            'confidence': confidence,
            'message': message,
            'processed_text': processed_text,
            'analysis_details': {
                'suspicious_words': suspicious_found,
                'reliable_indicators': reliable_found,
                'suspicious_score': suspicious_score,
                'reliable_score': reliable_score
            },
            'suspicious_words_found': suspicious_found,
            'reliable_indicators_found': reliable_found,
            'debug_info': {
                'text_length': len(text),
                'processed_length': len(processed_text),
                'model_type': 'fallback',
                'algorithm_version': '5.2_fallback_balanced'
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

        lda_model = joblib.load(lda_model_path)
        vectorizer = joblib.load(vectorizer_path)

        with open(topics_path, 'r', encoding='utf-8') as f:
            topic_keywords = json.load(f)

        processed_text = preprocess_text(text)

        text_vector = vectorizer.transform([processed_text])

        # Predict topic probabilities
        topic_probs = lda_model.transform(text_vector)[0]

        # Lấy topic có xác suất cao nhất
        dominant_topic = np.argmax(topic_probs)
        confidence = topic_probs[dominant_topic]

        topic_names = {
            '0': 'Sức khỏe',
            '1': 'Chính trị',
            '2': 'Giáo dục',
            '3': 'Công nghệ',
            '4': 'Thể thao',
            '5': 'Đời sống',
            '6': 'Kinh tế',
            '7': 'Giáo dục',
            '8': 'Giao thông',
            '9': 'Kinh tế',
            '10': 'Du lịch',
            '11': 'Thời sự'
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

