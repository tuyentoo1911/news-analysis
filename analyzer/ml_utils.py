import re
import string
import numpy as np
from pathlib import Path
import os
import math


def preprocess_text(text):
    """
    Tiền xử lý văn bản - GIỮ NGUYÊN
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

def predict_news(model, text, threshold_low=0.4, threshold_high=0.8):
    """
    Dự đoán bài báo thật hay giả, với ngưỡng cảnh báo.
    """
    prob = model.predict_proba([text])[0][1]  # Xác suất là tin thật
    percentage = round(prob * 100, 2)

    if prob >= threshold_high:
        return f"✅ Đây có thể là tin thật ({percentage}%)"
    elif prob <= threshold_low:
        return f"❌ Đây có thể là tin giả ({100 - percentage}%)"
    else:
        return f"⚠️ Hệ thống không chắc chắn ({percentage}%), cần kiểm tra thủ công."

# Ví dụ chạy
news_text = "Bài báo giả mạo mà bạn vừa tạo để test hệ thống"


def analyze_topic(text):
    """
    Phân tích chủ đề - CẢI TIẾN
    """
    try:
        processed_text = preprocess_text(text)
        
        # Mở rộng từ khóa chủ đề
        topic_keywords = {
            'Chính trị': [
                'chính phủ', 'bộ trưởng', 'quốc hội', 'thủ tướng', 'chủ tịch', 
                'đảng', 'chính trị', 'bầu cử', 'nghị quyết', 'luật', 'chính sách'
            ],
            'Kinh tế': [
                'kinh tế', 'thị trường', 'đầu tư', 'chứng khoán', 'ngân hàng', 
                'tài chính', 'doanh nghiệp', 'gdp', 'lạm phát', 'xuất khẩu', 'nhập khẩu'
            ],
            'Thể thao': [
                'bóng đá', 'thể thao', 'vận động viên', 'world cup', 'sea games', 
                'olympic', 'tennis', 'bơi lội', 'điền kinh', 'võ thuật'
            ],
            'Giải trí': [
                'nghệ sĩ', 'ca sĩ', 'diễn viên', 'phim', 'âm nhạc', 'showbiz',
                'concert', 'mv', 'phim truyền hình', 'gameshow'
            ],
            'Công nghệ': [
                'công nghệ', 'smartphone', 'internet', 'ai', 'robot', 'ứng dụng',
                'software', 'hardware', 'startup', 'digital', 'blockchain'
            ],
            'Sức khỏe': [
                'sức khỏe', 'bệnh viện', 'bác sĩ', 'thuốc', 'y tế', 'covid',
                'vaccine', 'điều trị', 'phòng ngừa', 'dinh dưỡng'
            ],
            'Giáo dục': [
                'giáo dục', 'trường học', 'học sinh', 'sinh viên', 'giáo viên',
                'đại học', 'thi cử', 'học phí', 'chương trình', 'đào tạo'
            ],
            'Xã hội': [
                'xã hội', 'gia đình', 'hôn nhân', 'tội phạm', 'an ninh',
                'giao thông', 'môi trường', 'biến đổi khí hậu', 'ô nhiễm'
            ]
        }
        
        topic_scores = {}
        
        for topic, keywords in topic_keywords.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in processed_text:
                    score += 1
                    matched_keywords.append(keyword)
            
            # Tính điểm có trọng số theo tần suất
            if score > 0:
                # Tính tần suất từ khóa
                frequency_score = sum(processed_text.count(kw) for kw in matched_keywords)
                topic_scores[topic] = {
                    'score': score,
                    'frequency': frequency_score,
                    'total': score + frequency_score * 0.5,
                    'keywords': matched_keywords
                }
        
        if topic_scores:
            # Chọn chủ đề có điểm cao nhất
            best_topic = max(topic_scores, key=lambda x: topic_scores[x]['total'])
            return {
                'topic': best_topic,
                'confidence': min(topic_scores[best_topic]['total'] / 10, 0.95),
                'details': topic_scores[best_topic]
            }
        else:
            return {
                'topic': 'Không xác định',
                'confidence': 0.1,
                'details': {}
            }
            
    except Exception as e:
        return {
            'topic': 'Lỗi phân tích',
            'confidence': 0.0,
            'details': {'error': str(e)}
        }


# THÊM HÀM TỐI ƯU HÓA SUMMARIZATION
def summarize_text(text):
    """
    Tóm tắt văn bản CẢI TIẾN - thông minh hơn
    """
    try:
        if not text or len(text.strip()) < 80:
            return {
                'summary': text.strip(),
                'compression_ratio': 1.0,
                'message': 'Văn bản quá ngắn để tóm tắt'
            }
        
        # Tách câu thông minh hơn
        sentences = []
        for s in re.split(r'[.!?]+', text.strip()):
            s = s.strip()
            if len(s) > 10:  # Chỉ giữ câu có ý nghĩa
                sentences.append(s)
        
        if len(sentences) <= 2:
            return {
                'summary': text.strip(),
                'compression_ratio': 1.0,
                'message': 'Đã đủ ngắn gọn'
            }
        
        # Từ khóa quan trọng mở rộng
        important_keywords = [
            # Từ chỉ thông tin quan trọng
            'quan trọng', 'chính', 'kết quả', 'quyết định', 'thông báo', 'cho biết',
            'theo', 'nghiên cứu', 'báo cáo', 'phát hiện', 'khám phá',
            # Từ chỉ nguyên nhân kết quả  
            'vì', 'do', 'bởi vì', 'dẫn đến', 'gây ra', 'tác động', 'ảnh hưởng',
            # Từ chỉ thời gian
            'hiện tại', 'tương lai', 'sắp tới', 'năm nay', 'tháng này'
        ]
        
        # Tính điểm cho từng câu (cải tiến)
        scored_sentences = []
        total_words = sum(len(s.split()) for s in sentences)
        
        for i, sentence in enumerate(sentences):
            words = sentence.lower().split()
            score = 0
            
            # 1. Điểm vị trí (câu đầu, cuối quan trọng)
            position_ratio = i / len(sentences)
            if i == 0:
                score += 3.0  # Câu đầu rất quan trọng
            elif i == len(sentences) - 1:
                score += 2.0  # Câu cuối quan trọng
            elif position_ratio < 0.3:
                score += 1.5  # Phần đầu quan trọng
            elif position_ratio > 0.7:
                score += 1.0  # Phần cuối cũng quan trọng
            
            # 2. Điểm độ dài (câu vừa phải tốt nhất)
            word_count = len(words)
            if 10 <= word_count <= 25:
                score += 2.0  # Độ dài lý tưởng
            elif 8 <= word_count <= 30:
                score += 1.5  # Độ dài tốt
            elif word_count < 5:
                score -= 1.0  # Quá ngắn
            elif word_count > 40:
                score -= 0.5  # Quá dài
            
            # 3. Điểm từ khóa quan trọng
            keyword_score = 0
            for keyword in important_keywords:
                if keyword in sentence.lower():
                    keyword_score += 0.5
            score += min(keyword_score, 2.0)  # Cap tối đa 2.0
            
            # 4. Điểm dựa trên tần suất từ
            word_freq_score = 0
            for word in words:
                if len(word) > 3:  # Chỉ tính từ dài > 3 ký tự
                    freq_in_text = text.lower().count(word)
                    if freq_in_text > 1:
                        word_freq_score += 0.1
            score += min(word_freq_score, 1.0)
            
            # 5. Điểm số và thông tin cụ thể
            if re.search(r'\d+', sentence):
                score += 0.5  # Có số liệu
            if re.search(r'\d{1,2}[/-]\d{1,2}', sentence):
                score += 0.3  # Có ngày tháng
            
            scored_sentences.append((sentence, score, i))
        
        # Sắp xếp theo điểm số
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Chọn số lượng câu phù hợp
        if len(sentences) <= 4:
            target_count = 2
        elif len(sentences) <= 8:
            target_count = 3
        else:
            target_count = min(4, max(2, len(sentences) // 3))
        
        # Chọn top câu tốt nhất nhưng đảm bảo độ đa dạng
        selected = []
        for sentence, score, original_index in scored_sentences:
            if len(selected) >= target_count:
                break
            
            # Kiểm tra độ tương đồng với câu đã chọn
            is_similar = False
            for selected_sentence in selected:
                common_words = set(sentence.lower().split()) & set(selected_sentence[0].lower().split())
                if len(common_words) > len(sentence.split()) * 0.6:  # >60% từ giống nhau
                    is_similar = True
                    break
            
            if not is_similar:
                selected.append((sentence, score, original_index))
        
        # Sắp xếp lại theo thứ tự gốc
        selected.sort(key=lambda x: x[2])
        
        # Tạo tóm tắt
        summary_sentences = [s[0] for s in selected]
        summary = '. '.join(summary_sentences) + '.'
        
        # Tính tỷ lệ nén
        compression_ratio = len(summary) / len(text)
        
        return {
            'summary': summary,
            'compression_ratio': round(compression_ratio, 3),
            'message': f'Tóm tắt từ {len(sentences)} câu xuống {len(selected)} câu',
            'selected_scores': [round(s[1], 2) for s in selected]
        }
        
    except Exception as e:
        # Fallback đơn giản khi có lỗi
        words = text.split()
        if len(words) > 50:
            summary = ' '.join(words[:40]) + '...'
            return {
                'summary': summary,
                'compression_ratio': 0.6,
                'message': 'Tóm tắt cơ bản do lỗi xử lý'
            }
        return {
            'summary': text,
            'compression_ratio': 1.0,
            'message': f'Lỗi tóm tắt: {str(e)[:50]}'
        }