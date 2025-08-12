# -*- coding: utf-8 -*-
"""
AI Models Module
================
Ba mô hình AI mô phỏng: Phân tích cảm xúc, Phát hiện tin giả, Phân loại chủ đề.

Không phụ thuộc vào mô hình nặng; dùng luật/keyword và thống kê đơn giản để chạy offline.
"""

from __future__ import annotations

import random
import time
import numpy as np
from typing import Dict, List, Optional


class SentimentAnalyzer:
    """Mô hình phân tích cảm xúc (giả lập dựa trên từ khóa + heuristics)."""

    def __init__(self) -> None:
        self.positive_words: List[str] = [
            'tuyệt vời', 'xuất sắc', 'tốt', 'hay', 'thích', 'yêu', 'hạnh phúc', 'vui',
            'thú vị', 'tích cực', 'đẹp', 'hoàn hảo', 'ấn tượng', 'thành công', 'khích lệ'
        ]
        self.negative_words: List[str] = [
            'tệ', 'xấu', 'ghét', 'buồn', 'thất vọng', 'tức giận', 'khủng khiếp', 'kinh khủng',
            'tiêu cực', 'thất bại', 'sai lầm', 'điên rồ', 'tồi tệ', 'bất cập'
        ]

    def analyze(self, text: str) -> Dict:
        if not text or len(text.strip()) < 10:
            return {
                'sentiment': 'Trung tính',
                'confidence': 0.5,
                'probabilities': {'Tích cực': 0.33, 'Tiêu cực': 0.33, 'Trung tính': 0.34},
                'keywords': []
            }

        time.sleep(0.2)
        text_lower = text.lower()

        positive_count = sum(1 for w in self.positive_words if w in text_lower)
        negative_count = sum(1 for w in self.negative_words if w in text_lower)

        total = positive_count + negative_count
        if total == 0:
            # Không có từ khóa cảm xúc rõ ràng -> trung tính
            sentiment = 'Trung tính'
            confidence = 0.55
            probs = {'Tích cực': 0.25, 'Tiêu cực': 0.25, 'Trung tính': 0.5}
        else:
            pos_ratio = positive_count / total
            neg_ratio = negative_count / total

            if pos_ratio - neg_ratio > 0.15:
                sentiment = 'Tích cực'
                confidence = 0.65 + pos_ratio * 0.3
            elif neg_ratio - pos_ratio > 0.15:
                sentiment = 'Tiêu cực'
                confidence = 0.65 + neg_ratio * 0.3
            else:
                sentiment = 'Trung tính'
                confidence = 0.5 + abs(pos_ratio - neg_ratio) * 0.3

            if sentiment == 'Tích cực':
                probs = {
                    'Tích cực': round(confidence, 3),
                    'Tiêu cực': round((1 - confidence) * 0.35, 3),
                    'Trung tính': 0
                }
            elif sentiment == 'Tiêu cực':
                probs = {
                    'Tích cực': round((1 - confidence) * 0.35, 3),
                    'Tiêu cực': round(confidence, 3),
                    'Trung tính': 0
                }
            else:
                probs = {
                    'Tích cực': round((1 - confidence) * 0.5, 3),
                    'Tiêu cực': round((1 - confidence) * 0.5, 3),
                    'Trung tính': round(confidence, 3)
                }

            # Chuẩn hóa xác suất
            s = sum(probs.values())
            probs = {k: round(v / s, 3) for k, v in probs.items()}

        keywords: List[str] = []
        for w in self.positive_words + self.negative_words:
            if w in text_lower:
                keywords.append(w)

        return {
            'sentiment': sentiment,
            'confidence': round(float(confidence), 3),
            'probabilities': probs,
            'keywords': keywords[:5]
        }


class FakeNewsDetector:
    """Mô hình phát hiện tin giả (giả lập dựa trên tín hiệu ngôn ngữ và định dạng)."""

    def __init__(self) -> None:
        self.fake_indicators = [
            'độc quyền', 'bí mật', 'chấn động', 'không thể tin được', 'khẩn cấp', 'chia sẻ ngay',
            'bạn sẽ không tin', '100%', 'tuyệt đối', 'không ai biết', 'giật gân', 'siêu hot'
        ]
        self.reliable_indicators = [
            'theo nghiên cứu', 'chuyên gia cho biết', 'dữ liệu thống kê', 'báo cáo chính thức',
            'nguồn tin đáng tin cậy', 'xác nhận từ', 'nghiên cứu khoa học', 'theo báo cáo'
        ]

    def analyze(self, text: str) -> Dict:
        if not text or len(text.strip()) < 15:
            return {
                'prediction': 'Không đủ thông tin',
                'confidence': 0.5,
                'is_fake': None,
                'indicators': {'fake_signals': [], 'reliable_signals': []},
                'risk_score': 0.5,
            }

        time.sleep(0.2)
        lower = text.lower()

        fake_count = sum(1 for k in self.fake_indicators if k in lower)
        rel_count = sum(1 for k in self.reliable_indicators if k in lower)

        exclam = text.count('!')
        caps_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))

        risk = 0.5
        risk += 0.2 * fake_count
        risk -= 0.15 * rel_count
        if exclam >= 3:
            risk += 0.1
        if caps_ratio > 0.1:
            risk += 0.15

        risk += random.uniform(-0.08, 0.08)
        risk = max(0.05, min(0.95, risk))

        if risk > 0.65:
            pred = 'Tin giả'
            is_fake = True
            conf = risk
        elif risk < 0.35:
            pred = 'Tin thật'
            is_fake = False
            conf = 1 - risk
        else:
            pred = 'Không chắc chắn'
            is_fake = None
            conf = 0.5

        return {
            'prediction': pred,
            'confidence': round(float(conf), 3),
            'is_fake': is_fake,
            'indicators': {
                'fake_signals': [k for k in self.fake_indicators if k in lower][:3],
                'reliable_signals': [k for k in self.reliable_indicators if k in lower][:3],
            },
            'risk_score': round(float(risk), 3),
            'text_stats': {
                'exclamation_count': exclam,
                'caps_ratio': round(float(caps_ratio), 3),
            },
        }


class TopicClassifier:
    """Mô hình phân loại chủ đề (giả lập dựa trên keyword matching có trọng số)."""

    def __init__(self) -> None:
        self.topics: Dict[str, Dict] = {
            'Công nghệ': {
                'keywords': ['công nghệ', 'ai', 'robot', 'máy tính', 'phần mềm', 'internet', 'ứng dụng', 'dữ liệu', 'blockchain'],
                'description': 'Tin tức về công nghệ, AI và chuyển đổi số',
            },
            'Kinh tế': {
                'keywords': ['kinh tế', 'đầu tư', 'chứng khoán', 'ngân hàng', 'tài chính', 'thị trường', 'doanh nghiệp', 'gdp', 'lạm phát'],
                'description': 'Kinh tế vĩ mô, tài chính doanh nghiệp, thị trường',
            },
            'Sức khỏe': {
                'keywords': ['sức khỏe', 'bệnh viện', 'bác sĩ', 'thuốc', 'điều trị', 'y tế', 'vaccine', 'dinh dưỡng', 'bệnh tật'],
                'description': 'Y tế, sức khỏe cộng đồng, chăm sóc sức khỏe',
            },
            'Giáo dục': {
                'keywords': ['giáo dục', 'học sinh', 'sinh viên', 'trường học', 'đại học', 'giáo viên', 'thi cử', 'khóa học'],
                'description': 'Hệ thống giáo dục và đào tạo',
            },
            'Thể thao': {
                'keywords': ['thể thao', 'bóng đá', 'bóng rổ', 'tennis', 'giải đấu', 'vận động viên', 'huấn luyện', 'trận đấu'],
                'description': 'Sự kiện thể thao và vận động viên',
            },
            'Giải trí': {
                'keywords': ['giải trí', 'phim', 'ca sĩ', 'diễn viên', 'âm nhạc', 'show', 'sân khấu', 'điện ảnh'],
                'description': 'Văn hóa, nghệ thuật, điện ảnh - âm nhạc',
            },
            'Chính trị': {
                'keywords': ['chính trị', 'chính phủ', 'thủ tướng', 'bộ trưởng', 'quốc hội', 'luật', 'bầu cử', 'chính sách'],
                'description': 'Tin tức chính trị và chính sách',
            },
            'Xã hội': {
                'keywords': ['xã hội', 'gia đình', 'trẻ em', 'người già', 'phụ nữ', 'cộng đồng', 'đời sống', 'văn hóa'],
                'description': 'Đời sống và các vấn đề xã hội',
            },
        }

    def analyze(self, text: str) -> Dict:
        if not text or len(text.strip()) < 12:
            return {
                'topic': 'Không xác định',
                'confidence': 0.3,
                'keywords': [],
                'description': 'Không đủ thông tin để phân loại',
                'probabilities': {}
            }

        time.sleep(0.2)
        lower = text.lower()
        scores: Dict[str, float] = {}
        topic_keywords: Dict[str, List[str]] = {}

        for name, info in self.topics.items():
            score = 0.0
            found: List[str] = []
            for kw in info['keywords']:
                if kw in lower:
                    count = lower.count(kw)
                    score += count * max(1.0, len(kw) / 6.0)
                    found.extend([kw] * count)
            scores[name] = score
            topic_keywords[name] = found[:5]

        if all(v == 0 for v in scores.values()):
            topic = random.choice(list(self.topics.keys()))
            confidence = 0.35
            found_keywords: List[str] = []
        else:
            topic = max(scores, key=scores.get)
            total = sum(scores.values())
            confidence = 0.5 + (scores[topic] / total) * 0.45 if total > 0 else 0.5
            found_keywords = topic_keywords[topic]

        # Probabilities for all topics
        total = sum(scores.values())
        if total > 0:
            probs = {k: round(v / total, 3) for k, v in scores.items()}
        else:
            base = round(1.0 / len(self.topics), 3)
            probs = {k: base for k in self.topics.keys()}

        return {
            'topic': topic,
            'confidence': round(float(confidence), 3),
            'keywords': list(dict.fromkeys(found_keywords)),
            'description': self.topics[topic]['description'],
            'probabilities': dict(sorted(probs.items(), key=lambda x: x[1], reverse=True)),
        }


# Khởi tạo và hàm tiện cho phần app
sentiment_analyzer = SentimentAnalyzer()
fake_news_detector = FakeNewsDetector()
topic_classifier = TopicClassifier()


def analyze_text(text: str) -> Dict:
    """Chạy cả 3 mô hình và trả về kết quả tổng hợp."""
    return {
        'sentiment': sentiment_analyzer.analyze(text),
        'fake_news': fake_news_detector.analyze(text),
        'topic': topic_classifier.analyze(text),
        'word_count': len(text.split()) if text else 0,
        'text_length': len(text) if text else 0,
    }






