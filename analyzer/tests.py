from django.test import TestCase, Client
from django.urls import reverse
from django.utils import timezone
from .models import NewsArticle, AnalysisResult
from .ml_utils import preprocess_text, predict_fake_news


class AnalyzerModelTest(TestCase):
    def setUp(self):
        self.article = NewsArticle.objects.create(
            title="Test Article",
            content="This is a test article content",
            url="https://example.com/test",
            category="Test",
            published_date=timezone.now()
        )

    def test_article_creation(self):
        self.assertEqual(self.article.title, "Test Article")
        self.assertEqual(str(self.article), "Test Article")

    def test_analysis_result_creation(self):
        result = AnalysisResult.objects.create(
            article=self.article,
            is_fake_prediction=True,
            confidence_score=0.85,
            model_version="1.0"
        )
        self.assertEqual(result.article, self.article)
        self.assertTrue(result.is_fake_prediction)


class AnalyzerViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_home_view(self):
        response = self.client.get(reverse('analyzer:home'))
        self.assertEqual(response.status_code, 200)

    def test_analyze_view_get(self):
        response = self.client.get(reverse('analyzer:analyze'))
        self.assertEqual(response.status_code, 200)

    def test_analyze_view_post_valid(self):
        response = self.client.post(reverse('analyzer:analyze'), {
            'news_text': 'This is a test news content with more than fifty characters to meet the minimum requirement'
        })
        self.assertEqual(response.status_code, 200)

    def test_analyze_view_post_invalid(self):
        response = self.client.post(reverse('analyzer:analyze'), {
            'news_text': 'Short'  # Less than 50 characters
        })
        self.assertEqual(response.status_code, 200)

    def test_stats_view(self):
        response = self.client.get(reverse('analyzer:stats'))
        self.assertEqual(response.status_code, 200)

    def test_about_view(self):
        response = self.client.get(reverse('analyzer:about'))
        self.assertEqual(response.status_code, 200)


class MLUtilsTest(TestCase):
    def test_preprocess_text(self):
        text = "This is a TEST text with 123 numbers!"
        processed = preprocess_text(text)
        self.assertEqual(processed, "this is a test text with numbers")

    def test_preprocess_text_empty(self):
        processed = preprocess_text("")
        self.assertEqual(processed, "")

    def test_predict_fake_news(self):
        text = "This is a normal news article with sufficient content to test the analysis function properly"
        result = predict_fake_news(text)
        self.assertIn('is_fake', result)
        self.assertIn('confidence', result)
        self.assertIn('message', result)
        self.assertIsInstance(result['is_fake'], bool)
        self.assertIsInstance(result['confidence'], float)

    def test_predict_fake_news_suspicious(self):
        text = "GIẬT GÂN!!! Tin sốc không thể tin được về sự kiện bí mật này!!!"
        result = predict_fake_news(text)
        # Với văn bản có nhiều từ đáng ngờ, khả năng cao sẽ được đánh giá là tin giả
        self.assertIsInstance(result['is_fake'], bool)

