from django.db import models


class NewsArticle(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    url = models.URLField(unique=True)
    category = models.CharField(max_length=100, blank=True)
    published_date = models.DateTimeField(null=True, blank=True)
    is_fake = models.BooleanField(null=True, blank=True)
    confidence_score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title

    class Meta:
        ordering = ['-published_date']


class AnalysisResult(models.Model):
    article = models.ForeignKey(NewsArticle, on_delete=models.CASCADE)
    
    # Model 1: Fake News Detection
    is_fake_prediction = models.BooleanField()
    fake_confidence_score = models.FloatField()
    
    # Model 2: Text Summarization
    summary = models.TextField(blank=True, null=True)
    compression_ratio = models.FloatField(blank=True, null=True)
    
    # Model 3: Topic Classification
    topic = models.CharField(max_length=50, blank=True, null=True)
    topic_confidence = models.FloatField(blank=True, null=True)
    topic_id = models.IntegerField(blank=True, null=True)
    
    # Model 4: Sentiment Analysis (Future)
    sentiment = models.CharField(max_length=20, blank=True, null=True)
    sentiment_confidence = models.FloatField(blank=True, null=True)
    
    # Meta information
    analysis_date = models.DateTimeField(auto_now_add=True)
    model_version = models.CharField(max_length=50, default='2.0')
    processing_time = models.FloatField(blank=True, null=True)  # in seconds

    def __str__(self):
        return f"Analysis for {self.article.title}"
    
    @property
    def confidence(self):
        """Backward compatibility for template"""
        return self.fake_confidence_score

