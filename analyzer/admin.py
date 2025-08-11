from django.contrib import admin
from .models import NewsArticle, AnalysisResult


@admin.register(NewsArticle)
class NewsArticleAdmin(admin.ModelAdmin):
    list_display = ('title', 'category', 'published_date', 'is_fake', 'confidence_score', 'created_at')
    list_filter = ('category', 'is_fake', 'published_date', 'created_at')
    search_fields = ('title', 'content', 'url')
    readonly_fields = ('created_at', 'updated_at')
    list_per_page = 20
    

@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display = ('article', 'is_fake_prediction', 'fake_confidence_score', 'has_summary', 'analysis_date', 'model_version')
    list_filter = ('is_fake_prediction', 'analysis_date', 'model_version')
    search_fields = ('article__title', 'article__content', 'summary')
    readonly_fields = ('analysis_date', 'processing_time')
    list_per_page = 20
    
    def has_summary(self, obj):
        return bool(obj.summary)
    has_summary.boolean = True
    has_summary.short_description = 'Has Summary'

