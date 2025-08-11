from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.db import models
import json
from .models import NewsArticle, AnalysisResult
from .ml_utils import preprocess_text, predict_fake_news, summarize_text


def home(request):
    """Trang chủ"""
    return render(request, 'analyzer/home.html')


def about(request):
    """Trang giới thiệu"""
    return render(request, 'analyzer/about.html')


def analyze(request):
    """Trang phân tích tin tức"""
    if request.method == 'POST':
        news_text = request.POST.get('news_text', '').strip()
        news_url = request.POST.get('news_url', '').strip()
        
        if not news_text:
            messages.error(request, 'Vui lòng nhập nội dung tin tức để phân tích.')
            return render(request, 'analyzer/analyze.html')
        
        if len(news_text) < 50:
            messages.error(request, 'Nội dung tin tức phải có ít nhất 50 ký tự.')
            return render(request, 'analyzer/analyze.html')
        
        try:
            import time
            start_time = time.time()
            
            # Model 1: Phân tích tin giả
            fake_news_result = predict_fake_news(news_text)
            
            # Model 2: Tóm tắt văn bản
            summary_result = summarize_text(news_text)
            
            processing_time = time.time() - start_time
            
            # Kết hợp kết quả từ 2 model
            result = {
                # Model 1 - Fake News Detection
                'is_fake': fake_news_result['is_fake'],
                'confidence': fake_news_result['confidence'],
                
                # Model 2 - Text Summarization
                'summary': summary_result['summary'],
                'compression_ratio': summary_result['compression_ratio'],
                
                # Meta information
                'message': f"Phân tích: {fake_news_result['message']} • Tóm tắt: {summary_result['message']}",
                'processing_time': round(processing_time, 3),
                'models_used': ['Fake News Detection', 'Text Summarization']
            }
            
            # Lưu kết quả vào database nếu có URL
            if news_url:
                try:
                    article, created = NewsArticle.objects.get_or_create(
                        url=news_url,
                        defaults={
                            'title': news_text[:255],
                            'content': news_text,
                            'category': 'Unknown',
                            'published_date': timezone.now()
                        }
                    )
                    
                    AnalysisResult.objects.create(
                        article=article,
                        is_fake_prediction=result['is_fake'],
                        fake_confidence_score=result['confidence'],
                        summary=result['summary'],
                        compression_ratio=result['compression_ratio'],
                        processing_time=processing_time,
                        model_version='2.0'
                    )
                except Exception as db_error:
                    # Nếu có lỗi database, vẫn hiển thị kết quả
                    print(f"Database error: {db_error}")
            
            context = {
                'result': result,
                'news_text': news_text,
                'news_url': news_url
            }
            return render(request, 'analyzer/analyze.html', context)
            
        except Exception as e:
            messages.error(request, f'Có lỗi xảy ra khi phân tích: {str(e)}')
            return render(request, 'analyzer/analyze.html', {'news_text': news_text, 'news_url': news_url})
    
    return render(request, 'analyzer/analyze.html')


@csrf_exempt
def analyze_api(request):
    """API phân tích tin tức với 2 model"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            news_text = data.get('text', '').strip()
            
            if not news_text:
                return JsonResponse({
                    'error': 'Text is required'
                }, status=400)
            
            import time
            start_time = time.time()
            
            # Model 1: Fake News Detection
            fake_news_result = predict_fake_news(news_text)
            
            # Model 2: Text Summarization
            summary_result = summarize_text(news_text)
            
            processing_time = time.time() - start_time
            
            # Kết quả tối ưu
            result = {
                'fake_news': {
                    'is_fake': fake_news_result['is_fake'],
                    'confidence': fake_news_result['confidence'],
                    'status': fake_news_result['message']
                },
                'summary': {
                    'text': summary_result['summary'],
                    'ratio': summary_result['compression_ratio'],
                    'status': summary_result['message']
                },
                'meta': {
                    'time': round(processing_time, 3),
                    'models': 2,
                    'version': '2.1'
                }
            }
            
            return JsonResponse(result)
            
        except Exception as e:
            return JsonResponse({
                'error': str(e)
            }, status=500)
    
    return JsonResponse({
        'error': 'Only POST method allowed'
    }, status=405)


def stats(request):
    """Trang thống kê"""
    try:
        total_articles = NewsArticle.objects.count()
        total_analyses = AnalysisResult.objects.count()
        fake_count = AnalysisResult.objects.filter(is_fake_prediction=True).count()
        real_count = AnalysisResult.objects.filter(is_fake_prediction=False).count()
        
        # Thống kê cho model summarization
        analyses_with_summary = AnalysisResult.objects.filter(summary__isnull=False).count()
        avg_compression_ratio = AnalysisResult.objects.filter(
            compression_ratio__isnull=False
        ).aggregate(avg_ratio=models.Avg('compression_ratio'))['avg_ratio']
        
        # Thống kê processing time
        avg_processing_time = AnalysisResult.objects.filter(
            processing_time__isnull=False
        ).aggregate(avg_time=models.Avg('processing_time'))['avg_time']
        
        context = {
            # Model 1: Fake News Detection
            'total_articles': total_articles,
            'total_analyses': total_analyses,
            'fake_count': fake_count,
            'real_count': real_count,
            'fake_percentage': round((fake_count / total_analyses * 100) if total_analyses > 0 else 0, 2),
            'real_percentage': round((real_count / total_analyses * 100) if total_analyses > 0 else 0, 2),
            
            # Model 2: Text Summarization
            'analyses_with_summary': analyses_with_summary,
            'summary_percentage': round((analyses_with_summary / total_analyses * 100) if total_analyses > 0 else 0, 2),
            'avg_compression_ratio': round(avg_compression_ratio * 100, 1) if avg_compression_ratio else 0,
            
            # Performance Stats
            'avg_processing_time': round(avg_processing_time, 3) if avg_processing_time else 0,
            
            # Model versions
            'model_versions': AnalysisResult.objects.values('model_version').annotate(
                count=models.Count('id')
            ).order_by('-count')[:5]
        }
    except Exception as e:
        # Nếu có lỗi database, hiển thị dữ liệu mặc định
        context = {
            'total_articles': 0,
            'total_analyses': 0,
            'fake_count': 0,
            'real_count': 0,
            'fake_percentage': 0,
            'real_percentage': 0,
            'analyses_with_summary': 0,
            'summary_percentage': 0,
            'avg_compression_ratio': 0,
            'avg_processing_time': 0,
            'model_versions': []
        }
    
    return render(request, 'analyzer/stats.html', context)

