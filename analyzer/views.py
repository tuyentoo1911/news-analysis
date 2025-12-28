from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.db import models
import json
import requests
from bs4 import BeautifulSoup
import re
from .models import NewsArticle, AnalysisResult
from .ml_utils import preprocess_text, predict_fake_news, summarize_text, analyze_topic, analyze_sentiment


def fetch_content_from_url(url):
    """Lấy nội dung tin tức từ URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Xóa các thẻ không cần thiết
        for script in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            script.decompose()

        content_selectors = [
            'article',
            '.article-content',
            '.post-content',
            '.content',
            '.entry-content',
            '.main-content',
            '#content',
            '.article-body',
            '.story-body',
            '.news-content',
            'main'
        ]

        content = ""
        title = ""

        # Lấy tiêu đề
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()

        # Tìm nội dung theo thứ tự ưu tiên
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                paragraphs = element.find_all(['p', 'div'], string=True)
                content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                if len(content) > 100:
                    break

        if len(content) < 100:
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])

        # Làm sạch nội dung
        content = re.sub(r'\s+', ' ', content).strip()

        if len(content) < 50:
            return None, "Không thể lấy đủ nội dung từ URL này"

        return {
            'title': title,
            'content': content,
            'url': url
        }, None

    except requests.RequestException as e:
        return None, f"Lỗi kết nối: {str(e)}"
    except Exception as e:
        return None, f"Lỗi xử lý: {str(e)}"


def home(request):
    """Trang chủ"""
    return render(request, 'analyzer/home.html')


def about(request):
    """Trang giới thiệu"""
    return render(request, 'analyzer/about.html')


def analyze(request):
    """Trang phân tích tin tức với 3 phương thức nhập: URL, Text, File"""
    if request.method == 'POST':
        news_text = request.POST.get('news_text', '').strip()
        news_url = request.POST.get('news_url', '').strip()
        news_file = request.FILES.get('news_file')

        input_source = "unknown"

        # Xử lý file upload
        if news_file:
            input_source = "file"
            try:
                if news_file.size > 10 * 1024 * 1024:
                    messages.error(request, 'File quá lớn! Vui lòng chọn file nhỏ hơn 10MB.')
                    return render(request, 'analyzer/analyze.html')

                # Đọc nội dung file
                if news_file.name.endswith('.txt'):
                    news_text = news_file.read().decode('utf-8')
                elif news_file.name.endswith('.docx'):
                    try:
                        import docx
                        doc = docx.Document(news_file)
                        news_text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                    except ImportError:
                        messages.error(request, 'Không hỗ trợ file .docx. Vui lòng sử dụng file .txt.')
                        return render(request, 'analyzer/analyze.html')
                elif news_file.name.endswith('.pdf'):
                    try:
                        import PyPDF2
                        pdf_reader = PyPDF2.PdfReader(news_file)
                        news_text = ''
                        for page in pdf_reader.pages:
                            news_text += page.extract_text()
                    except ImportError:
                        messages.error(request, 'Không hỗ trợ file .pdf. Vui lòng sử dụng file .txt.')
                        return render(request, 'analyzer/analyze.html')
                else:
                    messages.error(request, 'Định dạng file không được hỗ trợ. Chỉ chấp nhận .txt, .docx, .pdf')
                    return render(request, 'analyzer/analyze.html')

                news_text = news_text.strip()
                messages.success(request, f'Đã tải thành công file "{news_file.name}" ({len(news_text)} ký tự)')

            except Exception as e:
                messages.error(request, f'Lỗi khi đọc file: {str(e)}')
                return render(request, 'analyzer/analyze.html')

        # Xử lý URL
        elif news_url and not news_text:
            input_source = "url"
            fetched_data, error = fetch_content_from_url(news_url)
            if error:
                messages.error(request, f'Không thể lấy nội dung từ URL: {error}')
                return render(request, 'analyzer/analyze.html', {'news_url': news_url})
            else:
                news_text = fetched_data['content']
                messages.success(request, f'Đã tự động lấy nội dung từ URL: {fetched_data["title"][:100]}...')

        # Xử lý text input
        elif news_text:
            input_source = "text"

        # Kiểm tra nội dung
        if not news_text:
            messages.error(request, 'Vui lòng nhập nội dung tin tức, URL hoặc upload file để phân tích.')
            return render(request, 'analyzer/analyze.html')

        if len(news_text) < 50:
            messages.error(request, 'Nội dung tin tức phải có ít nhất 50 ký tự.')
            context = {
                'news_text': news_text if input_source == "text" else "",
                'news_url': news_url if input_source == "url" else ""
            }
            return render(request, 'analyzer/analyze.html', context)

        try:
            import time
            start_time = time.time()

            # Model 1: Phân tích cảm xúc
            sentiment_result = analyze_sentiment(news_text)

            # Model 2: Phân tích tin giả
            fake_news_result = predict_fake_news(news_text)

            # Model 3: Tóm tắt văn bản
            summary_result = summarize_text(news_text)

            # Model 4: Phân loại chủ đề
            topic_result = analyze_topic(news_text)

            processing_time = time.time() - start_time

            # Kết hợp kết quả từ 4 model
            result = {
                # Model 1 - Sentiment Analysis
                'sentiment': sentiment_result['sentiment'],
                'sentiment_confidence': sentiment_result['confidence'],
                'sentiment_confidence_percent': round(sentiment_result['confidence'] * 100, 1),
                'sentiment_emoji': sentiment_result['emoji'],
                'sentiment_id': sentiment_result['sentiment_id'],

                # Model 2 - Fake News Detection
                'is_fake': fake_news_result['is_fake'],
                'confidence': fake_news_result['confidence'],
                'confidence_percent': round(fake_news_result['confidence'] * 100, 1),

                # Model 3 - Text Summarization
                'summary': summary_result['summary'],
                'compression_ratio': summary_result['compression_ratio'],
                'compression_percent': round(summary_result['compression_ratio'] * 100, 1),

                # Model 4 - Topic Classification
                'topic': topic_result['topic'],
                'topic_confidence': topic_result['confidence'],
                'topic_confidence_percent': round(topic_result['confidence'] * 100, 1),
                'topic_id': topic_result['topic_id'],

                # Meta information
                'message': f"Cảm xúc: {sentiment_result['message']} • Tin giả: {fake_news_result['message']} • Tóm tắt: {summary_result['message']} • Chủ đề: {topic_result['message']}",
                'processing_time': round(processing_time, 3),
                'models_used': ['Sentiment Analysis', 'Fake News Detection', 'Text Summarization', 'Topic Classification']
            }

            # Lưu kết quả vào database cho tất cả loại input
            try:
                # Tạo title từ nội dung nếu không có URL
                article_title = news_text[:255] if len(news_text) <= 255 else news_text[:252] + '...'

                # Xác định input source
                if news_url:
                    article, created = NewsArticle.objects.get_or_create(
                        url=news_url,
                        defaults={
                            'title': article_title,
                            'content': news_text,
                            'input_source': 'url',
                            'category': result['topic'],
                            'published_date': timezone.now()
                        }
                    )
                else:
                    # Lưu text hoặc file input (không có URL unique constraint)
                    article = NewsArticle.objects.create(
                        title=article_title,
                        content=news_text,
                        input_source=input_source,  # 'text' hoặc 'file'
                        category=result['topic'],
                        published_date=timezone.now()
                    )
                    created = True

                # Lưu kết quả phân tích
                AnalysisResult.objects.create(
                    article=article,
                    # Sentiment Analysis
                    sentiment=result['sentiment'],
                    sentiment_confidence=result['sentiment_confidence'],
                    # Fake News Detection
                    is_fake_prediction=result['is_fake'],
                    fake_confidence_score=result['confidence'],
                    # Text Summarization
                    summary=result['summary'],
                    compression_ratio=result['compression_ratio'],
                    # Topic Classification
                    topic=result['topic'],
                    topic_confidence=result['topic_confidence'],
                    topic_id=result['topic_id'],
                    processing_time=processing_time,
                    model_version='4.0'  # Cập nhật version vì thêm sentiment model
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
    """Trang thống kê & lịch sử"""
    try:
        total_articles = NewsArticle.objects.count()
        total_analyses = AnalysisResult.objects.count()
        fake_count = AnalysisResult.objects.filter(is_fake_prediction=True).count()
        real_count = AnalysisResult.objects.filter(is_fake_prediction=False).count()

        # Thống kê theo nguồn input
        url_count = NewsArticle.objects.filter(input_source='url').count()
        text_count = NewsArticle.objects.filter(input_source='text').count()
        file_count = NewsArticle.objects.filter(input_source='file').count()

        # Thống kê theo cảm xúc
        sentiment_stats = AnalysisResult.objects.values('sentiment').annotate(
            count=models.Count('id')
        ).order_by('-count')

        positive_count = sum(item['count'] for item in sentiment_stats if item['sentiment'] == 'tích cực')
        negative_count = sum(item['count'] for item in sentiment_stats if item['sentiment'] == 'tiêu cực')
        neutral_count = sum(item['count'] for item in sentiment_stats if item['sentiment'] == 'trung tính')

        # Thống kê theo chủ đề
        topic_stats = AnalysisResult.objects.values('topic').annotate(
            count=models.Count('id')
        ).order_by('-count').exclude(topic__isnull=True)[:10]

        # Thống kê cho model summarization
        analyses_with_summary = AnalysisResult.objects.filter(summary__isnull=False).count()
        avg_compression_ratio = AnalysisResult.objects.filter(
            compression_ratio__isnull=False
        ).aggregate(avg_ratio=models.Avg('compression_ratio'))['avg_ratio']

        # Thống kê processing time
        avg_processing_time = AnalysisResult.objects.filter(
            processing_time__isnull=False
        ).aggregate(avg_time=models.Avg('processing_time'))['avg_time']

        # Thống kê độ tin cậy trung bình
        avg_fake_confidence = AnalysisResult.objects.filter(
            fake_confidence_score__isnull=False
        ).aggregate(avg_conf=models.Avg('fake_confidence_score'))['avg_conf']

        avg_sentiment_confidence = AnalysisResult.objects.filter(
            sentiment_confidence__isnull=False
        ).aggregate(avg_conf=models.Avg('sentiment_confidence'))['avg_conf']

        avg_topic_confidence = AnalysisResult.objects.filter(
            topic_confidence__isnull=False
        ).aggregate(avg_conf=models.Avg('topic_confidence'))['avg_conf']

        # Lịch sử phân tích với phân trang và lọc
        articles = NewsArticle.objects.all().order_by('-created_at')

        # Lọc theo nguồn input nếu có
        source_filter = request.GET.get('source', 'all')
        if source_filter != 'all':
            articles = articles.filter(input_source=source_filter)

        # Phân trang: 10 bài mỗi trang
        from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
        paginator = Paginator(articles, 10)
        page = request.GET.get('page', 1)

        try:
            articles_page = paginator.page(page)
        except PageNotAnInteger:
            articles_page = paginator.page(1)
        except EmptyPage:
            articles_page = paginator.page(paginator.num_pages)

        context = {
            # Tổng quan
            'total_articles': total_articles,
            'total_analyses': total_analyses,
            'fake_count': fake_count,
            'real_count': real_count,
            'fake_percentage': round((fake_count / total_analyses * 100) if total_analyses > 0 else 0, 1),
            'real_percentage': round((real_count / total_analyses * 100) if total_analyses > 0 else 0, 1),

            # Thống kê theo nguồn
            'url_count': url_count,
            'text_count': text_count,
            'file_count': file_count,
            'url_percentage': round((url_count / total_articles * 100) if total_articles > 0 else 0, 1),
            'text_percentage': round((text_count / total_articles * 100) if total_articles > 0 else 0, 1),
            'file_percentage': round((file_count / total_articles * 100) if total_articles > 0 else 0, 1),

            # Thống kê cảm xúc
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'positive_percentage': round((positive_count / total_analyses * 100) if total_analyses > 0 else 0, 1),
            'negative_percentage': round((negative_count / total_analyses * 100) if total_analyses > 0 else 0, 1),
            'neutral_percentage': round((neutral_count / total_analyses * 100) if total_analyses > 0 else 0, 1),

            # Thống kê chủ đề
            'topic_stats': topic_stats,

            # Model 2: Text Summarization
            'analyses_with_summary': analyses_with_summary,
            'summary_percentage': round((analyses_with_summary / total_analyses * 100) if total_analyses > 0 else 0, 1),
            'avg_compression_ratio': round(avg_compression_ratio * 100, 1) if avg_compression_ratio else 0,

            # Performance Stats
            'avg_processing_time': round(avg_processing_time, 2) if avg_processing_time else 0,
            'avg_fake_confidence': round(avg_fake_confidence * 100, 1) if avg_fake_confidence else 0,
            'avg_sentiment_confidence': round(avg_sentiment_confidence * 100, 1) if avg_sentiment_confidence else 0,
            'avg_topic_confidence': round(avg_topic_confidence * 100, 1) if avg_topic_confidence else 0,

            # Model versions
            'model_versions': AnalysisResult.objects.values('model_version').annotate(
                count=models.Count('id')
            ).order_by('-count')[:5],

            # History section with pagination
            'articles': articles_page,
            'source_filter': source_filter,
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
            'url_count': 0,
            'text_count': 0,
            'file_count': 0,
            'url_percentage': 0,
            'text_percentage': 0,
            'file_percentage': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'positive_percentage': 0,
            'negative_percentage': 0,
            'neutral_percentage': 0,
            'topic_stats': [],
            'analyses_with_summary': 0,
            'summary_percentage': 0,
            'avg_compression_ratio': 0,
            'avg_processing_time': 0,
            'avg_fake_confidence': 0,
            'avg_sentiment_confidence': 0,
            'avg_topic_confidence': 0,
            'model_versions': [],
            'articles': [],
            'source_filter': 'all',
        }

    return render(request, 'analyzer/stats.html', context)


def delete_article(request, article_id):
    """Xóa một bài báo khỏi lịch sử"""
    if request.method == 'POST':
        try:
            article = NewsArticle.objects.get(id=article_id)
            article_title = article.title[:50]
            article.delete()
            messages.success(request, f'Đã xóa bài báo: "{article_title}..."')
        except NewsArticle.DoesNotExist:
            messages.error(request, 'Bài báo không tồn tại.')
        except Exception as e:
            messages.error(request, f'Lỗi khi xóa bài báo: {str(e)}')

    # Redirect về trang thống kê
    return redirect('analyzer:stats')


def delete_all_history(request):
    """Xóa tất cả lịch sử phân tích"""
    if request.method == 'POST':
        try:
            # Xóa tất cả các bài báo (các AnalysisResult sẽ tự động xóa do CASCADE)
            count = NewsArticle.objects.count()
            NewsArticle.objects.all().delete()
            messages.success(request, f'Đã xóa thành công {count} bài báo khỏi lịch sử.')
        except Exception as e:
            messages.error(request, f'Lỗi khi xóa lịch sử: {str(e)}')

    # Redirect về trang thống kê
    return redirect('analyzer:stats')

