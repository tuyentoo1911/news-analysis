from django import forms


class NewsAnalysisForm(forms.Form):
    news_text = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 8,
            'placeholder': 'Nhập nội dung tin tức cần phân tích...'
        }),
        label='Nội dung tin tức',
        max_length=5000,
        required=True,
        help_text='Tối thiểu 50 ký tự'
    )
    
    news_url = forms.URLField(
        widget=forms.URLInput(attrs={
            'class': 'form-control',
            'placeholder': 'https://example.com/news-article (tùy chọn)'
        }),
        label='URL tin tức',
        required=False,
        help_text='Đường dẫn đến bài báo gốc (không bắt buộc)'
    )
    
    def clean_news_text(self):
        news_text = self.cleaned_data.get('news_text')
        if news_text and len(news_text.strip()) < 50:
            raise forms.ValidationError('Nội dung tin tức phải có ít nhất 50 ký tự.')
        return news_text

