from django import template

register = template.Library()

@register.filter
def percentage(value, total):
    """Tính phần trăm"""
    try:
        return round((value / total) * 100, 2) if total > 0 else 0
    except (ValueError, ZeroDivisionError):
        return 0

@register.filter
def confidence_class(confidence):
    """Trả về class CSS dựa trên độ tin cậy"""
    if confidence >= 0.8:
        return 'high-confidence'
    elif confidence >= 0.6:
        return 'medium-confidence'
    else:
        return 'low-confidence'

@register.filter
def fake_label(is_fake):
    """Trả về nhãn cho tin tức"""
    return 'Tin giả' if is_fake else 'Tin thật'

@register.filter
def multiply(value, arg):
    """Nhân hai số"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0

