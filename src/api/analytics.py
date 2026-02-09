import re
from textblob import TextBlob
from urllib.parse import urlparse

def get_sentiment(text):
    """
    Returns sentiment polarity and a human-readable label.
    """
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.1:
        return polarity, "Positive"
    elif polarity < -0.1:
        return polarity, "Negative"
    else:
        return polarity, "Neutral"

def get_domain_trust(url):
    """
    Very basic domain trust scoring logic based on known patterns.
    Higher is better.
    """
    if not url or not url.startswith('http'):
        return 50, "Unknown"
        
    domain = urlparse(url).netloc.lower()
    
    trusted_tlds = ['.gov', '.edu', '.org']
    reputable_news = ['bbc.com', 'reuters.com', 'apnews.com', 'nytimes.com', 'theguardian.com']
    
    if any(domain.endswith(tld) for tld in trusted_tlds):
        return 90, "High (Trusted TLD)"
    
    if any(news in domain for news in reputable_news):
        return 95, "High (Reputable News Source)"
        
    # Generic suspicion for common "fake" news patterns in domains
    suspicious_patterns = ['daily-truth', 'breaking-fast', 'real-news-only', 'truth-bomb']
    if any(p in domain for p in suspicious_patterns):
        return 20, "Low (Suspicious Domain Pattern)"
        
    return 60, "Moderate (Standard Web Resource)"

def get_readability_stats(text):
    """
    Calculates basic text stats for the dashboard.
    """
    words = text.split()
    word_count = len(words)
    avg_word_len = sum(len(w) for w in words) / word_count if word_count > 0 else 0
    
    return {
        "word_count": word_count,
        "avg_word_length": round(avg_word_len, 2)
    }
