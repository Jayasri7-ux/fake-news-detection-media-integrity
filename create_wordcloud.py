from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# Absolute project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Output directory
OUTPUT_DIR = os.path.join(BASE_DIR, "visuals", "wordclouds")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "fake_news_wordcloud.png")

text = """
fake news misinformation false misleading rumor hoax
media truth viral clickbait politics breaking news
"""

wc = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(text)

# SAVE USING PIL (MOST RELIABLE METHOD)
wc.to_image().save(OUTPUT_FILE)

print("IMAGE SAVED SUCCESSFULLY AT:")
print(OUTPUT_FILE)
