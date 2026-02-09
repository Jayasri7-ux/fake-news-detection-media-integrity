from src.preprocessing.translator import Translator
import sys

t = Translator()
text = "This is a test news article."
target = "te"

try:
    print(f"Testing translation to {target}...")
    result = t.translate_to_target(text, target)
    print(f"Result: {result}")
    
    hindi_text = "अगले दो दिनों में भारी बारिश की संभावना है।"
    print(f"Testing translation of Hindi to English...")
    eng_result = t.translate_to_english(hindi_text)
    print(f"Result: {eng_result}")
    
except Exception as e:
    print(f"Error: {e}")
