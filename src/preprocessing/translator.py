from deep_translator import GoogleTranslator

class Translator:
    def __init__(self):
        # We will use GoogleTranslator from deep-translator
        self.en_translator = GoogleTranslator(source='auto', target='en')

    def translate_to_english(self, text, lang=None):
        """
        Translates input text to English for AI analysis.
        """
        if not text or len(text.strip()) < 5:
            return text
        
        try:
            # If lang is provided and is not 'en', force translation
            # 'auto' usually works well for deep-translator
            return self.en_translator.translate(text)
        except Exception as e:
            print(f"Translation to English failed: {e}")
            return text

    def translate_to_target(self, text, target_lang):
        """
        Translates text to a specific target language (e.g., 'hi', 'te').
        """
        if not text or target_lang == 'en' or len(text.strip()) < 5:
            return text

        try:
            translator = GoogleTranslator(source='auto', target=target_lang)
            return translator.translate(text)
        except Exception as e:
            print(f"Translation to {target_lang} failed: {e}")
            return text
