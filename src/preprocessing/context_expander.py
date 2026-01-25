class ContextExpander:
    def expand(self, text):
        words = text.lower().split()

        # For very short messages, expand slightly like social media style
        if len(words) <= 5:
            return f"Social media post claims: {text}"

        return text
