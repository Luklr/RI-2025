import re
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

class Tokenizer:
    PATTERNS = {
        "abbreviations": r'\b(?:Dr|Lic|Ing|Sr|Sra|S\.A|etc)\.\b',
        "acronyms": r'\b[A-Z](?:\.?[A-Z]+){1,7}\b',
        "numbers": r'\b\d{1,4}(?:[.-]?\d{1,4}){0,3}\b',
        "urls": r'\b(?:ftp|https|http)?://[^\s/$.?#].[^\s]*\b',
        "emails": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b',
        "names": r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b(?!\.)',
        "dates": r'\b\d{2}[-/.]\d{2}[-/.]\d{4}\b',
        "words": r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+(?:-[a-zA-Z]+)*'
    }

    def __init__(self, **kwargs):
        patterns = [regex for key, regex in self.PATTERNS.items() if kwargs.get(key, False)]
        self.regex = re.compile('|'.join(patterns) if patterns else '|'.join(self.PATTERNS.values()))
        self.porter = PorterStemmer()
        self.lancaster = LancasterStemmer()
        self.snowball = SnowballStemmer("spanish")

    def remove_html_tags(self, text_list):
        return [re.sub(r'<[^>]+>', '', text) for text in text_list]

    def remove_accents(self, text_list):
        replacement = str.maketrans("áéíóúüÁÉÍÓÚÜ", "aeiouuAEIOUU")
        text_without_accents = []
        for word in text_list:
            word = word.translate(replacement)
            text_without_accents.append(word)
        return text_without_accents
    
    def stemming(self, text_list, method = "porter"):
        if method == "porter":
            stemmed_list = [self.porter.stem(word) for word in text_list]
        elif method == "lancaster":
            stemmed_list = [self.lancaster.stem(word) for word in text_list]
        elif method == "snowball":
            stemmed_list = [self.snowball.stem(word) for word in text_list]
        else:
            raise ValueError("Invalid method. Use 'porter', 'lancaster' or 'snowball'.")
        return stemmed_list

    def remove_stopwords(self, text_list, stopwords_path):
        with open(stopwords_path, 'r', encoding="utf-8") as file:
            stopwords = file.read()
        stopwords = stopwords.lower()
        stopwords = re.split(r'[, \n]+', stopwords)
        stopwords = self.remove_accents(stopwords)
        text_without_stopwords = []
        
        for word in text_list:
            if word not in stopwords:
                text_without_stopwords.append(word)
        return text_without_stopwords

    def tokenize(self, text, html_tags: bool = False, stopwords: bool = False, stopwords_path: str = None, stemming: bool = False, stemming_method: str = "porter"):
        text_list = self.regex.findall(text)
        text_list = self.remove_accents(text_list)
        if html_tags:
            text_list = self.remove_html_tags(text_list)
        processed_tokens = []
        for token in text_list:
            # Aplicar minúscula solo a palabras normales
            if re.fullmatch(r'[a-zA-Z]+(?:-[a-zA-Z]+)*', token):
                token = token.lower()
            processed_tokens.append(token)
        if stopwords:
            processed_tokens = self.remove_stopwords(processed_tokens, stopwords_path)
        if stemming:
            processed_tokens = self.stemming(processed_tokens, stemming_method)
        return processed_tokens