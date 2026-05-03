import re
import string

import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from typing_extensions import override

from src.data.interfaces.preprocessor import ITextPreprocessor


for res in ["wordnet", "omw-1.4", "stopwords"]:
    try:
        nltk.data.find(f"corpora/{res}")
    except LookupError:
        nltk.download(res, quiet=True)


class TwitterTextPreprocessor(ITextPreprocessor):

    CONTRACTIONS: dict[str, str] = {
        "can't": "cannot", "won't": "will not", "n't": " not",
        "'re": " are", "'ve": " have", "'ll": " will",
        "'d": " would", "'m": " am", "it's": "it is",
    }
    LEETSPEAK: dict[str, str] = {
        "@": "a", "3": "e", "1": "i",
        "0": "o", "5": "s", "7": "t", "$": "s",
    }

    def __init__(
        self,
        remove_stopwords: bool = False,
        lemmatize: bool = True
    ):
        self._remove_stopwords = remove_stopwords
        self._lemmatize = lemmatize
        self._lemmatizer = WordNetLemmatizer()
        self._stop_words = set(stopwords.words("english"))

    def _remove_urls(self, text: str) -> str:
        return re.sub(r"https?://\S+|www\.\S+", " ", text)

    def _remove_mentions_and_hashtags(self, text: str) -> str:
        text = re.sub(r"@\w+", " ", text)
        text = re.sub(r"#(\w+)", r"\1", text)
        return text

    def _expand_contractions(self, text: str) -> str:
        for contraction, expansion in self.CONTRACTIONS.items():
            text = text.replace(contraction, expansion)
        return text

    def _deobfuscate_leetspeak(self, text: str) -> str:
        for leet, char in self.LEETSPEAK.items():
            text = text.replace(leet, char)
        return text

    def _normalize_repeated_chars(self, text: str) -> str:
        return re.sub(r"(.)\1{2,}", r"\1\1", text)

    def _remove_punctuation(self, text: str) -> str:
        punct = string.punctuation.replace("'", "")
        return text.translate(str.maketrans("", "", punct))

    def _remove_numbers(self, text: str) -> str:
        return re.sub(r"\d+", " ", text)

    def _tokenize_and_normalize(self, text: str) -> str:
        tokens = text.split()
        if self._remove_stopwords:
            tokens = [t for t in tokens if t not in self._stop_words]
        if self._lemmatize:
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)

    @override
    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = self._remove_urls(text)
        text = self._remove_mentions_and_hashtags(text)
        text = self._expand_contractions(text)
        text = self._deobfuscate_leetspeak(text)
        text = self._normalize_repeated_chars(text)
        text = self._remove_punctuation(text)
        text = self._remove_numbers(text)
        text = " ".join(text.split())
        text = self._tokenize_and_normalize(text)
        return text.strip()

    @override
    def clean_batch(self, texts: list[str]) -> list[str]:
        return [self.clean(text) for text in texts]