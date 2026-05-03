from __future__ import annotations

from abc import abstractmethod, ABC


class ITextPreprocessor(ABC):

    @abstractmethod
    def clean(self, text: str) -> str: ...

    @abstractmethod
    def clean_batch(self, texts: list[str]) -> list[str]: ...