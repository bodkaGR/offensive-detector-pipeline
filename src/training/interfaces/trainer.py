from abc import ABC, abstractmethod

from torch.utils.data import DataLoader


class ITrainer(ABC):

    @abstractmethod
    def fit(self, epochs: int) -> dict: ...

    @abstractmethod
    def evaluate(self, loader: DataLoader) -> dict: ...