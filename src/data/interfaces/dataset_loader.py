from abc import abstractmethod, ABC


class IDatasetLoader(ABC):

    @abstractmethod
    def load(self, path: str) -> dict: ...