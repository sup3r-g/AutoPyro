from typing import Union


class BaseClassifier:
    def classify_single(author: str) -> float:
        pass

    def classify_multi(return_all=False) -> Union[float, dict[str, float]]:
        pass


class MatterTypeClassifier(BaseClassifier):
    def classify_single(author: str) -> float:
        pass

    def classify_multi(return_all=False) -> Union[float, dict[str, float]]:
        pass


class MaturityClassifier(BaseClassifier):
    pass
