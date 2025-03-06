from io import BytesIO
from pathlib import Path
from typing import Literal, Self, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import train_test_split

# from catboost import CatBoostRegressor, Pool
from AutoPyro.core.calculators import CALCULATORS_MAP
from AutoPyro.core.classifiers import CLASSIFIERS_MAP


TRANSLATION_MAP = {
    "count": "Общее количество",
    "unique": "Количество уникальных",
    "top": "Мода",
    "freq": "Частота моды",
    "mean": "Среднее",
    "50%": "Медиана (50%)",
    "std": "Стандартное отклонение",
    "min": "Минимум",
    "max": "Максимум",
    "first": "Первый",
    "last": "Последний",
}


# class ModelRegister(dict):

#     DEFAULT_PARAMS = {
#         "learning_rate": 0.05,
#         "loss_function": "RMSE",
#         "eval_metric": "RMSE",
#         "task_type": "CPU",
#         "iterations": 2000,
#         "verbose": False,
#     }

#     def keys_of(self, value):
#         for k, v in self.items():
#             if v == value:
#                 yield k

#     def add(self, name: str, model: CatBoostRegressor, overwrite: bool = False) -> None:
#         if name in self and not overwrite:
#             raise KeyError(
#                 f"{name} is already in use. To overwrite model in the registry use overwrite=True."
#             )

#         if overwrite:
#             print("Overwrite enabled.")

#         self[name] = model

#     def delete(self, name: str) -> None:
#         if name not in self:
#             raise KeyError(f"Can't delete. Model '{name}' is not in use.")

#         del self[name]


class DataTable:
    __slots__ = "table", "__queries", "register"
    PATTERN = r", | \("

    def __init__(self, table: pd.DataFrame) -> None:
        self.table = table
        self.__queries = []
        # self.register = ModelRegister()

    def __getattr__(self, attr):
        return getattr(self.table, attr)

    @property
    def queries(self) -> list[Any]:
        return self.__queries

    @classmethod
    def from_file(cls, file: BytesIO, **pandas_kwargs) -> Self:
        filepath = Path(file)
        ext = filepath.suffix

        if ext == ".xlsx":
            data = pd.read_excel(file, **pandas_kwargs)
        elif ext == ".csv":
            data = pd.read_csv(file, **pandas_kwargs)
        else:
            raise FileNotFoundError("Unknown extension type encountered")

        return cls(table=cls._clean_column_names(data))

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame) -> Self:
        return cls(table=cls._clean_column_names(dataframe))

    @staticmethod
    def _clean_column_names(table: pd.DataFrame) -> pd.DataFrame:
        table.columns = table.columns.str.split(
            pat=DataTable.PATTERN, expand=False
        ).map(lambda split_list: split_list[0])

        return table

    def select(self, expression: str, overwrite: bool = False) -> None | pd.DataFrame:
        if overwrite:
            self.table = self.table.query(expression)
        else:
            self.queries.append(expression)
            return self.table.query(expression)

    # def impute(
    #     self,
    #     features: list[str],
    #     target: str,
    #     stratify: Union[str, None] = None,
    #     save_model: bool = False,
    #     **catboost_kwargs,
    # ) -> npt.NDArray[np.float64]:
    #     test_size = catboost_kwargs.pop("test_size", 0.2)

    #     cols = self.table.columns
    #     if features not in cols or target not in cols:
    #         raise ValueError(
    #             "Provided 'features' or 'target' is not present in the table columns"
    #         )

    #     X_train, X_test, y_train, y_test = train_test_split(
    #         self.data[features],
    #         self.data[target],
    #         test_size=test_size,
    #         shuffle=True,
    #         stratify=stratify,
    #     )
    #     train_pool = Pool(X_train, y_train)
    #     test_pool = Pool(X_test, y_test)

    #     model = CatBoostRegressor(self.DEFAULT_PARAMS)
    #     model.fit(train_pool, eval_set=test_pool)

    #     y_pred = model.predict(X_test)

    #     if save_model:
    #         self.register.add(name=target, model=model)

    #     return y_pred

    def get_statistics(
        self, *columns, group: Union[str, None] = None, as_dict: bool = False
    ) -> dict[Hashable, Any] | pd.DataFrame:
        columns = list(columns) if columns else self.table.columns

        if group is not None:
            table = self.table[columns + [group]].groupby("Structural element Russian")
        else:
            table = self.table[columns]

        stats = table.describe(include="all").rename(columns=TRANSLATION_MAP)

        if as_dict:
            return stats.to_dict()

        return stats

    def get_histogram(
        self,
        column: str,
        bins: int | str = "auto",
        **hist_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.histogram(self.table[column], bins=bins, **hist_kwargs)

    def plot_histogram(
        self, column: str, color, bins: int | str = "auto", **hist_kwargs
    ):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(column, fontweight="bold", fontsize=16)
        ax.set_axisbelow(True)
        ax.set_xlabel("Значение", fontweight="bold", fontsize=14)
        ax.set_ylabel("Частота", fontweight="bold", fontsize=14)
        ax.tick_params(axis="both", labelsize=14)
        self.table[column].hist(
            bins=bins, ax=ax, facecolor=color, edgecolor="k", **hist_kwargs
        )

        return fig, ax

    def classify(
        self,
        x_name: str,
        y_name: str,
        author: str,
        mode: Literal["single", "multi"] = "single",
        parameter: Literal["matter", "maturity", "potential"] = "matter",
    ) -> None:
        if x_name not in self.table.columns or y_name not in self.table.columns:
            raise ValueError(
                "Provided 'x' or 'y' names are not present in the table columns"
            )

        classifier = CLASSIFIERS_MAP.get(parameter, None)
        if classifier is None:
            raise NotImplementedError(
                f"Classification parameter '{parameter}' is not available yet"
            )

        X, Y = self.table[x_name], self.table[y_name]
        if mode == "single":
            result = classifier.classify_single(X, Y, author)
        elif mode == "multi":
            result = classifier.classify_multi(X, Y, author)
        else:
            raise ValueError("Invalid 'mode' value")

        self.table[classifier.column_name] = result

    def calculate(self, target="HIo", option="plot", **calculator_kwargs) -> None:
        calculator = CALCULATORS_MAP.get(target, None)
        if calculator is None:
            raise NotImplementedError(
                f"Calculation target '{target}' is not available yet"
            )

        option = option.replace(", ", "_")
        if not hasattr(calculator, option):
            raise AttributeError(f"Invalid calculation 'option' {option} supplied")

        self.table[calculator.column_name] = getattr(calculator, option)(
            **calculator_kwargs
        )
