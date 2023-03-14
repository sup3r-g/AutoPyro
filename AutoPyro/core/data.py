from dataclasses import dataclass
from io import BytesIO
from typing import ClassVar

import pandas as pd
from calculators import *
from catboost import CatBoostRegressor, Pool
from plots import Plot
from sklearn.model_selection import train_test_split


@dataclass(slots=True)
class DataTable:
    DEFAULT_PARAMS: ClassVar[dict] = {
        "learning_rate": 0.5,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "task_type": "CPU",
        "iterations": 5000,
        "verbose": False,
    }

    table: pd.DataFrame
    models: dict[str, CatBoostRegressor]

    @classmethod
    def load_data(file: BytesIO, **pandas_kwargs):
        if file == ".xlsx":
            data = pd.read_excel(file, **pandas_kwargs)
        elif file == ".csv":
            data = pd.read_csv(file, **pandas_kwargs)
        else:
            raise FileNotFoundError("Bruh")

        return data

    def impute(self, features: list[str], target: str, stratify: str | None = None, **catboost_kwargs):
        test_size = catboost_kwargs.pop("test_size", 0.2)
        
        cols = self.table.columns
        if features not in cols or target not in cols:
            raise ValueError("Provided features or target is not present in the table")

        X_train, X_test, y_train, y_test = train_test_split(
            self.data[features],
            self.data[target],
            test_size=test_size,
            shuffle=True,
            stratify=stratify  # self.data["obj_name"]
        )
        train_pool = Pool(X_train, y_train)
        test_pool = Pool(X_test, y_test)

        model = CatBoostRegressor(self.DEFAULT_PARAMS)
        model.fit(train_pool, eval_set=test_pool)

        y_pred = model.predict(X_test)
        
        return y_pred
    
    def classify(self):

        return Plot().classify()
    
    def calculate(self, target="HIo", method="Fit") -> None:
        if target == "HIo":
            calc = HIo()
        elif target == "TOCo":
            calc = TOCo()
        elif target == "TOCo":
            calc = TR()
        else:
            raise KeyError("Calculator not implemented yet")

        self.table[calc.column_name] = calc.Cornford_2001()
        
        
