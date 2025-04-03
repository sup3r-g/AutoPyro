from io import BytesIO, StringIO

# import holoviews as hv
# import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn
import param

import sys
import os

# module_path = os.path.abspath("../../AutoPyro")
# if module_path not in sys.path:
#     sys.path.append(module_path)
# from core.data import DataTable

pn.extension("plotly", "tabulator", sizing_mode="stretch_width")

tabulator_editors = {
    "int": {"type": "number", "step": 1},
    "float": {"type": "number"},
    "bool": {"type": "tickCross", "indeterminateValue": None},
    "str": {"type": "list", "valuesLookup": True},
    "date": "date",
    "datetime": "datetime",
}


class DataTableUI(param.Parameterized):
    
    DEFAULT_PARAMS = {
        "learning_rate": 0.5,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "task_type": "CPU",
        "iterations": 5000,
        "verbose": False,
    }
    PATTERN = r", | \("

    file_input = param.Parameter()
    dataframe = param.DataFrame()

    features = pn.widgets.MultiChoice(name="Признаковые значения", solid=False)
    target = pn.widgets.AutocompleteInput(name="Целевое значение", placeholder="TOC")
    column_stats = pn.widgets.MultiChoice(name="Колонки статистики", solid=False)
    # impute = param.Action(label="Запустить модель")

    file_name = param.String(default="data.csv", label="Название файла с расширением")

    def __init__(self, **params):
        super().__init__(file_input=pn.widgets.FileInput(accept=".csv,.json"), **params)
        self.table = pn.widgets.Tabulator(
            self.dataframe,
            layout="fit_data_table",
            theme="materialize",
            editors=tabulator_editors,
            embed_content=True,
            header_filters=True,
            pagination="remote",
            page_size=50,
        )
        self.download = pn.widgets.FileDownload(
            name="",
            filename=self.file_name,
            callback=self._download_callback,
            button_type="primary",
        )

    @pn.depends("file_name", watch=True)
    def _update_filename(self):
        self.download.filename = self.file_name

    @pn.depends("file_input.value", watch=True)
    def _parse_file_input(self):
        value = self.file_input.value
        if value:
            string_io = StringIO(value.decode("utf-8"))
            df = pd.read_csv(string_io, delimiter=";")
            df = self._clean_column_names(df)
            columns = df.columns.to_list()
            self.features.options = columns
            self.target.options = columns
            self.column_stats.options = columns
            self.dataframe = df   

    @pn.depends("dataframe", watch=True)
    def _update_table(self):
        if hasattr(self, "table"):
            self.table.value = self.dataframe

    @staticmethod
    def _clean_column_names(df) -> pd.DataFrame:
        df.columns = df.columns.str.split(pat=DataTableUI.PATTERN, expand=False).map(
            lambda split_list: split_list[0]
        )

        return df

    def _download_callback(self) -> BytesIO | StringIO:
        self.download.filename = self.file_name

        data_format = self.file_name.split(".")[1]
        if data_format == "xlsx":
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine="xlsxwriter")
            self.dataframe.to_excel(writer, sheet_name="Data")
            writer.save()
        if data_format == "csv":
            output = StringIO()
            self.dataframe.to_csv(output, index=False)
        if data_format == "json":
            output = BytesIO()
            self.dataframe.to_json(output, orient="records")

        output.seek(0)
        return output

    def statistics(self) -> dict[str, dict]:
        value = self.column_stats.value
        if value:
            return self.dataframe[value].describe()

    # @pn.depends("impute", watch=True)
    # def impute(self):
    #     # DataTable.impute(self.features.value, self.target.value)
    #     pass

    @pn.depends("column_stats.value")
    def view_stats(self):
        return pn.widgets.Tabulator(
            self.statistics(),
            layout="fit_data_table",
            theme="materialize",
            embed_content=True,
        )

    def view(self):
        return pn.Column(
            "# Таблицы",
            self.file_input,
            self.table,
            # pn.layout.VSpacer(),
            pn.Tabs(
                ("Восстановление значений", pn.Row(self.features, self.target)),
                ("Статистика", pn.Column(self.column_stats, self.view_stats)),
                (
                    "Скачивание готовых данных",
                    pn.Column(self.param.file_name, self.download),
                ),
                dynamic=True,
            ),
            align="end",
        )

# pn.Param(DataTableUI.param, widgets={
#     'features': pn.widgets.MultiChoice,
#     'target': pn.widgets.AutocompleteInput}
# )

TABLE_UI = DataTableUI().view()

if __name__ == "__main__":
    pn.Row(DataTableUI().view).show()

# pn.template.MaterialTemplate(
#     title="Таблица данных",
#     logo="img/table.svg",
#     favicon="img/table.svg",
#     main=[
#         description,
#         sample_data_app_view,
#     ],
# ).servable()
