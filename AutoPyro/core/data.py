from io import BytesIO
import pandas as pd

class DataTable:
    
    # __slots__ = ("table")
    
    def __init__(self, file: BytesIO) -> None:
        self.table = self.load_data(file)

    @staticmethod
    def load_data(file: BytesIO):
        if file == ".xlsx":
            df = pd.read_excel()
        elif file == ".csv":
            df = pd.read_csv()
        else:
            raise FileNotFoundError("Bruh")
        
        return df
    
    def impute(self, X: list[str], y: str):
        cols = self.table.columns
        if X not in cols or y not in cols:
            raise 
        
    