from pandas import read_csv, DataFrame


def load_data_file(filename: str) -> DataFrame:
    data: DataFrame = read_csv(filename)
    return data
