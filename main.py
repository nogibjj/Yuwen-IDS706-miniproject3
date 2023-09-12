import polars as pl

def create_dataframe():
    data = {
        'age': [25, 30, 35, 40, 45],
        'height': [165, 170, 175, 180, 185],
        'weight': [60, 70, 80, 90, 100]
    }
    return pl.DataFrame(data)

def calculate_mean_weight(dataframe):
    return dataframe['weight'].mean()

def calculate_sum_weight(dataframe):
    return dataframe['weight'].sum()

def calculate_correlation_matrix(dataframe):
    return dataframe[['age', 'height', 'weight']].corr()
