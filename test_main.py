import polars as pl
from main import create_dataframe, calculate_mean_weight, calculate_sum_weight, calculate_correlation_matrix

def sample_dataframe():
    data = {
        'age': [25, 30, 35, 40, 45],
        'height': [165, 170, 175, 180, 185],
        'weight': [60, 70, 80, 90, 100]
    }
    return pl.DataFrame(data)

def test_create_dataframe():
    dataframe = create_dataframe()
    assert len(dataframe) == 5

def test_calculate_mean_weight(sample_dataframe):
    mean_weight = calculate_mean_weight(sample_dataframe)
    assert mean_weight == 80.0

def test_calculate_sum_weight(sample_dataframe):
    sum_weight = calculate_sum_weight(sample_dataframe)
    assert sum_weight == 400

def test_calculate_correlation_matrix(sample_dataframe):
    correlation_matrix = calculate_correlation_matrix(sample_dataframe)
    assert correlation_matrix.to_pandas().equals(
        sample_dataframe[['age', 'height', 'weight']].corr().to_pandas()
    )