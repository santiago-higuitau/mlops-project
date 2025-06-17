
import sys
import pandas as pd
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{message}")


if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def read_dataframe(**kwargs) -> pd.DataFrame:
    """
    Template for loading data from API
    """
    if all(k in kwargs for k in ('year', 'month', 'dataset')):
        year = kwargs.get('year')
        month = kwargs.get('month')
        dataset = kwargs.get('dataset', 'green')
    else:
        exe_date = kwargs.get('execution_date').date()
        logger.info(f'Year and month not found in kwargs, using execution date -> {exe_date}')

        year = exe_date.year
        month = exe_date.month - 3  # last month available in the repo
        dataset = 'green'

    logger.info(f'Input data used to download dataset from NYC trip -> {year}, {month:02d}, {dataset}')

    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{dataset}_tripdata_{year}-{month:02d}.parquet'
    try:
        df = pd.read_parquet(url)
        logger.info(f'Raw data dimensions: {df.shape}')

        return df
    except Exception as e:
        logger.info(f'⚠️ Month {month} not available to download: {str(e)}')
        return pd.DataFrame()


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
    assert isinstance(output, pd.DataFrame), 'Output is not a DataFrame'

    if output.empty:
        logger.info('⚠️ Warning: Dataframe is empty. Month not available to download.')
    else:
        assert output.shape[1] > 0, 'DataFrame has no columnas'
