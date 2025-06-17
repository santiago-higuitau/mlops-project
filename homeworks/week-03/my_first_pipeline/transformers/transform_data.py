
import sys
from loguru import logger
from pandas import DataFrame

logger.remove()
logger.add(sys.stdout, level="INFO", format="{message}")


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def execute_transformer_action(df: DataFrame, **kwargs) -> DataFrame:
    """
    Execute Transformer Action
    """
    dataset = kwargs.get('dataset', 'green')

    if dataset == 'green':
        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    else:
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df_filtered = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    categorical = ['PULocationID', 'DOLocationID']
    df_filtered[categorical] = df_filtered[categorical].astype(str)

    # df_filtered['PU_DO'] = df_filtered['PULocationID'] + '_' + df_filtered['DOLocationID']
    logger.info(f'Processed data dimensions: {df_filtered.shape}')
    
    return df_filtered


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
    assert isinstance(output, DataFrame), 'Output is not a DataFrame'
    assert not output.empty, 'Transformed DataFrame is empty'

    expected_cols = {'duration', 'PULocationID', 'DOLocationID'}
    missing = expected_cols - set(output.columns)
    assert not missing, f'Missing expected columns: {missing}'

    assert output['duration'].between(1, 60).all(), 'Some duration values are out of range [1, 60]'
    assert output['PULocationID'].dtype == object, 'PULocationID is not string'
    assert output['DOLocationID'].dtype == object, 'DOLocationID is not string'
