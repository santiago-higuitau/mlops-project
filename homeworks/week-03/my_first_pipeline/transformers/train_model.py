
import os
import sys
import pickle
import mlflow
import mlflow.tracking
from loguru import logger
from pandas import DataFrame

from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer


logger.remove()
logger.add(sys.stdout, level="INFO", format="{message}")


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


def create_X(dataset: DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    records = dataset[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    X = dv.fit_transform(records)

    return X, dv


def register_model(run_id: str):
    mlflow.register_model(
        model_uri=f"runs:/{run_id}/models",
        name='linear-regressor-mage-ai'
    )


def get_model_size_bytes(run_id: str, artifact_path: str):
    try:
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id, path=artifact_path)

        for artifact in artifacts:
            if artifact.file_size and artifact.path == f'{artifact_path}/model.pkl':
                size_bytes = artifact.file_size

        return size_bytes
    except KeyError:
        raise ValueError(f"'model_size_bytes' not found in flavor metadata at {artifact_path}")


@transformer
def train_model(df: DataFrame, **kwargs):

    target = 'duration'

    X_train, dv = create_X(df)
    y_train = df[target].values


    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("nyc-taxi-experiment-mage-ai")

    with mlflow.start_run() as run:

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        logger.info(f"Intercept: {lr.intercept_}")

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("vectorizer_features", dv.feature_names_)
        mlflow.log_param("intercept_", lr.intercept_)


        os.makedirs("models", exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.sklearn.log_model(lr, artifact_path="regression_model")

        run_id = run.info.run_id
        logger.info(f'Run id -> {run_id}')
        register_model(run_id)

    model_size_bytes = get_model_size_bytes(run_id, "regression_model")
    logger.info(f'Model size bytes -> {model_size_bytes}')

    return run_id


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
