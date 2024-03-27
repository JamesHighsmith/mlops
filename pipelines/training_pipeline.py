from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluation

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@pipeline(enable_cache=False)
def train_pipeline():
    df = ingest_data()
    x_train, x_test, y_train, y_test = clean_data(data=df)
    model = train_model(x_train, x_test, y_train, y_test)
    r2_score, rmse = evaluation(model, x_test, y_test)
    return r2_score, rmse
