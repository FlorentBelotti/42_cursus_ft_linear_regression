import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loading.csv_dataset_loader import CsvDatasetLoader
from model.linear_regression_model import LinearRegressionModel
from model.model_parameter_storage import ModelParameterStorage
from normalization.min_max_feature_normalizer import MinMaxFeatureNormalizer
from training.gradient_descent_trainer import GradientDescentTrainer

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATASET = os.path.join(_PROJECT_DIR, "data.csv")
_PARAMETERS_FILE = os.path.join(_PROJECT_DIR, "model_parameters.json")

_LEARNING_RATE = 0.1
_ITERATIONS = 1000


def main() -> None:
    dataset_file = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_DATASET

    loader = CsvDatasetLoader(dataset_file)
    mileages, prices = loader.load()

    mileage_normalizer = MinMaxFeatureNormalizer(mileages)
    price_normalizer = MinMaxFeatureNormalizer(prices)

    mileages_norm = mileage_normalizer.normalize(mileages)
    prices_norm = price_normalizer.normalize(prices)

    model = LinearRegressionModel()
    trainer = GradientDescentTrainer(_LEARNING_RATE, _ITERATIONS)
    trainer.train(model, mileages_norm, prices_norm)

    theta0, theta1 = mileage_normalizer.denormalize_theta_parameters(
        model.theta0, model.theta1, price_normalizer
    )

    storage = ModelParameterStorage()
    storage.save(_PARAMETERS_FILE, theta0, theta1)

    print(f"Training complete ({_ITERATIONS} iterations, lr={_LEARNING_RATE})")
    print(f"θ0 = {theta0:.6f}")
    print(f"θ1 = {theta1:.10f}")


if __name__ == "__main__":
    main()
