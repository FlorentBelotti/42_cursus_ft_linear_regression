import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.linear_regression_model import LinearRegressionModel
from model.model_parameter_storage import ModelParameterStorage

_PARAMETERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_parameters.json")


def main() -> None:
    storage = ModelParameterStorage()
    theta0, theta1 = storage.load(_PARAMETERS_FILE)
    model = LinearRegressionModel(theta0, theta1)

    try:
        raw = input("Enter mileage (km): ")
        mileage = float(raw)
    except ValueError:
        print("Error: please enter a valid number.", file=sys.stderr)
        sys.exit(1)

    price = model.predict(mileage)
    print(f"Estimated price: {price:.2f}")


if __name__ == "__main__":
    main()
