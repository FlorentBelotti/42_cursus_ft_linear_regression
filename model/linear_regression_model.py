class LinearRegressionModel:
    def __init__(self, theta0: float = 0.0, theta1: float = 0.0) -> None:
        self._theta0 = theta0
        self._theta1 = theta1

    @property
    def theta0(self) -> float:
        return self._theta0

    @property
    def theta1(self) -> float:
        return self._theta1

    def predict(self, mileage: float) -> float:
        return self._theta0 + self._theta1 * mileage

    def update_parameters(self, theta0: float, theta1: float) -> None:
        self._theta0 = theta0
        self._theta1 = theta1
