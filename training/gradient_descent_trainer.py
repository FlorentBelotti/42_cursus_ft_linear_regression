from model.linear_regression_model import LinearRegressionModel


class GradientDescentTrainer:
    def __init__(self, learning_rate: float, iterations: int) -> None:
        self._learning_rate = learning_rate
        self._iterations = iterations

    def train(
        self,
        model: LinearRegressionModel,
        mileages: list[float],
        prices: list[float],
    ) -> None:
        m = len(mileages)
        for _ in range(self._iterations):
            errors = [model.predict(mileages[i]) - prices[i] for i in range(m)]
            tmp_theta0 = self._learning_rate * sum(errors) / m
            tmp_theta1 = (
                self._learning_rate
                * sum(errors[i] * mileages[i] for i in range(m))
                / m
            )
            model.update_parameters(
                model.theta0 - tmp_theta0,
                model.theta1 - tmp_theta1,
            )
