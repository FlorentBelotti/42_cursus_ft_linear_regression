class MinMaxFeatureNormalizer:
    def __init__(self, values: list[float]) -> None:
        self._min = min(values)
        self._max = max(values)

    @property
    def min_val(self) -> float:
        return self._min

    @property
    def max_val(self) -> float:
        return self._max

    def normalize(self, values: list[float]) -> list[float]:
        return [self.normalize_value(v) for v in values]

    def normalize_value(self, value: float) -> float:
        return (value - self._min) / (self._max - self._min)

    def denormalize_value(self, value: float) -> float:
        return value * (self._max - self._min) + self._min

    def denormalize_theta_parameters(
        self,
        theta0_norm: float,
        theta1_norm: float,
        price_normalizer: "MinMaxFeatureNormalizer",
    ) -> tuple[float, float]:
        # In normalized space: y_norm = theta0_norm + theta1_norm * x_norm
        # Expanding: y = y_min + (y_max-y_min)*theta0_norm
        #                + (y_max-y_min)*theta1_norm/(x_max-x_min) * x
        #                - (y_max-y_min)*theta1_norm*x_min/(x_max-x_min)
        price_range = price_normalizer.max_val - price_normalizer.min_val
        mileage_range = self._max - self._min

        theta1 = theta1_norm * price_range / mileage_range
        theta0 = (
            price_normalizer.min_val
            + theta0_norm * price_range
            - theta1 * self._min
        )
        return theta0, theta1
