import json
import os


class ModelParameterStorage:
    def save(self, filepath: str, theta0: float, theta1: float) -> None:
        with open(filepath, "w") as f:
            json.dump({"theta0": theta0, "theta1": theta1}, f)

    def load(self, filepath: str) -> tuple[float, float]:
        if not os.path.exists(filepath):
            return 0.0, 0.0
        with open(filepath) as f:
            data = json.load(f)
        return float(data["theta0"]), float(data["theta1"])
