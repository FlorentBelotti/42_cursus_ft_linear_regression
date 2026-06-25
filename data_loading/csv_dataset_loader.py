class CsvDatasetLoader:
    def __init__(self, filepath: str) -> None:
        self._filepath = filepath

    def load(self) -> tuple[list[float], list[float]]:
        mileages: list[float] = []
        prices: list[float] = []
        with open(self._filepath, newline="") as f:
            lines = f.read().splitlines()
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) != 2:
                continue
            mileages.append(float(parts[0].strip()))
            prices.append(float(parts[1].strip()))
        return mileages, prices
