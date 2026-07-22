# FUNCTION.md

Chronological walkthrough of every file and function, in the order they are called at runtime.

---

## 1. `train_linear_regression_model.py` — Training entry point

### `main()`
Orchestrates the entire training pipeline: loads data, normalises features, runs gradient descent, converts the learned parameters back to the original scale, and writes them to disk.
It is the only place where all modules are wired together; every other file is a focused, single-responsibility component called from here.

---

## 2. `data_loading/csv_dataset_loader.py`

### `__init__(filepath)`
Stores the path to the CSV file that will be read later.
No I/O happens here; the object is just configured and ready.

### `load()`
Opens the CSV, skips the header line, and splits each row into a mileage value and a price value.
Returns two parallel lists of floats — `mileages` and `prices` — that feed the rest of the pipeline.

---

## 3. `normalization/min_max_feature_normalizer.py`

### `__init__(values)`
Scans the list of raw values once to record their minimum and maximum.
These two boundaries define the scaling formula used by every subsequent method.

### `min_val` / `max_val` *(properties)*
Expose the recorded boundaries as read-only values.
Used by `denormalize_theta_parameters` when converting thetas from the price normalizer's scale.

### `normalize(values)`
Applies `normalize_value` to every element of a list and returns the rescaled list.
Called once for mileages and once for prices before training begins.

### `normalize_value(value)`
Maps a single raw value to the [0, 1] range using `(x − min) / (max − min)`.
The core scaling formula; called by `normalize` and available for one-off values.

### `denormalize_value(value)`
Reverses the scaling of a single normalised value back to the original unit.
Available for future use (e.g. converting a normalised prediction back to euros).

### `denormalize_theta_parameters(theta0_norm, theta1_norm, price_normalizer)`
Algebraically converts θ0 and θ1 learned in normalised space back to the original km / euro scale.
This is critical: without it the saved parameters would only work on [0, 1] inputs instead of real mileages, making the predictor useless.

---

## 4. `model/linear_regression_model.py`

### `__init__(theta0, theta1)`
Initialises the model with a pair of parameters, defaulting to (0.0, 0.0) before any training.
Starting at zero means the model predicts a price of 0 for every mileage until training runs.

### `theta0` / `theta1` *(properties)*
Expose the current parameter values as read-only scalars.
Used by the trainer at each iteration to compute the gradient without risking accidental mutation.

### `predict(mileage)`
Computes `θ0 + θ1 × mileage` — the hypothesis function defined by the subject.
Used both during training (to measure the error at each step) and by the predictor at inference time.

### `update_parameters(theta0, theta1)`
Replaces the stored θ values with new ones after one gradient descent step.
The trainer calls this once per iteration to make the simultaneous update required by the algorithm.

---

## 5. `model/model_parameter_storage.py`

### `save(filepath, theta0, theta1)`
Serialises θ0 and θ1 to a JSON file at the given path.
Persists the result of training so the predictor can load it in a completely separate process.

### `load(filepath)`
Reads θ0 and θ1 from a JSON file, or returns (0.0, 0.0) if the file does not exist yet.
Guarantees that the predictor can run safely before training without crashing.

---

## 6. `training/gradient_descent_trainer.py`

### `__init__(learning_rate, iterations)`
Stores the two hyper-parameters that control the training loop: step size and number of passes.
Chosen values (lr = 0.1, 1000 iterations) are calibrated for normalised input in the [0, 1] range.

### `train(model, mileages, prices)`
Runs the gradient descent loop for the configured number of iterations on normalised data.
At each step it computes the prediction errors, derives the two partial gradients defined by the subject's formulas, and calls `model.update_parameters` for a simultaneous update of both θ values.

---

## 7. `predict_car_price.py` — Prediction entry point

### `main()`
Loads the saved θ values from disk, builds a model, prompts the user for a mileage, and prints the estimated price.
If no parameters file exists the model defaults to (0.0, 0.0) and prints 0.00; if the input is not a valid number it exits with a clear error message.
