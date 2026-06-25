# ft_linear_regression

An introduction to machine learning — predict car prices from mileage using gradient descent.

---

## What is linear regression?

Linear regression is one of the simplest supervised learning algorithms. Given a set of observations, it tries to fit a straight line through the data so that the line can be used to predict an output value (here: price) from an input value (here: mileage).

The line is described by two parameters:

```
estimatedPrice(mileage) = θ0 + θ1 × mileage
```

- **θ0** (theta0) is the intercept — the predicted price when mileage is 0.
- **θ1** (theta1) is the slope — how much the price changes per additional kilometre.

The algorithm's goal is to find the values of θ0 and θ1 that minimise the **mean squared error** (MSE) between the predictions and the actual prices in the training dataset:

```
MSE = (1/m) × Σ (estimatedPrice(mileage[i]) − price[i])²
```

where *m* is the number of data points.

---

## How it works

### Training with gradient descent

Rather than solving the optimal θ values analytically, this project uses **gradient descent**: an iterative optimisation algorithm that repeatedly nudges θ0 and θ1 in the direction that reduces the cost.

At each iteration, both parameters are updated **simultaneously** using the partial derivatives of the MSE:

```
tmpθ0 = learningRate × (1/m) × Σ (estimatedPrice(mileage[i]) − price[i])
tmpθ1 = learningRate × (1/m) × Σ (estimatedPrice(mileage[i]) − price[i]) × mileage[i]

θ0 = θ0 − tmpθ0
θ1 = θ1 − tmpθ1
```

The **learning rate** controls the step size at each iteration. A value that is too large causes the algorithm to overshoot and diverge; a value that is too small makes convergence very slow.

### Feature normalisation

Mileages (tens of thousands of km) and prices (thousands of euros) live on very different numerical scales. Feeding raw values into gradient descent would make convergence impractically slow. Before training, both features are rescaled to the [0, 1] range using **min-max normalisation**:

```
x_norm = (x − x_min) / (x_max − x_min)
```

After training the thetas are algebraically converted back to the original scale so that the predictor can work directly with raw km values.

### Project structure

```
ft_linear_regression/
├── data.csv                          # Training dataset (mileage, price)
├── train_linear_regression_model.py  # Training entry point
├── predict_car_price.py              # Prediction entry point
├── model_parameters.json             # Saved θ0 and θ1 (created after training)
├── data_loading/
│   └── csv_dataset_loader.py         # Reads the CSV dataset
├── normalization/
│   └── min_max_feature_normalizer.py # Min-max scaling + theta denormalisation
├── model/
│   ├── linear_regression_model.py    # Hypothesis function
│   └── model_parameter_storage.py    # Persist / load θ values (JSON)
└── training/
    └── gradient_descent_trainer.py   # Gradient descent loop
```

---

## Requirements

- Python 3.9 or later
- No third-party dependencies (standard library only)

---

## Usage

### 1. Train the model

Run the training script against the provided dataset:

```bash
python3 train_linear_regression_model.py
```

You can also pass a custom dataset path as the first argument:

```bash
python3 train_linear_regression_model.py path/to/data.csv
```

The script will print the learned parameters and write them to `model_parameters.json`:

```
Training complete (1000 iterations, lr=0.1)
θ0 = 8481.172797
θ1 = -0.0212737191
```

### 2. Predict a price

```bash
python3 predict_car_price.py
```

You will be prompted for a mileage:

```
Enter mileage (km): 100000
Estimated price: 6353.80
```

If the model has not been trained yet, θ0 and θ1 default to 0 and the estimated price will be 0.00.
