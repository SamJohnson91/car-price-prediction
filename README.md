# Car price prediction

Predict the price of a used car from its mileage, age, engine and brand.
First machine learning project: linear regression, then random forest.

## Data
`data/used_cars.csv`: 4,009 used cars (brand, model year, mileage, fuel, engine, transmission, accident history, price).
Source: Kaggle (download as `archive.zip`, unzip into `data/`).

## Run
```
pip install pandas scikit-learn matplotlib
python car_price.py
```
Results and plots are saved in `results/`.

## Steps
1. Load the CSV
2. Clean: turn `"$10,300"` and `"51,000 mi."` into numbers, pull horsepower and engine size out of the engine text, remove the 1% most expensive cars
3. Choose features and target (`price`)
4. Split 80% train / 20% test
5. Prepare: fill missing numbers with the median, one-hot encode brand and fuel type
6. Train 4 models
7. Evaluate with MAE and R2 on the test set

## Results (test set, 794 cars)
| Model | MAE ($) | R2 |
|---|---|---|
| Baseline (average price) | 23,277 | 0.00 |
| Linear regression | 15,187 | 0.48 |
| Linear regression (log price) | 10,402 | 0.58 |
| Random forest (log price) | 8,517 | 0.73 |

Most important features: mileage, horsepower, car age, engine size.

## Ideas to improve
- Use the `model` column (e.g. the model name has a lot of price information)
- Try cross-validation instead of one split
- Tune the random forest (`n_estimators`, `max_depth`)
