# 🚗 Car Price Prediction using Machine Learning

## Project Overview
This project estimates the selling price of a used car using regression models and an interactive Streamlit application.

The notebook compares Linear Regression, Random Forest, and a GridSearchCV-tuned Random Forest. Linear Regression achieved the strongest test-set R² score in the recorded experiment, so it is selected for the application.

## Dataset
- File: `car data.csv`
- 299 rows after removing 2 duplicate records.
- Target: `Selling_Price`
- Features include present price, kilometers driven, fuel type, selling type, transmission, owner count, and car age.
- The supplied dataset is commonly interpreted in Indian Rupees lakhs. The app fetches an INR/PKR reference exchange rate and caches it for one hour; if the rate service is unavailable, it displays a disclosed fallback rate.

**Important limitation:** Converting the dataset's INR-based estimate to PKR does not make this a Pakistan-market-trained model. Its estimates are educational approximations and should not be treated as actual local market valuations.

## Model Evaluation
Recorded results on the 20% test split (random_state=42):

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 1.473 | 2.525 | 0.753 |
| Random Forest | 1.466 | 3.510 | 0.522 |
| Tuned Random Forest | 1.508 | 3.610 | 0.494 |

Linear Regression was selected because it had the highest recorded test R² score. The Streamlit app fits this selected model on the complete cleaned dataset at startup. It retrieves an INR/PKR reference rate at runtime (one-hour cache) and shows a fallback warning if the service cannot be reached.

## Technologies
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook

## Project Files
- `Car_Price_Prediction.ipynb` — data cleaning, EDA, model comparison, tuning, and final model export.
- `app.py` — Streamlit application; trains the selected Linear Regression model from the included CSV.
- `car data.csv` — source dataset (filename used by the notebook and app).
- `requirements.txt` — Python dependencies, including the HTTP client used for the exchange-rate lookup.

## Live Demo
[Open the Car Price Prediction App](https://codealphacarpriceprediction-nvu7mzjo2yasvnx3bmsm8g.streamlit.app/)

## Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/AbdulRehmanAmjad/Code-Alpha_CarPricePrediction.git
   ```

2. Open the project folder:
   ```bash
   cd Code-Alpha_CarPricePrediction
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:
   ```bash
   streamlit run app.py
   ```

## Author
Abdul Rehman

## Internship
CodeAlpha — Data Science Internship
