# Indian Housing Prices Backend


This project is a backend machine learning service that predicts housing prices in Indian cities. It is built using **FastAPI** and **CatBoost**, and deployed seamlessly via **Render**. The model was trained on real estate data and serves predictions through a RESTful API.

---

## 🚀 Tech Stack

- **Python** – Core language
- **FastAPI** – High-performance web framework for APIs
- **CatBoost Regressor** – Gradient boosting model optimized for categorical features
- **Pandas & NumPy** – Data preprocessing and numerical operations
- **scikit-learn** – Evaluation metrics and utility functions
- **YAML** – Configuration and schema definitions (`params.yaml`, `schema.yaml`)
- **Render** – Cloud deployment of the FastAPI app

---

## 📊 Model Performance

| Metric                     | Value  |
|----------------------------|--------|
| MAE (Mean Absolute Error)  | 0.1770 |
| MSE (Mean Squared Error)   | 0.0669 |
| RMSE (Root Mean Squared Error) | 0.2586 |
| R² Score                   | 0.9559 |
| Median Absolute Error      | 0.1193 |
| Explained Variance Score   | 0.9559 |

---

## 🌐 Live Demo

You can try the live API hosted on Render:

👉 **[Live API Endpoint](https://indian-housing-prices-backend.onrender.com)**  
🌍 Deployed Project: [https://india-rental-prices.netlify.app/](https://india-rental-prices.netlify.app/)



## 🛠️ Setup Instructions (Local Development)

Clone the repository

```bash
https://github.com/Templar121/Indian-Housing-Prices-Backend
```

### STEP A - Create a conda environment after opening the repository

```bash
conda create -n mlproj python=3.8 -y
```

```bash
conda activate mlproj
```

### OR
### STEP B - Create a venv environment

```bash
python -m venv mlproj
```

```bash
source mlproj/Scripts/activate
```


### STEP 02 - Install the Requirements

```bash
pip install -r requirements.txt
```


### Running the API Server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### 1. Health Check  
**GET /**  
Check if the API is running.  

**Response:**  
```json
{
  "status": "success",
  "message": "Real Estate Price Prediction API is live."
}
```

### 2. Training
**GET /train**
Train the model using the provided data. The model will be saved in the artifacts directory.

***Response***

```json 
{
  "status": "success",
  "message": "Model training completed successfully."
}
```

### 3. Prediction

**POST /predict**
Make a prediction using the trained model.

***Rquest JSON TEMPLATE***

```json
{
  "house_type": "Apartment",
  "city": "Delhi",
  "location": "Connaught Place",
  "numBathrooms": 2,
  "numBalconies": 1,
  "isNegotiable": 1.0,
  "verificationDate": "yyyy-mm-dd",
  "SecurityDeposit": 50000,
  "Status": 1.0,
  "bhk": 3,
  "house_size_sqft": 1200
}
```

***Response***

```json
{
  "status": "success",
  "predicted_price": 12000000.0
}
```


### Model Evaluation Metrics

```json
{
    "MAE": 0.17703379725819957,
    "MSE": 0.06688113072753754,
    "RMSE": 0.25861386414408943,
    "R2 Score": 0.9559500040899803,
    "Median AE": 0.11930351102836667,
    "Explained Variance Score": 0.9559519251384015
}
```
