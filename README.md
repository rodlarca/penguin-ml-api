# 🐧 Penguin ML Inference API

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Production--Ready-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.1.post1-orange)
![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI-lightgrey)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A production-oriented Machine Learning inference API built with **FastAPI**, serving a pre-trained **scikit-learn** model.

This project demonstrates how to deploy a serialized ML model into a REST API environment with proper validation, error handling, and environment version control.

---

## 🚀 Project Overview

This repository demonstrates:

- REST API development using FastAPI  
- GET and POST endpoints  
- ASGI server with Uvicorn  
- Model loading using joblib  
- Input validation using Pydantic  
- Proper HTTP status handling  
- Feature alignment between API and ML model  
- Virtual environment and dependency version control  
- Basic MLOps best practices  

---

## 🧠 Model Information

- Model type: Scikit-learn Pipeline  
- Dataset: Palmer Penguins  
- Serialization format: joblib (.pkl)  
- Output: Penguin species classification  

### Expected Input Features

- `bill_length_mm`
- `bill_depth_mm`
- `flipper_length_mm`
- `body_mass_g`

### Example Output

```json
{
  "predicted_species": "Adelie"
}
```

---

## 🏗 Architecture Flow

```
Client (curl / browser)
        ↓
FastAPI
        ↓
Pydantic validation
        ↓
Pandas DataFrame
        ↓
Scikit-learn Pipeline
        ↓
Prediction
        ↓
JSON Response
```

---

## 📦 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/penguin-ml-api.git
cd penguin-ml-api
```

### 2️⃣ Create virtual environment (Python 3.11 recommended)

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the API

```bash
python app.py
```

The server will start at:

```
http://localhost:8080
```

Interactive documentation:

```
http://localhost:8080/docs
```

---

## 🔍 API Endpoints

### Health Check

Request:

```
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

---

### Prediction

Request:

```
POST /predict
```

Example body:

```json
{
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181,
  "body_mass_g": 3750
}
```

Example response:

```json
{
  "predicted_species": "Adelie"
}
```

---

## 🛠 Tech Stack

- Python 3.11
- FastAPI
- Uvicorn
- Scikit-learn
- Pandas
- Joblib
- Pydantic

---

## ⚙️ Engineering Decisions

### Version Compatibility

The model was trained with `scikit-learn==1.4.1.post1`.  
The runtime environment matches this version to avoid serialization incompatibilities.

### Input Validation

Pydantic ensures:

- Correct data types  
- Required fields  
- Positive numeric values  
- Automatic HTTP 422 responses for invalid inputs  

### Clean Feature Alignment

The API uses the same feature names expected by the trained model to avoid unnecessary transformation logic.

---

## 📚 Concepts Practiced

- Machine Learning model serving  
- RESTful API design  
- Data validation  
- Environment isolation  
- Dependency management  
- Basic MLOps workflow  
- Error handling and HTTP semantics  

---

## 🎯 Purpose

This project was built as a hands-on exercise to demonstrate how to deploy a Machine Learning model into a production-style API using FastAPI.

It highlights backend engineering, ML integration, and environment management best practices.

---

## 📈 Possible Improvements

- Add probability confidence (`predict_proba`)  
- Add structured logging  
- Dockerize the application  
- Add model versioning  
- Deploy to cloud  
- Add automated tests  

---

## 📄 License

MIT License