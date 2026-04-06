# IDS Net — Intrusion Detection System

A hybridized machine learning web application for detecting network intrusions.
Built as a Final Year Project at Redeemer's University Nigeria, 2026.

## Stack
- **Backend**: FastAPI + Uvicorn
- **Frontend**: HTML + CSS (Jinja2 templates)
- **ML Model**: Stacking Ensemble (KNN + LinearSVC + Logistic Regression)
- **Dataset**: SIMARGL 2022

## Features
- Role-based access control (Admin / User)
- CSV bulk prediction with filtering and pagination
- Single record manual entry
- Prediction history per user
- Admin dashboard with user management
- Password change

## Setup

### 1. Install dependencies
```bash
pip install fastapi uvicorn joblib pandas scikit-learn xgboost python-multipart jinja2 itsdangerous bcrypt
```

### 2. Add model files
Place the following files in the `models/` folder:
- `stacking_model.pkl`
- `scaler.pkl`
- `feature_columns.pkl`

### 3. Create required files
Create `logs.json` with content: `[]`
Create `users.json` with your admin credentials (see `users.json.example`)

### 4. Run the server
```bash
python -m uvicorn main:app --reload
```

Open `http://127.0.0.1:8000` in your browser.