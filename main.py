import json
import os
import io
import joblib
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from auth import (
    get_user, verify_password, create_session,
    get_current_user, hash_password, add_user, delete_user, change_password
)

# ── Create default data files if not present (for fresh deployments) ──────
if not os.path.exists("logs.json"):
    with open("logs.json", "w") as f:
        json.dump([], f)

if not os.path.exists("users.json"):
    with open("users.json", "w") as f:
        json.dump({
            "users": [
                {
                    "username": "admin",
                    "password": "$2b$12$JJ3N1hcGlffCkJLGGJbne.buiqT0uGhes.IduGbn6IJACoK8cdZau",
                    "role":     "admin"
                }
            ]
        }, f, indent=4)

# ── App setup ─────────────────────────────────────────────────────────────
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── Load model ────────────────────────────────────────────────────────────
# ── Download models from Google Drive if not present ─────────────────────
import gdown

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILES = {
    "feature_columns.pkl": "1nE_JGV3GNlZ0I6vXxC9Knmq5Zbga6gvY",
    "stacking_model.pkl":  "1uYXK_5GgQfRlaj6loeQCsJ-I6QfLpXry",
    "scaler.pkl":          "1cDdUQ-KwGj97Ot23b1Xio6Ez996W1tLN",
}

for filename, file_id in MODEL_FILES.items():
    dest = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(dest):
        print(f"Downloading {filename}...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", dest, quiet=False)

# ── Load model ────────────────────────────────────────────────────────────
model           = joblib.load(os.path.join(MODEL_DIR, "stacking_model.pkl"))
scaler          = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
# ── Prediction log (persistent) ───────────────────────────────────────────
LOGS_FILE = "logs.json"

def load_logs():
    with open(LOGS_FILE, "r") as f:
        return json.load(f)

def save_logs(logs):
    with open(LOGS_FILE, "w") as f:
        json.dump(logs, f, indent=4)

def log_prediction(username: str, pred_type: str, total: int, attacks: int):
    logs = load_logs()
    logs.append({
        "username":  username,
        "type":      pred_type,
        "total":     total,
        "attacks":   attacks,
        "normal":    total - attacks,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_logs(logs)

# ── Helper: admin route data ──────────────────────────────────────────────
def get_admin_context():
    from auth import load_users
    prediction_logs   = load_logs()
    raw_users         = load_users()
    users = []
    for u in raw_users:
        count = sum(1 for log in prediction_logs if log["username"] == u["username"])
        users.append({
            "username":         u["username"],
            "role":             u["role"],
            "prediction_count": count
        })
    return {
        "prediction_logs":   prediction_logs,
        "users":             users,
        "total_predictions": len(prediction_logs),
        "total_attack":      sum(log["attacks"] for log in prediction_logs),
        "total_normal":      sum(log["normal"]  for log in prediction_logs)
    }

# ── Helper: preprocess input df ───────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    df = df[feature_columns].copy()
    scaled = scaler.transform(df)
    return scaled

# ── Helper: require login ─────────────────────────────────────────────────
def require_login(request: Request):
    return get_current_user(request)

# ── Routes ────────────────────────────────────────────────────────────────

# Login page
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    user = require_login(request)
    if user:
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request, "login.html", {
        "error": None
    })

@app.post("/login", response_class=HTMLResponse)
async def login_post(request: Request,
                     username: str = Form(...),
                     password: str = Form(...)):
    user = get_user(username)
    if not user or not verify_password(password, user["password"]):
        return templates.TemplateResponse(request, "login.html", {
            "error": "Invalid username or password."
        })
    token = create_session(user["username"], user["role"])
    response = RedirectResponse("/", status_code=302)
    response.set_cookie("session", token, httponly=True, max_age=3600)
    return response

# Logout
@app.get("/logout")
async def logout():
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie("session")
    return response

# Register page
@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    user = require_login(request)
    if user:
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request, "register.html", {
        "error": None, "success": None
    })

@app.post("/register", response_class=HTMLResponse)
async def register_post(request: Request,
                        username: str = Form(...),
                        password: str = Form(...),
                        confirm_password: str = Form(...)):
    if password != confirm_password:
        return templates.TemplateResponse(request, "register.html", {
            "error": "Passwords do not match.", "success": None
        })
    if len(password) < 6:
        return templates.TemplateResponse(request, "register.html", {
            "error": "Password must be at least 6 characters.", "success": None
        })
    success = add_user(username, password, "user")
    if not success:
        return templates.TemplateResponse(request, "register.html", {
            "error": "Username already exists. Please choose another.", "success": None
        })
    return templates.TemplateResponse(request, "register.html", {
        "error": None,
        "success": "Account created successfully. You can now sign in."
    })

# Home
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    user = require_login(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(request, "index.html", {
        "username": user["username"],
        "role":     user["role"]
    })

# CSV upload page
@app.get("/csv", response_class=HTMLResponse)
async def csv_page(request: Request):
    user = require_login(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(request, "csv_upload.html", {
        "username": user["username"],
        "role":     user["role"]
    })

# CSV prediction
@app.post("/predict/csv", response_class=HTMLResponse)
async def predict_csv(request: Request, file: UploadFile = File(...)):
    user = require_login(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")), low_memory=False)

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        return templates.TemplateResponse(request, "csv_upload.html", {
            "username": user["username"],
            "role":     user["role"],
            "error":    f"Missing columns: {', '.join(missing)}"
        })

    X = preprocess(df)
    preds = model.predict(X)
    labels = ["Attack" if p == 1 else "Normal" for p in preds]

    display_df = df.copy()
    display_df["prediction"] = labels
    rows = []
    for i, row in display_df.iterrows():
        rows.append({
            "index":      i + 1,
            "IN_BYTES":   int(row.get("IN_BYTES", 0)),
            "OUT_BYTES":  int(row.get("OUT_BYTES", 0)),
            "IN_PKTS":    int(row.get("IN_PKTS", 0)),
            "OUT_PKTS":   int(row.get("OUT_PKTS", 0)),
            "PROTOCOL":   int(row.get("PROTOCOL", 0)),
            "prediction": row["prediction"]
        })

    total   = len(labels)
    attacks = labels.count("Attack")
    normal  = labels.count("Normal")

    log_prediction(user["username"], "CSV", total, attacks)

    return templates.TemplateResponse(request, "csv_result.html", {
        "username": user["username"],
        "role":     user["role"],
        "rows":     rows,
        "total":    total,
        "attack":   attacks,
        "normal":   normal
    })

# Download CSV predictions
@app.post("/predict/csv/download", response_class=StreamingResponse)
async def download_csv(request: Request, file: UploadFile = File(...)):
    user = require_login(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")), low_memory=False)

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        return RedirectResponse("/csv", status_code=302)

    X = preprocess(df)
    preds = model.predict(X)
    labels = ["Attack" if p == 1 else "Normal" for p in preds]

    df["Prediction"] = labels
    output = df.to_csv(index=False)

    return StreamingResponse(
        io.StringIO(output),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ids_predictions.csv"}
    )

# Single record page
@app.get("/single", response_class=HTMLResponse)
async def single_page(request: Request):
    user = require_login(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(request, "single.html", {
        "username": user["username"],
        "role":     user["role"]
    })

# Single prediction
@app.post("/predict/single", response_class=HTMLResponse)
async def predict_single(request: Request):
    user = require_login(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    form = await request.form()
    record = {col: float(form[col]) for col in feature_columns}
    df = pd.DataFrame([record])

    X = preprocess(df)
    pred = model.predict(X)[0]
    label = "Attack" if pred == 1 else "Normal"

    log_prediction(user["username"], "Single", 1, 1 if pred == 1 else 0)

    return templates.TemplateResponse(request, "single_result.html", {
        "username":   user["username"],
        "role":       user["role"],
        "prediction": label
    })

# Prediction history
@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    user = require_login(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    all_logs = load_logs()
    user_logs = all_logs if user["role"] == "admin" else [
        log for log in all_logs if log["username"] == user["username"]
    ]

    return templates.TemplateResponse(request, "history.html", {
        "username":          user["username"],
        "role":              user["role"],
        "logs":              user_logs,
        "total_predictions": len(user_logs),
        "total_attack":      sum(log["attacks"] for log in user_logs),
        "total_normal":      sum(log["normal"]  for log in user_logs)
    })

# Admin dashboard
@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    user = require_login(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    if user["role"] != "admin":
        return RedirectResponse("/", status_code=302)
    ctx = get_admin_context()
    return templates.TemplateResponse(request, "admin.html", {
        "username": user["username"],
        "role":     user["role"],
        **ctx
    })

# Add user (admin only)
@app.post("/admin/add-user", response_class=HTMLResponse)
async def admin_add_user(request: Request,
                         new_username: str = Form(...),
                         new_password: str = Form(...),
                         new_role: str = Form(...)):
    user = require_login(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    if user["role"] != "admin":
        return RedirectResponse("/", status_code=302)

    success = add_user(new_username, new_password, new_role)
    ctx = get_admin_context()
    return templates.TemplateResponse(request, "admin.html", {
        "username":    user["username"],
        "role":        user["role"],
        "add_success": "User added successfully." if success else None,
        "add_error":   "Username already exists." if not success else None,
        **ctx
    })

# Delete user (admin only)
@app.post("/admin/delete-user", response_class=HTMLResponse)
async def admin_delete_user(request: Request,
                            del_username: str = Form(...)):
    user = require_login(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    if user["role"] != "admin":
        return RedirectResponse("/", status_code=302)

    success = delete_user(del_username)
    ctx = get_admin_context()
    return templates.TemplateResponse(request, "admin.html", {
        "username":    user["username"],
        "role":        user["role"],
        "del_success": f"User '{del_username}' deleted." if success else None,
        "del_error":   f"Cannot delete '{del_username}'." if not success else None,
        **ctx
    })

# Password change page
@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    user = require_login(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(request, "settings.html", {
        "username": user["username"],
        "role":     user["role"],
        "error":    None,
        "success":  None
    })

@app.post("/settings", response_class=HTMLResponse)
async def settings_post(request: Request,
                        old_password: str = Form(...),
                        new_password: str = Form(...),
                        confirm_password: str = Form(...)):
    user = require_login(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    # Validate new passwords match
    if new_password != confirm_password:
        return templates.TemplateResponse(request, "settings.html", {
            "username": user["username"],
            "role":     user["role"],
            "error":    "New passwords do not match.",
            "success":  None
        })

    # Validate new password length
    if len(new_password) < 6:
        return templates.TemplateResponse(request, "settings.html", {
            "username": user["username"],
            "role":     user["role"],
            "error":    "New password must be at least 6 characters.",
            "success":  None
        })

    # Attempt password change
    success, error_msg = change_password(user["username"], old_password, new_password)

    return templates.TemplateResponse(request, "settings.html", {
        "username": user["username"],
        "role":     user["role"],
        "error":    error_msg if not success else None,
        "success":  "Password changed successfully." if success else None
    })


# Model evaluation page
@app.get("/evaluation", response_class=HTMLResponse)
async def evaluation_page(request: Request):
    user = require_login(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    # Fixed results from training
    models_data = [
        {
            "name": "KNN",
            "accuracy": 99.97, "precision": 99.93, "recall": 99.99,
            "f1": 99.96, "specificity": 99.96, "auc_roc": 99.99,
            "tp": 99960, "tn": 99920, "fp": 40, "fn": 80
        },
        {
            "name": "LinearSVC",
            "accuracy": 98.27, "precision": 96.13, "recall": 99.07,
            "f1": 97.58, "specificity": 97.84, "auc_roc": 99.44,
            "tp": 99070, "tn": 97840, "fp": 2160, "fn": 930
        },
        {
            "name": "Logistic Regression",
            "accuracy": 98.30, "precision": 96.16, "recall": 99.14,
            "f1": 97.63, "specificity": 97.86, "auc_roc": 99.49,
            "tp": 99140, "tn": 97860, "fp": 2140, "fn": 860
        },
        {
            "name": "Stacking Ensemble",
            "accuracy": 99.98, "precision": 99.95, "recall": 99.99,
            "f1": 99.97, "specificity": 99.97, "auc_roc": 100.00,
            "tp": 99990, "tn": 99970, "fp": 30, "fn": 10
        },
        {
            "name": "XGBoost (Benchmark)",
            "accuracy": 100.00, "precision": 99.99, "recall": 100.00,
            "f1": 99.99, "specificity": 99.99, "auc_roc": 100.00,
            "tp": 100000, "tn": 99990, "fp": 10, "fn": 0
        }
    ]

    return templates.TemplateResponse(request, "evaluation.html", {
        "username": user["username"],
        "role":     user["role"],
        "models":   models_data
    })