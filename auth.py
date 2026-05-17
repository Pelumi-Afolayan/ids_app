import os
import bcrypt
from datetime import datetime
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from sqlalchemy import create_engine, Column, Integer, String, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker

# ── Config ────────────────────────────────────────────────────────────────
SECRET_KEY   = "ids-net-secret-key-2026"
SESSION_AGE  = 3600  # 1 hour in seconds
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# SQLAlchemy requires postgresql:// not postgres://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

serializer = URLSafeTimedSerializer(SECRET_KEY)

# ── Database setup ────────────────────────────────────────────────────────
# DATABASE_URL is only available on Render — skip engine creation locally
if DATABASE_URL:
    engine  = create_engine(DATABASE_URL)
    Base    = declarative_base()
    Session = sessionmaker(bind=engine)
else:
    engine  = None
    Base    = declarative_base()
    Session = None

# ── Models ────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    user_id       = Column(Integer, primary_key=True, autoincrement=True)
    username      = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role          = Column(String(10), nullable=False)
    created_at    = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    log_id          = Column(Integer, primary_key=True, autoincrement=True)
    user_id         = Column(Integer, nullable=False)
    prediction_type = Column(String(10), nullable=False)
    total_records   = Column(Integer, nullable=False)
    attack_count    = Column(Integer, nullable=False)
    normal_count    = Column(Integer, nullable=False)
    timestamp       = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create tables and seed default admin account if not present."""
    if not engine:
        print("No DATABASE_URL found — skipping database initialisation.")
        return
    Base.metadata.create_all(engine)
    db = Session()
    try:
        existing = db.query(User).filter_by(username="admin").first()
        if not existing:
            admin = User(
                username      = "admin",
                password_hash = hash_password("admin123"),
                role          = "admin"
            )
            db.add(admin)
            db.commit()
            print("Default admin account created.")
    finally:
        db.close()

# ── Password helpers ──────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

# ── Session helpers ───────────────────────────────────────────────────────

def create_session(username: str, role: str) -> str:
    """Create a signed session token."""
    return serializer.dumps({"username": username, "role": role})


def decode_session(token: str):
    """Decode session token. Returns dict or None."""
    try:
        return serializer.loads(token, max_age=SESSION_AGE)
    except (BadSignature, SignatureExpired):
        return None


def get_current_user(request):
    """Extract current user from request cookie."""
    token = request.cookies.get("session")
    if not token:
        return None
    return decode_session(token)

# ── User helpers ──────────────────────────────────────────────────────────

def get_user(username: str):
    """Fetch a user by username. Returns dict or None."""
    db = Session()
    try:
        user = db.query(User).filter_by(username=username).first()
        if not user:
            return None
        return {
            "username": user.username,
            "password": user.password_hash,
            "role":     user.role
        }
    finally:
        db.close()


def load_users():
    """Return all users as a list of dicts."""
    db = Session()
    try:
        users = db.query(User).all()
        return [
            {"username": u.username, "role": u.role}
            for u in users
        ]
    finally:
        db.close()


def add_user(username: str, plain_password: str, role: str) -> bool:
    """Add a new user. Returns False if username already exists."""
    db = Session()
    try:
        existing = db.query(User).filter_by(username=username).first()
        if existing:
            return False
        new_user = User(
            username      = username,
            password_hash = hash_password(plain_password),
            role          = role
        )
        db.add(new_user)
        db.commit()
        return True
    except Exception:
        db.rollback()
        return False
    finally:
        db.close()


def delete_user(username: str) -> bool:
    """Delete a user. Cannot delete admin."""
    if username == "admin":
        return False
    db = Session()
    try:
        user = db.query(User).filter_by(username=username).first()
        if not user:
            return False
        db.delete(user)
        db.commit()
        return True
    except Exception:
        db.rollback()
        return False
    finally:
        db.close()


def change_password(username: str, old_password: str, new_password: str) -> tuple:
    """
    Change a user's password.
    Returns (True, None) on success or (False, error_message) on failure.
    """
    db = Session()
    try:
        user = db.query(User).filter_by(username=username).first()
        if not user:
            return False, "User not found."
        if not verify_password(old_password, user.password_hash):
            return False, "Current password is incorrect."
        user.password_hash = hash_password(new_password)
        db.commit()
        return True, None
    except Exception:
        db.rollback()
        return False, "An error occurred. Please try again."
    finally:
        db.close()

# ── Prediction log helpers ────────────────────────────────────────────────

def log_prediction_db(username: str, pred_type: str, total: int, attacks: int):
    """Log a prediction session to the database."""
    db = Session()
    try:
        user = db.query(User).filter_by(username=username).first()
        if not user:
            return
        log = PredictionLog(
            user_id         = user.user_id,
            prediction_type = pred_type,
            total_records   = total,
            attack_count    = attacks,
            normal_count    = total - attacks
        )
        db.add(log)
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def load_logs_db(username: str = None, role: str = "user"):
    """
    Load prediction logs from the database.
    Admins get all logs; standard users get only their own.
    Returns a list of dicts compatible with the existing templates.
    """
    db = Session()
    try:
        if role == "admin":
            logs = db.query(PredictionLog, User).join(
                User, PredictionLog.user_id == User.user_id
            ).order_by(PredictionLog.timestamp.desc()).all()
        else:
            user = db.query(User).filter_by(username=username).first()
            if not user:
                return []
            logs = db.query(PredictionLog, User).join(
                User, PredictionLog.user_id == User.user_id
            ).filter(PredictionLog.user_id == user.user_id).order_by(
                PredictionLog.timestamp.desc()
            ).all()

        return [
            {
                "username":  u.username,
                "type":      log.prediction_type,
                "total":     log.total_records,
                "attacks":   log.attack_count,
                "normal":    log.normal_count,
                "timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            }
            for log, u in logs
        ]
    finally:
        db.close()


def load_all_logs_db():
    """Load all prediction logs for admin dashboard."""
    return load_logs_db(role="admin")