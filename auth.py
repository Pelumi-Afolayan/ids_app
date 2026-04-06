import json
import bcrypt
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

# ── Config ────────────────────────────────────────────────────────────────
SECRET_KEY  = "ids-net-secret-key-2026"
SESSION_AGE = 3600  # 1 hour in seconds
USERS_FILE  = "users.json"

serializer = URLSafeTimedSerializer(SECRET_KEY)

# ── User helpers ──────────────────────────────────────────────────────────

def load_users():
    with open(USERS_FILE, "r") as f:
        return json.load(f)["users"]


def get_user(username: str):
    for user in load_users():
        if user["username"] == username:
            return user
    return None


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


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

def add_user(username: str, plain_password: str, role: str) -> bool:
    """Add a new user to users.json. Returns False if username already exists."""
    with open(USERS_FILE, "r") as f:
        data = json.load(f)

    # Check for duplicate username
    for user in data["users"]:
        if user["username"] == username:
            return False

    # Hash password and append
    data["users"].append({
        "username": username,
        "password": hash_password(plain_password),
        "role":     role
    })

    with open(USERS_FILE, "w") as f:
        json.dump(data, f, indent=4)

    return True

def delete_user(username: str) -> bool:
    """Delete a user from users.json. Cannot delete admin."""
    with open(USERS_FILE, "r") as f:
        data = json.load(f)

    # Protect the admin account from deletion
    if username == "admin":
        return False

    original_count = len(data["users"])
    data["users"] = [u for u in data["users"] if u["username"] != username]

    if len(data["users"]) == original_count:
        return False  # User not found

    with open(USERS_FILE, "w") as f:
        json.dump(data, f, indent=4)

    return True

def change_password(username: str, old_password: str, new_password: str) -> tuple:
    """
    Change a user's password.
    Returns (True, None) on success or (False, error_message) on failure.
    """
    with open(USERS_FILE, "r") as f:
        data = json.load(f)

    for user in data["users"]:
        if user["username"] == username:
            # Verify old password
            if not verify_password(old_password, user["password"]):
                return False, "Current password is incorrect."
            # Update to new password
            user["password"] = hash_password(new_password)
            with open(USERS_FILE, "w") as f:
                json.dump(data, f, indent=4)
            return True, None

    return False, "User not found."