from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Depends, Request
from typing import Optional, List, Dict, Any
from fastapi.responses import JSONResponse
import shutil, os, mimetypes, tempfile
from sqlalchemy import select, or_
from pipeline.ocr_model import ExpenseExtractor
from database.repo import DatabaseActions
from database.core import init_db, SessionLocal
from database.models import UserProfile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
import time
from passlib.context import CryptContext
from api.input_validation import LoginIn, RegisterIn,ChatIn, UserUpdateIn # ensure path matches your file layout
from sqlalchemy.exc import IntegrityError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage


from pipeline.nlp_chatbot import FinancialNLPChatbot
from pipeline.financial_insights import FinancialInsightsAnalyzer


SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
ALGORITHM = "HS256"
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI(title="OCR Expense Extractor API", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

extractor = ExpenseExtractor()
db_actions = DatabaseActions()

# --- Chat model init (place near extractor/db_actions)
chat_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# main.py (after extractor/db_actions)
chatbot = FinancialNLPChatbot(model_name=os.getenv("CHAT_MODEL_NAME","gemini-1.5-flash"),
                              db_connection_string=os.getenv("DATABASE_URL","sqlite:///expenses.db"))
insights_engine = FinancialInsightsAnalyzer(model_name=os.getenv("CHAT_MODEL_NAME","gemini-1.5-flash"),
                                            db_connection_string=os.getenv("DATABASE_URL","sqlite:///expenses.db"))






def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

@app.on_event("startup")
def load_assets():
    init_db()

# ---------- Unified response helpers ----------
def ok(data=None, message="OK", status_code: int = 200):
    return JSONResponse(
        status_code=status_code,
        content={"success": True, "message": message, "data": data, "error": None},
    )

def fail(message="Error", code="ERROR", details=None, status_code: int = 400):
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "message": message, "data": None, "error": {"code": code, "details": details}},
    )



def build_user_context(user_id: str, days: int | None = None, limit: int | None = None) -> Dict[str, Any]:
    # Profile
    profile = db_actions.fetch_user_profile(user_id=user_id)

    # Expenses
    if days is None:
        all_exp = db_actions.list_all_expenses(user_id=user_id)  # NEW repo method
        expenses = all_exp.get("expenses", [])
    else:
        exp = db_actions.list_expenses(user_id=user_id, days=days, category=None)
        expenses = exp.get("expenses", [])
    if limit and isinstance(expenses, list):
        expenses = expenses[:max(0, limit)]

    # Insights
    ins = db_actions.list_insights(user_id=user_id)
    insights = ins.get("insights", [])

    # Calculate analytics summaries
    category_summary = {}
    daily_summary = {}
    if expenses:
        for exp in expenses:
            # Category summary
            cat = exp.get("category", "Misc")
            if cat not in category_summary:
                category_summary[cat] = {"amount": 0, "count": 0}
            category_summary[cat]["amount"] += exp.get("amount", 0)
            category_summary[cat]["count"] += 1
            
            # Daily summary
            pdate = exp.get("date")
            if pdate and isinstance(pdate, str):
                day = pdate.split("T")[0]
                if day not in daily_summary:
                    daily_summary[day] = {"amount": 0, "count": 0}
                daily_summary[day]["amount"] += exp.get("amount", 0)
                daily_summary[day]["count"] += 1

    return {
        "profile": profile or {},
        "expenses": expenses,
        "insights": insights,
        "analytics_summary": {
            "category_summary": category_summary,
            "daily_summary": daily_summary
        },
        "context_meta": {
            "source": "smart-expense-tracker",
            "profile_present": bool(profile),
            "expenses_count": len(expenses),
            "insights_count": len(insights),
        },
    }



# Centralized exception shaping
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException as FastApiHTTPException

@app.exception_handler(FastApiHTTPException)
async def http_exception_handler(request: Request, exc: FastApiHTTPException):
    return fail(message=exc.detail if isinstance(exc.detail, str) else "HTTP error", code=f"HTTP_{exc.status_code}", details=exc.detail, status_code=exc.status_code)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return fail(message="Validation error", code="VALIDATION_ERROR", details=exc.errors(), status_code=422)

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return fail(message="An unexpected error occurred.", code="UNEXPECTED_ERROR", details=str(exc), status_code=500)

# ---------- Auth helpers ----------
def create_access_token(user_id: str, ttl_seconds: int = 3600) -> str:
    payload = {"sub": user_id, "exp": int(time.time()) + ttl_seconds}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise FastApiHTTPException(status_code=401, detail="Invalid token subject")
    except JWTError:
        raise FastApiHTTPException(status_code=401, detail="Invalid or expired token")

    with SessionLocal() as db:
        row = db.execute(select(UserProfile.name).where(UserProfile.user_id == user_id)).scalar_one_or_none()
    if row is None:
        raise FastApiHTTPException(status_code=404, detail="User not found")
    return {"user_id": user_id, "user_name": row}

@app.get("/chat/history")
def get_chat_history(current_user: dict = Depends(get_current_user)):
    history = db_actions.get_chat_history(user_id=current_user["user_id"])
    return ok(data=history, message="Chat history retrieved")

# ---------- Auth endpoints ----------
@app.post("/auth/register")
def register(payload: RegisterIn):
    pw_hash = hash_password(payload.password)
    try:
        profile = db_actions.create_user_credentials(
            name=payload.name,
            email=payload.email,
            username=payload.username,
            password_hash=pw_hash,
            savings_goal=payload.savings_goal,
            financial_goals=payload.financial_goals,
        )
        return ok(data={"user_id": profile["user_id"], "email": payload.email, "username": payload.username}, message="Registered")
    except Exception as e:
        return fail(message="Registration failed", code="REGISTER_FAILED", details=str(e), status_code=400)

@app.post("/auth/login_human")
def login_human(payload: LoginIn):
    with SessionLocal() as db:
        u = db.execute(
            select(UserProfile).where(or_(UserProfile.email == payload.login.lower(), UserProfile.username == payload.login))
        ).scalars().first()
    if not u or not verify_password(payload.password, u.password_hash):
        return fail(message="Invalid credentials", code="INVALID_CREDENTIALS", status_code=401)
    token = create_access_token(user_id=u.user_id, ttl_seconds=payload.ttl_seconds)
    return ok(data={"access_token": token, "token_type": "bearer", "user_id": u.user_id, "name": u.name}, message="Logged in")

# ---------- Me endpoints ----------
@app.get("/me/profile")
def me_profile(current=Depends(get_current_user)):
    data = db_actions.fetch_user_profile(user_id=current["user_id"])
    if not data:
        return fail(message="Profile not found", code="PROFILE_NOT_FOUND", status_code=404)
    return ok(data={"user_id": current["user_id"], "user_name": current["user_name"], "profile": data}, message="Profile")

@app.get("/me/expenses")
def me_expenses(days: int = 30, category: Optional[str] = None, current=Depends(get_current_user)):
    result = db_actions.list_expenses(user_id=current["user_id"], days=days, category=category)
    result["user_name"] = current["user_name"]
    items = result.pop("expenses", [])
    return ok(data={**result, "items": items}, message="Expenses")

@app.get("/me/insights")
def me_insights(current=Depends(get_current_user)):
    result = db_actions.list_insights(user_id=current["user_id"])
    result["user_name"] = current["user_name"]
    items = result.pop("insights", [])
    return ok(data={**result, "items": items}, message="Insights")

# ---------- Extract (producer) ----------
@app.post("/extract")
def extraction(file: UploadFile = File(..., description="upload document or image")):
    suffix = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tp:
        shutil.copyfileobj(file.file, tp)
        temp_path = tp.name
    try:
        mime, _ = mimetypes.guess_type(file.filename)
        input_type = "pdf" if mime == "application/pdf" or suffix == ".pdf" else "image"
        extracted = extractor.extract_expense(input_data=temp_path, input_type=input_type)
        if isinstance(extracted, str):
            return fail(message=extracted, code="NOT_A_BILL", status_code=400)
        # Normalize to a list
        if isinstance(extracted, dict):
            items = extracted.get("extracted data") or extracted.get("items") or extracted.get("data", {}).get("items")
        else:
            items = extracted
        if not isinstance(items, list):
            return fail(message="Unexpected OCR output", code="BAD_OCR_SHAPE", details=str(type(items)))
        # Unified: data.items so it can be pasted into /insert_expenses
        return ok(data={"items": items, "input_type": input_type}, message="Extracted")
    except Exception as e:
        return fail(message="OCR extraction failed", code="OCR_FAILED", details=str(e), status_code=500)
    finally:
        try: os.remove(temp_path)
        except Exception: pass

# ---------- Expenses (consumer + CRUD) ----------
@app.post("/insert_expenses")
async def insert_expenses(user_id: str, body: dict = Body(...)):
    with SessionLocal() as db:
        exists = db.execute(select(UserProfile.user_id).where(UserProfile.user_id == user_id)).scalar_one_or_none()
    if not exists:
        return fail(message=f"user_id {user_id} not found", code="USER_NOT_FOUND", status_code=404)
    items = ((body.get("data") or {}).get("items")) or body.get("items")
    if not isinstance(items, list) or not items:
        return fail(message="Provide non-empty data.items (list)", code="EMPTY_ITEMS", status_code=400)
    inserted = db_actions.insert_expenses(user_id=user_id, expenses=items)
    return ok(data={"user_id": user_id, "inserted": inserted}, message="Inserted")

@app.get("/expenses/{user_id}")
def list_expenses(user_id: str, days: int = 30, category: Optional[str] = None):
    try:
        result = db_actions.list_expenses(user_id=user_id, days=days, category=category)
        items = result.pop("expenses", [])
        return ok(data={**result, "items": items}, message="Expenses")
    except Exception as e:
        return fail(message="Expenses fetch failed", code="EXPENSES_FETCH_FAILED", details=str(e), status_code=500)

@app.get("/expense/{expense_id}")
def get_expense(expense_id: int):
    data = db_actions.get_expense(expense_id)
    if not data:
        return fail(message="Expense not found", code="EXPENSE_NOT_FOUND", status_code=404)
    return ok(data=data, message="Expense")

@app.put("/expense/{expense_id}")
def update_expense(expense_id: int, payload: dict):
    updated = db_actions.update_expense(expense_id, payload)
    if not updated:
        return fail(message="Expense not found or no valid fields", code="UPDATE_FAILED", status_code=404)
    return ok(data=updated, message="Updated")

# Scoped deletes
@app.delete("/users/{user_id}/expenses/{expense_id}")
def delete_user_expense(user_id: str, expense_id: int):
    deleted = db_actions.delete_expense_for_user(user_id=user_id, expense_id=expense_id)
    if deleted == 0:
        return fail(message="Expense not found for this user", code="DELETE_NOT_FOUND", status_code=404)
    return ok(data={"deleted": deleted}, message="Deleted")

@app.delete("/users/{user_id}/expenses")
def delete_user_expenses(user_id: str):
    deleted = db_actions.delete_all_expenses_for_user(user_id=user_id)
    return ok(data={"deleted": deleted}, message="Deleted all expenses")




# ----------------------------
# User profile endpoints
# ----------------------------
@app.put("/me/profile")
def update_my_profile(payload: UserUpdateIn, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    # Build update dict; hash password if present
    update_data: Dict[str, Any] = {}
    if payload.name is not None:
        update_data["name"] = payload.name
    if payload.email is not None:
        update_data["email"] = payload.email.strip().lower()
    if payload.username is not None:
        update_data["username"] = payload.username.strip()
    if payload.password is not None:
        update_data["password_hash"] = hash_password(payload.password)  # store hash only
    if payload.savings_goal is not None:
        update_data["savings_goal"] = payload.savings_goal
    if payload.financial_goals is not None:
        update_data["financial_goals"] = payload.financial_goals
    if payload.has_completed_onboarding is not None:
        update_data["has_completed_onboarding"] = payload.has_completed_onboarding

    if not update_data:
        return fail(message="No valid fields provided", code="EMPTY_UPDATE", status_code=400)

    try:
        updated = db_actions.update_user_any(user_id, update_data)
        if not updated:
            return fail(message="User profile not found", code="USER_NOT_FOUND", status_code=404)
        return ok(data=updated, message="User profile updated")
    except IntegrityError as e:
        # Likely unique constraint on email/username
        return fail(message="Email or username already in use", code="DUPLICATE_KEY", details=str(e), status_code=409)
    except Exception as e:
        return fail(message="Update failed", code="UPDATE_FAILED", details=str(e), status_code=500)


@app.delete("/me/profile")
def delete_my_profile(cascade: bool = False, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    try:
        deleted = db_actions.delete_user(user_id=user_id, cascade=cascade)
        if deleted == 0:
            return fail(message="User not found", code="USER_NOT_FOUND", status_code=404)
        return ok(data={"deleted": deleted, "cascade": cascade}, message="User deleted")
    except Exception as e:
        return fail(message="Delete failed", code="DELETE_FAILED", details=str(e), status_code=500)


# ---------- Insights ----------
@app.post("/insights/{user_id}")
def save_insight(user_id: str, insights_data: str, insights_type: Optional[str] = None):
    try:
        saved = db_actions.save_insight(user_id=user_id, insights_data=insights_data, insights_type=insights_type)
        return ok(data=saved, message="Saved insight")
    except Exception as e:
        return fail(message="Save insight failed", code="INSIGHT_SAVE_FAILED", details=str(e), status_code=500)

@app.get("/insights/{user_id}")
def list_insights(user_id: str):
    try:
        res = db_actions.list_insights(user_id=user_id)
        items = res.pop("insights", [])
        return ok(data={**res, "items": items}, message="Insights")
    except Exception as e:
        return fail(message="Insights fetch failed", code="INSIGHTS_FETCH_FAILED", details=str(e), status_code=500)

@app.get("/insights/{user_id}/latest")
def latest_insight(user_id: str):
    data = db_actions.get_latest_insight(user_id=user_id)
    if not data:
        return fail(message="No insights found", code="INSIGHT_NOT_FOUND", status_code=404)
    return ok(data=data, message="Latest insight")

@app.get("/insight/{insight_id}")
def get_insight(insight_id: int):
    data = db_actions.get_insight(insight_id)
    if not data:
        return fail(message="Insight not found", code="INSIGHT_NOT_FOUND", status_code=404)
    return ok(data=data, message="Insight")

@app.delete("/users/{user_id}/insights/{insight_id}")
def delete_user_insight(user_id: str, insight_id: int):
    deleted = db_actions.delete_insight_for_user(user_id=user_id, insight_id=insight_id)
    if deleted == 0:
        return fail(message="Insight not found for this user", code="DELETE_NOT_FOUND", status_code=404)
    return ok(data={"deleted": deleted}, message="Deleted")

@app.delete("/users/{user_id}/insights")
def delete_user_insights(user_id: str):
    deleted = db_actions.delete_all_insights_for_user(user_id=user_id)
    return ok(data={"deleted": deleted}, message="Deleted all insights")









CHAT_SYSTEM_PROMPT = (
    "You are a helpful and friendly finance assistant for a personal expense tracker. "
    "Always use the provided JSON context to ground your answers in facts. "
    "The context includes user profile, recent expenses, past insights, and an analytics_summary of daily and category spending. "
    "Provide detailed, actionable, and specific advice based on all this data. "
    "If data is missing, explain what you can do with the available information and suggest the next step. "
    "Your tone should be encouraging and conversational."
)

@app.post("/chat")
def chat(payload: ChatIn, current=Depends(get_current_user)):
    try:
        # Build context
        ctx = build_user_context(
            user_id=current["user_id"],
            days=payload.days,
            limit=payload.limit,
        )

        # Compose messages
        system = SystemMessage(content=CHAT_SYSTEM_PROMPT)
        context_blob = JSONResponse(content=ctx).body.decode("utf-8")  # serialize dict to compact JSON string
        human = HumanMessage(content=f"Context(JSON): {context_blob}\n\nUser: {payload.message}")

        # Call LLM
        resp = chat_llm.invoke([system, human])
        assistant_text = resp.content if isinstance(resp.content, str) else str(resp.content)

        # Save conversation
        db_actions.add_chat_message(user_id=current["user_id"], role="user", content=payload.message)
        db_actions.add_chat_message(user_id=current["user_id"], role="ai", content=assistant_text)

        # Unified envelope
        data = {"reply": assistant_text}
        if payload.include_context:
            data["context"] = ctx
        return ok(data=data, message="Chat")
    except Exception as e:
        return fail(message="Chat failed", code="CHAT_FAILED", details=str(e), status_code=500)