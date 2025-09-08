# database/repo.py
from sqlalchemy import select, desc, update, delete, Boolean
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timedelta, timezone, date
from typing import List, Dict, Any, Optional
from database.core import SessionLocal
from database.models import Expense, UserProfile, FinancialInsight, ChatHistory

def _to_iso(dt):
    return dt.isoformat() if isinstance(dt, datetime) else None


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    # Expect ISO "YYYY-MM-DD"; be forgiving if a datetime arrives
    try:
        # Try date-only first
        return date.fromisoformat(s)
    except Exception:
        try:
            return datetime.fromisoformat(s).date()
        except Exception:
            return None


def list_all_expenses(self, user_id: str) -> Dict[str, Any]:
        with SessionLocal() as db:
            name = self._get_user_name(db, user_id)
            stmt = select(Expense).where(Expense.user_id == user_id).order_by(desc(Expense.created_date))
            rows = db.execute(stmt).scalars().all()
            return {
                "user_id": user_id,
                "user_name": name,
                "expenses": [self._expense_to_dict(r) for r in rows],
            }






class DatabaseActions:
    def __init__(self):
        pass

    # ----------------------------
    # Helpers
    # ----------------------------
    def _get_user_name(self, db, user_id: str) -> Optional[str]:
        stmt = select(UserProfile.name).where(UserProfile.user_id == user_id)
        return db.execute(stmt).scalar_one_or_none()

    def _expense_to_dict(self, r: Expense) -> Dict[str, Any]:
        return {
            "id": r.id,
            "user_id": r.user_id,
            "bill_no": r.bill_no,
            "expence_name": r.expence_name,
            "amount": r.amount,
            "category": r.category,
            "mode": r.mode,
            "date": r.purchase_date.isoformat() if r.purchase_date else None,  # NEW
            "created_date": _to_iso(r.created_date),
        }

    def _profile_to_dict(self, u: UserProfile) -> Dict[str, Any]:
            return {
                "id": u.id,
                "user_id": u.user_id,
                "name": u.name,
                "email": u.email,             # NEW
                "username": u.username,       # NEW
                "savings_goal": u.savings_goal,
                "financial_goals": u.financial_goals,
                "has_completed_onboarding": u.has_completed_onboarding,
                "created_date": _to_iso(u.created_date),
                "updated_date": _to_iso(u.updated_date),
            }

    def _insight_to_dict(self, i: FinancialInsight) -> Dict[str, Any]:
        return {
            "id": i.id,
            "user_id": i.user_id,
            "insights_data": i.insights_data,
            "insights_type": i.insights_type,
            "generated_date": _to_iso(i.generated_date),
        }

    def ensure_user_exists(self, user_id: str) -> bool:
        with SessionLocal() as db:
            return db.execute(select(UserProfile.user_id).where(UserProfile.user_id == user_id)).scalar_one_or_none() is not None

    def create_user_credentials(self, name: str, email: str, username: str, password_hash: str, savings_goal: float = 0.0, financial_goals: str = "") -> dict:
        with SessionLocal() as db:
            u = UserProfile(
                name=name or "unnamed",
                email=email.lower().strip(),
                username=username.strip(),
                password_hash=password_hash,
                savings_goal=savings_goal,
                financial_goals=financial_goals or "",
            )
            db.add(u)
            db.commit()
            db.refresh(u)
            return self._profile_to_dict(u)

    def get_user_by_email_or_username(self, login: str) -> Optional[dict]:
        login_norm = login.strip().lower()
        with SessionLocal() as db:
            stmt = select(UserProfile).where((UserProfile.email == login_norm) | (UserProfile.username == login_norm))
            u = db.execute(stmt).scalars().first()
            return self._profile_to_dict(u) if u else None

    # ----------------------------
    # Expenses CRUD
    # ----------------------------
    def insert_expenses(self, user_id: str, expenses: List[Dict[str, Any]]) -> int:
            with SessionLocal() as db:
                objs = []
                for e in expenses:
                    objs.append(
                        Expense(
                            user_id=user_id,
                            bill_no=e.get("bill_no"),
                            expence_name=e["expence_name"],
                            amount=float(e["amount"]),
                            category=e["category"],
                            mode=e["mode"],
                            purchase_date=_parse_date(e.get("date")),  # NEW
                        )
                    )
                db.add_all(objs)
                db.commit()
                return len(objs)
# ----------------------------------------------------------------------
    def list_expenses(self, user_id: str, days: int = 30, category: Optional[str] = None) -> Dict[str, Any]:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        with SessionLocal() as db:
            name = self._get_user_name(db, user_id)
            stmt = select(Expense).where(Expense.user_id == user_id, Expense.created_date >= since)
            if category:
                stmt = stmt.where(Expense.category == category)
            stmt = stmt.order_by(desc(Expense.created_date))
            rows = db.execute(stmt).scalars().all()
            return {
                "user_id": user_id,
                "user_name": name,
                "period_days": days,
                "category_filter": category,
                "expenses": [self._expense_to_dict(r) for r in rows],
            }

    def get_expense(self, expense_id: int) -> Optional[Dict[str, Any]]:
        with SessionLocal() as db:
            r = db.get(Expense, expense_id)
            return self._expense_to_dict(r) if r else None

    def update_expense(self, expense_id: int, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        allowed = {"bill_no", "expence_name", "amount", "category", "mode"}
        payload = {k: v for k, v in data.items() if k in allowed}
        if not payload:
            return None
        with SessionLocal() as db:
            stmt = update(Expense).where(Expense.id == expense_id).values(**payload).execution_options(synchronize_session="fetch")
            res = db.execute(stmt)
            if res.rowcount == 0:
                db.rollback()
                return None
            db.commit()
            r = db.get(Expense, expense_id)
            return self._expense_to_dict(r) if r else None

    def delete_expense_for_user(self, user_id: str, expense_id: int) -> int:
        with SessionLocal() as db:
            stmt = delete(Expense).where(Expense.user_id == user_id, Expense.id == expense_id)
            res = db.execute(stmt)
            db.commit()
            return res.rowcount or 0

    def delete_all_expenses_for_user(self, user_id: str) -> int:
        with SessionLocal() as db:
            res = db.execute(delete(Expense).where(Expense.user_id == user_id))
            db.commit()
            return res.rowcount or 0

    # ----------------------------
    # UserProfiles CRUD
    # ----------------------------
    def create_user(self, user_name: str = "unnamed", savings_goal: float = 0.0, financial_goals: str = "") -> Dict[str, Any]:
        with SessionLocal() as db:
            u = UserProfile(name=user_name or "unnamed", savings_goal=savings_goal, financial_goals=financial_goals or "")
            db.add(u)
            db.commit()
            db.refresh(u)
            return self._profile_to_dict(u)

    def fetch_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        with SessionLocal() as db:
            stmt = select(UserProfile).where(UserProfile.user_id == user_id)
            u = db.execute(stmt).scalars().first()
            return self._profile_to_dict(u) if u else None

    # NEW: update any profile fields (password must be hashed by caller)
    def update_user_any(self, user_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        allowed = {"name", "email", "username", "password_hash", "savings_goal", "financial_goals", "has_completed_onboarding"}
        payload = {k: v for k, v in data.items() if k in allowed}
        if not payload:
            return None
        payload["updated_date"] = datetime.now(timezone.utc)
        with SessionLocal() as db:
            stmt = (
                update(UserProfile)
                .where(UserProfile.user_id == user_id)
                .values(**payload)
                .execution_options(synchronize_session="fetch")
            )
            res = db.execute(stmt)
            if res.rowcount == 0:
                db.rollback()
                return None
            db.commit()
            u = db.execute(select(UserProfile).where(UserProfile.user_id == user_id)).scalars().first()
            return self._profile_to_dict(u) if u else None

    def delete_user(self, user_id: str, cascade: bool = False) -> int:
        with SessionLocal() as db:
            if cascade:
                db.execute(delete(Expense).where(Expense.user_id == user_id))
                db.execute(delete(FinancialInsight).where(FinancialInsight.user_id == user_id))
            res = db.execute(delete(UserProfile).where(UserProfile.user_id == user_id))
            db.commit()
            return res.rowcount or 0

    def list_users(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        with SessionLocal() as db:
            stmt = select(UserProfile).order_by(UserProfile.created_date.desc()).limit(limit).offset(offset)
            rows = db.execute(stmt).scalars().all()
            return [self._profile_to_dict(u) for u in rows]

    # ----------------------------
    # FinancialInsights CRUD
    # ----------------------------
    def save_insight(self, user_id: str, insights_data: str, insights_type: Optional[str] = None) -> Dict[str, Any]:
        with SessionLocal() as db:
            fi = FinancialInsight(user_id=user_id, insights_data=insights_data, insights_type=insights_type)
            db.add(fi)
            db.commit()
            db.refresh(fi)
            return self._insight_to_dict(fi)

    def list_insights(self, user_id: str) -> Dict[str, Any]:
        with SessionLocal() as db:
            name = self._get_user_name(db, user_id)
            stmt = select(FinancialInsight).where(FinancialInsight.user_id == user_id).order_by(desc(FinancialInsight.generated_date))
            rows = db.execute(stmt).scalars().all()
            return {
                "user_id": user_id,
                "user_name": name,
                "insights": [self._insight_to_dict(i) for i in rows],
            }

    def get_latest_insight(self, user_id: str) -> Optional[Dict[str, Any]]:
        with SessionLocal() as db:
            stmt = select(FinancialInsight).where(FinancialInsight.user_id == user_id).order_by(desc(FinancialInsight.generated_date)).limit(1)
            i = db.execute(stmt).scalars().first()
            return self._insight_to_dict(i) if i else None

    def get_insight(self, insight_id: int) -> Optional[Dict[str, Any]]:
        with SessionLocal() as db:
            i = db.get(FinancialInsight, insight_id)
            return self._insight_to_dict(i) if i else None

    def delete_insight_for_user(self, user_id: str, insight_id: int) -> int:
        with SessionLocal() as db:
            stmt = delete(FinancialInsight).where(FinancialInsight.user_id == user_id, FinancialInsight.id == insight_id)
            res = db.execute(stmt)
            db.commit()
            return res.rowcount or 0

    def delete_all_insights_for_user(self, user_id: str) -> int:
        with SessionLocal() as db:
            res = db.execute(delete(FinancialInsight).where(FinancialInsight.user_id == user_id))
            db.commit()
            return res.rowcount or 0

    # ----------------------------
    # ChatHistory CRUD
    # ----------------------------
    def add_chat_message(self, user_id: str, role: str, content: str) -> dict:
        with SessionLocal() as db:
            msg = ChatHistory(user_id=user_id, role=role, content=content)
            db.add(msg)
            db.commit()
            db.refresh(msg)
            return {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }

    def get_chat_history(self, user_id: str, limit: int = 50) -> List[dict]:
        with SessionLocal() as db:
            stmt = select(ChatHistory).where(ChatHistory.user_id == user_id).order_by(ChatHistory.timestamp.asc()).limit(limit)
            rows = db.execute(stmt).scalars().all()
            return [
                {"role": r.role, "content": r.content}
                for r in rows
            ]
