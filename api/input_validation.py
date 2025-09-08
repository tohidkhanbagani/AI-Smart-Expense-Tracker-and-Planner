from pydantic import BaseModel, Field


class ExpenseItemIn(BaseModel):
    bill_no: str | None = None
    expence_name: str
    amount: float
    category: str
    mode: str

class InsertExpensesIn(BaseModel):
    items: list[ExpenseItemIn] = Field(default_factory=list)

class RegisterIn(BaseModel):
    name: str
    email: str
    username: str
    password: str
    savings_goal: float = 0.0
    financial_goals: str = ""

class LoginIn(BaseModel):
    login: str        # email or username
    password: str
    ttl_seconds: int = 3600


# NEW: partial or full update for user profile
class UserUpdateIn(BaseModel):
    name: str | None = None
    email: str | None = None
    username: str | None = None
    password: str | None = None
    savings_goal: float | None = None
    financial_goals: str | None = None
    has_completed_onboarding: bool | None = None


# input_validation.py (add at the end)
class ChatIn(BaseModel):
    message: str
    days: int | None = 30     # optional: instead of full history, limit context by days
    limit: int | None = 1000    # optional: cap number of expenses included in context
    include_context: bool = False  # optional: return context for debugging in response
