from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Date, Boolean
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone
import uuid


Base = declarative_base()


class Expense(Base):
    __tablename__ = "expenses"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    bill_no = Column(String, nullable=True)
    expence_name = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    category = Column(String, nullable=False)
    mode = Column(String, nullable=False)
    purchase_date = Column(Date, nullable=True)
    created_date = Column(DateTime, default= lambda: datetime.now(timezone.utc), index=True)

class UserProfile(Base):
    __tablename__ = "user_profiles"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, default="unnamed")
    email = Column(String, unique=True, nullable=False,default="no_email")            # NEW
    username = Column(String, unique=True, nullable=False,default="no_username")         # NEW
    password_hash = Column(String, nullable=False,default="no_password")                 # NEW
    savings_goal = Column(Float, nullable=True)
    financial_goals = Column(Text, nullable=True)
    has_completed_onboarding = Column(Boolean, default=False, nullable=False)
    created_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class FinancialInsight(Base):
    __tablename__ = "financial_insights"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True, nullable=False)
    insights_data = Column(Text, nullable=False)
    generated_date = Column(DateTime, default= lambda: datetime.now(timezone.utc), index=True)
    insights_type = Column(String, nullable=True)


class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)  # 'user' or 'ai'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)



