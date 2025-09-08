import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base, Expense, UserProfile, FinancialInsight

db_path = os.path.join(os.getcwd(), "expenses.db")
Base = Base

DATABASE_URL = f"sqlite:///{db_path}"

engine = create_engine(DATABASE_URL, future=True, echo=True, connect_args={"check_same_thread":False})

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

def init_db():
    print("Initializing the database...")
    Base.metadata.create_all(bind=engine)
    print("Database initialization complete.")


if __name__ == "__main__":
    init_db()