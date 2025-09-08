from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain.schema import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import json
from datetime import datetime, timedelta
import sqlite3
from sqlalchemy import create_engine, text
import pandas as pd


load_dotenv()

class FinancialInsightsAnalyzer:
    """
    Advanced Financial Insights Analyzer that automatically fetches user data 
    from database and provides personalized financial insights and recommendations.
    """
    
    def __init__(self, model_name: str = "gemini-1.5-flash", db_connection_string: str = None):
        self.model_name = model_name
        
        # Initialize LLM with same setup as OCR model
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Database connection
        self.db_connection = db_connection_string or os.getenv("DATABASE_URL", "sqlite:///expenses.db")
        self.engine = create_engine(self.db_connection)
        
        # Enhanced system prompt for financial insights
        self.financial_insights_prompt = """
You are an expert financial advisor AI with deep knowledge of personal finance, budgeting, and expense management. Your role is to analyze user expense data and provide precise, actionable financial insights.

CORE RESPONSIBILITIES:
- Analyze spending patterns with statistical precision
- Identify financial risks and opportunities  
- Provide personalized, evidence-based recommendations
- Generate realistic budget suggestions
- Assess financial health with scoring methodology

OUTPUT REQUIREMENTS:
- Return only valid JSON without explanations or markdown
- Use exact key names as specified in each method
- Ensure all numeric values are properly formatted
- Categories must match predefined lists exactly
- Recommendations must be specific and actionable

ANALYSIS PRINCIPLES:
- Base insights on actual data patterns, not assumptions
- Consider seasonal variations and spending trends
- Prioritize high-impact, achievable recommendations
- Account for user's income-to-expense ratios
- Focus on sustainable financial habits

FINANCIAL CATEGORIES (use only these):
["Food", "Grocery", "Transport", "Fuel", "Travel", "Utilities", "Rent", "Health", "Pharmacy", "Education", "Entertainment", "Shopping", "Electronics", "Home", "Services", "Subscriptions", "Fees", "Taxes", "Office", "Misc"]

RECOMMENDATION TYPES:
- Budget optimization
- Expense reduction strategies
- Savings opportunities
- Investment suggestions
- Debt management
- Emergency fund planning

POLICY:
- Only use data provided in the prompt and database.
- Never ask to connect bank accounts or external sources.
- If a field is unknown (e.g., income), set it to null and proceed with best-effort analysis.

"""
    
    # 3) Optionally include purchase_date in expenses if you want the LLM to use it
    def fetch_user_expenses(self, user_id: str, days: int = 30) -> List[Dict]:
        try:
            query = text(f"""
                SELECT bill_no, expence_name, amount, category, mode, 
                    COALESCE(purchase_date, DATE(created_date)) AS date, created_date
                FROM expenses
                WHERE user_id = :user_id
                AND created_date >= datetime('now', '-{days} days')
                ORDER BY created_date DESC
            """)
            with self.engine.connect() as conn:
                result = conn.execute(query, {"user_id": user_id})
                # row is Row; convert via _mapping for SQLAlchemy 2.x
                return [dict(r._mapping) for r in result]
        except Exception as e:
            print(f"Database error: {e}")
            return []
    
    # 2) Use actual columns from current UserProfile
    def fetch_user_profile(self, user_id: str) -> Dict[str, Any]:
        try:
            query = text("""
                SELECT name, email, username, savings_goal, financial_goals, created_date, updated_date
                FROM user_profiles
                WHERE user_id = :user_id
            """)
            with self.engine.connect() as conn:
                result = conn.execute(query, {"user_id": user_id})
                row = result.fetchone()
                return dict(row._mapping) if row else {}
        except Exception as e:
            print(f"Database error: {e}")
            return {}
    
    def analyze_comprehensive_insights(self, user_id: str, analysis_period: int = 30) -> Dict[str, Any]:
        """
        Generate comprehensive financial insights for a user by fetching data automatically.
        
        Args:
            user_id: User identifier
            analysis_period: Days to analyze (default: 30)
        
        Returns:
            Complete financial insights dictionary
        """
        
        # Fetch data from database
        expenses = self.fetch_user_expenses(user_id, analysis_period)
        user_profile = self.fetch_user_profile(user_id)
        
        if not expenses:
            return {"error": "No expense data found for user", "user_id": user_id}
        
        # Prepare comprehensive analysis prompt
        prompt = f"""
        {self.financial_insights_prompt}
        
        Analyze the following user's financial data and provide comprehensive insights:
        
        EXPENSE DATA ({len(expenses)} transactions over {analysis_period} days):
        {json.dumps(expenses, indent=2)}
        
        USER PROFILE:
        {json.dumps(user_profile, indent=2)}
        
        Provide analysis in this exact JSON format:
        {{
            "financial_summary": {{
                "total_spending": 0,
                "average_daily_spending": 0,
                "total_transactions": 0,
                "analysis_period_days": {analysis_period}
            }},
            "spending_breakdown": {{
                "category_analysis": {{
                    "Food": {{"amount": 0, "percentage": 0, "transaction_count": 0}},
                    "Grocery": {{"amount": 0, "percentage": 0, "transaction_count": 0}}
                }},
                "payment_mode_analysis": {{
                    "cash": {{"amount": 0, "percentage": 0, "count": 0}},
                    "card": {{"amount": 0, "percentage": 0, "count": 0}},
                    "upi": {{"amount": 0, "percentage": 0, "count": 0}}
                }}
            }},
            "financial_health_score": {{
                "overall_score": 0,
                "spending_discipline": 0,
                "budget_adherence": 0,
                "savings_rate": 0,
                "category_balance": 0
            }},
            "key_insights": [
                "Insight about spending patterns",
                "Insight about financial behavior",
                "Insight about budget efficiency"
            ],
            "personalized_recommendations": {{
                "immediate_actions": [
                    "Specific action 1",
                    "Specific action 2"
                ],
                "budget_optimizations": [
                    "Budget suggestion 1",
                    "Budget suggestion 2"
                ],
                "savings_opportunities": [
                    {{"category": "Food", "potential_monthly_savings": 0, "strategy": "Specific strategy"}},
                    {{"category": "Transport", "potential_monthly_savings": 0, "strategy": "Specific strategy"}}
                ]
            }},
            "spending_alerts": [
                {{"type": "high_spending", "category": "Food", "message": "Alert message", "priority": "medium"}},
                {{"type": "unusual_pattern", "category": "Entertainment", "message": "Alert message", "priority": "low"}}
            ],
            "projected_monthly_budget": {{
                "Food": 0,
                "Grocery": 0,
                "Transport": 0,
                "total_suggested_budget": 0,
                "projected_savings": 0
            }}
        }}
        """
        
        try:
            msg = HumanMessage(content=[{"type": "text", "text": prompt}])
            response = self.llm.invoke([msg])
            parser = JsonOutputParser()
            
            insights = parser.parse(response.content)
            insights["user_id"] = user_id
            insights["generated_at"] = datetime.now().isoformat()
            
            return insights
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}", "user_id": user_id}
    
    def generate_smart_budget(self, user_id: str, target_savings_percentage: float = 20.0) -> Dict[str, Any]:
        """
        Generate an intelligent budget plan based on user's historical data and profile.
        
        Args:
            user_id: User identifier
            target_savings_percentage: Target savings as percentage of income
        
        Returns:
            Detailed budget plan
        """
        
        expenses = self.fetch_user_expenses(user_id, 60)  # 2 months of data
        user_profile = self.fetch_user_profile(user_id)
        
        prompt = f"""
        {self.financial_insights_prompt}
        
        Create an intelligent budget plan based on historical spending and user profile:
        
        HISTORICAL EXPENSES (60 days):
        {json.dumps(expenses, indent=2)}
        
        USER PROFILE:
        {json.dumps(user_profile, indent=2)}
        
        TARGET SAVINGS: {target_savings_percentage}%
        
        Generate budget plan in this exact JSON format:
        {{
            "budget_overview": {{
                "monthly_income": 0,
                "target_savings_amount": 0,
                "available_for_expenses": 0,
                "current_avg_monthly_spending": 0
            }},
            "category_budgets": {{
                "Food": {{"suggested_budget": 0, "current_avg": 0, "adjustment": "increase/decrease/maintain", "reasoning": "explanation"}},
                "Grocery": {{"suggested_budget": 0, "current_avg": 0, "adjustment": "increase/decrease/maintain", "reasoning": "explanation"}},
                "Transport": {{"suggested_budget": 0, "current_avg": 0, "adjustment": "increase/decrease/maintain", "reasoning": "explanation"}}
            }},
            "budget_strategy": {{
                "priority_cuts": ["Category where cuts are most needed"],
                "acceptable_increases": ["Category where increases are acceptable"],
                "optimization_tips": ["Specific tip 1", "Specific tip 2"]
            }},
            "implementation_plan": {{
                "week_1": "Focus area for week 1",
                "week_2": "Focus area for week 2", 
                "month_1_goal": "Primary goal for month 1",
                "success_metrics": ["Metric 1", "Metric 2"]
            }},
            "risk_assessment": {{
                "budget_feasibility": "high/medium/low",
                "potential_challenges": ["Challenge 1", "Challenge 2"],
                "mitigation_strategies": ["Strategy 1", "Strategy 2"]
            }}
        }}
        """
        
        try:
            msg = HumanMessage(content=[{"type": "text", "text": prompt}])
            response = self.llm.invoke([msg])
            parser = JsonOutputParser()
            
            budget = parser.parse(response.content)
            budget["user_id"] = user_id
            budget["generated_at"] = datetime.now().isoformat()
            
            return budget
            
        except Exception as e:
            return {"error": f"Budget generation failed: {str(e)}", "user_id": user_id}
    
    def detect_spending_anomalies(self, user_id: str) -> Dict[str, Any]:
        """
        Detect unusual spending patterns and generate alerts.
        
        Args:
            user_id: User identifier
        
        Returns:
            Anomaly detection results and alerts
        """
        
        recent_expenses = self.fetch_user_expenses(user_id, 30)
        historical_expenses = self.fetch_user_expenses(user_id, 90)
        
        prompt = f"""
        {self.financial_insights_prompt}
        
        Analyze spending data to detect anomalies and unusual patterns:
        
        RECENT EXPENSES (30 days):
        {json.dumps(recent_expenses, indent=2)}
        
        HISTORICAL COMPARISON (90 days):
        {json.dumps(historical_expenses, indent=2)}
        
        Detect anomalies and provide alerts in this JSON format:
        {{
            "anomaly_summary": {{
                "total_anomalies_detected": 0,
                "high_priority_alerts": 0,
                "spending_trend": "increasing/decreasing/stable"
            }},
            "detected_anomalies": [
                {{
                    "type": "unusual_spike/unusual_frequency/category_shift",
                    "category": "category_name",
                    "description": "What was detected",
                    "severity": "high/medium/low",
                    "recommendation": "What user should do"
                }}
            ],
            "spending_trends": {{
                "month_over_month_change": 0,
                "category_trend_changes": {{
                    "Food": {{"trend": "up/down/stable", "change_percentage": 0}},
                    "Transport": {{"trend": "up/down/stable", "change_percentage": 0}}
                }}
            }},
            "recommendations": [
                "Specific recommendation based on detected patterns",
                "Action item based on anomalies"
            ]
        }}
        """
        
        try:
            msg = HumanMessage(content=[{"type": "text", "text": prompt}])
            response = self.llm.invoke([msg])
            parser = JsonOutputParser()
            
            anomalies = parser.parse(response.content)
            anomalies["user_id"] = user_id
            anomalies["analysis_date"] = datetime.now().isoformat()
            
            return anomalies
            
        except Exception as e:
            return {"error": f"Anomaly detection failed: {str(e)}", "user_id": user_id}
    
    def generate_financial_report(self, user_id: str, report_type: str = "monthly") -> Dict[str, Any]:
        """
        Generate comprehensive financial report for user.
        
        Args:
            user_id: User identifier
            report_type: "weekly", "monthly", "quarterly"
        
        Returns:
            Complete financial report
        """
        
        period_days = {"weekly": 7, "monthly": 30, "quarterly": 90}
        days = period_days.get(report_type, 30)
        
        # Get comprehensive insights
        insights = self.analyze_comprehensive_insights(user_id, days)
        
        # Get budget analysis
        budget = self.generate_smart_budget(user_id)
        
        # Get anomaly detection
        anomalies = self.detect_spending_anomalies(user_id)
        
        # Combine into comprehensive report
        report = {
            "report_metadata": {
                "user_id": user_id,
                "report_type": report_type,
                "period_days": days,
                "generated_at": datetime.now().isoformat()
            },
            "financial_insights": insights,
            "budget_analysis": budget,
            "anomaly_detection": anomalies,
            "executive_summary": {
                "overall_financial_health": insights.get("financial_health_score", {}).get("overall_score", 0),
                "top_3_priorities": [
                    "Priority extracted from analysis",
                    "Priority extracted from budget",
                    "Priority extracted from anomalies"
                ],
                "key_achievements": [
                    "Positive trend identified",
                    "Good financial behavior noted"
                ]
            }
        }
        
        return report
