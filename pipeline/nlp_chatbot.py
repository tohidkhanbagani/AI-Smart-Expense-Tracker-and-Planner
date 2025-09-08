# nlp_chatbot.py

from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain.schema import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
import re
from pipeline.financial_insights import FinancialInsightsAnalyzer

load_dotenv()

class FinancialNLPChatbot:
    """
    Advanced NLP Chatbot for financial queries that processes natural language 
    input and returns structured JSON responses with financial insights.
    """
    
    def __init__(self, model_name: str = "gemini-1.5-flash", db_connection_string: str = None):
        self.model_name = model_name
        
        # Initialize LLM with same setup as other models
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Initialize financial insights analyzer for data operations
        self.insights_analyzer = FinancialInsightsAnalyzer(db_connection_string=db_connection_string)
        
        # Intent classification categories
        self.intent_categories = [
            "spending_analysis", "budget_planning", "financial_health", 
            "savings_advice", "expense_breakdown", "spending_alerts", 
            "comparison_analysis", "general_inquiry", "recommendation_request"
        ]
        
        # System prompt for NLP processing
        self.nlp_system_prompt = """
You are an expert financial advisor AI chatbot that processes natural language queries about personal finance and expenses. Your role is to understand user intent, analyze their financial queries, and provide structured, actionable responses.

CORE CAPABILITIES:
- Intent classification and natural language understanding
- Financial data analysis and interpretation
- Personalized financial advice and recommendations
- Budget planning and optimization suggestions
- Expense pattern analysis and insights
- Financial health assessment and scoring

OUTPUT REQUIREMENTS:
- Return only valid JSON without explanations or markdown
- Classify user intent accurately
- Provide specific, actionable financial advice
- Include relevant data analysis when available
- Format responses for easy integration with frontend applications

INTENT CATEGORIES:
- spending_analysis: Questions about spending patterns, amounts, categories
- budget_planning: Requests for budget creation, optimization, planning
- financial_health: Queries about overall financial wellness, scores, assessment
- savings_advice: Questions about saving money, reducing expenses
- expense_breakdown: Requests for detailed expense categorization
- spending_alerts: Questions about unusual spending, alerts, warnings
- comparison_analysis: Comparing spending across periods or categories
- general_inquiry: General financial questions, education
- recommendation_request: Specific requests for financial recommendations

FINANCIAL CATEGORIES (use only these):
["Food", "Grocery", "Transport", "Fuel", "Travel", "Utilities", "Rent", "Health", "Pharmacy", "Education", "Entertainment", "Shopping", "Electronics", "Home", "Services", "Subscriptions", "Fees", "Taxes", "Office", "Misc"]

RESPONSE PRINCIPLES:
- Be conversational but professional
- Provide specific, actionable advice
- Include relevant data when available
- Maintain encouraging and supportive tone
- Focus on achievable financial goals

CONSTRAINTS:
- Use only the data provided; do not ask to connect bank accounts.
- If details are missing, proceed with what is available and clearly note assumptions.

"""
    
    def classify_intent(self, user_query: str, user_id: str = None) -> Dict[str, Any]:
        """
        Classify user's intent and extract relevant information from natural language query.
        
        Args:
            user_query: Natural language query from user
            user_id: Optional user identifier
        
        Returns:
            Dictionary containing intent classification and extracted information
        """
        
        prompt = f"""
        {self.nlp_system_prompt}
        
        Analyze the following user query and classify the intent with extracted information:
        
        USER QUERY: "{user_query}"
        
        Classify intent and extract information in this exact JSON format:
        {{
            "primary_intent": "one of the intent categories",
            "confidence_score": 0.95,
            "extracted_entities": {{
                "time_period": "30 days/this month/last week/etc or null",
                "categories_mentioned": ["Food", "Transport"],
                "amount_mentioned": 1000.0,
                "comparison_type": "month-to-month/category/period or null"
            }},
            "query_complexity": "simple/moderate/complex",
            "requires_data_fetch": true,
            "suggested_analysis_period": 30,
            "response_type": "analysis/advice/data/planning",
            "user_sentiment": "positive/neutral/concerned/confused",
            "key_topics": ["spending", "budget", "savings"],
            "actionable_request": "specific action user wants"
        }}
        """
        
        try:
            msg = HumanMessage(content=[{"type": "text", "text": prompt}])
            response = self.llm.invoke([msg])
            parser = JsonOutputParser()
            
            intent_data = parser.parse(response.content)
            intent_data["user_id"] = user_id
            intent_data["processed_at"] = datetime.now().isoformat()
            
            return intent_data
            
        except Exception as e:
            return {
                "error": f"Intent classification failed: {str(e)}",
                "user_query": user_query,
                "fallback_intent": "general_inquiry"
            }
    
    def process_query(self, user_query: str, user_id: str) -> Dict[str, Any]:
        """
        Process natural language query and return comprehensive response with financial insights.
        
        Args:
            user_query: Natural language query from user
            user_id: User identifier for data fetching
        
        Returns:
            Structured JSON response with financial insights and recommendations
        """
        
        # First classify the intent
        intent_data = self.classify_intent(user_query, user_id)
        
        if "error" in intent_data:
            return intent_data
        
        # Fetch user data if required
        user_data = {}
        if intent_data.get("requires_data_fetch", False):
            analysis_period = intent_data.get("suggested_analysis_period", 30)
            user_expenses = self.insights_analyzer.fetch_user_expenses(user_id, analysis_period)
            user_profile = self.insights_analyzer.fetch_user_profile(user_id)
            
            user_data = {
                "expenses": user_expenses,
                "profile": user_profile,
                "expense_count": len(user_expenses)
            }
        
        # Generate contextual response based on intent
        response_data = self.generate_contextual_response(
            user_query, intent_data, user_data, user_id
        )
        
        return response_data
    
    def generate_contextual_response(self, user_query: str, intent_data: Dict, 
                                   user_data: Dict, user_id: str) -> Dict[str, Any]:
        """
        Generate contextual response based on classified intent and available data.
        
        Args:
            user_query: Original user query
            intent_data: Classified intent information
            user_data: User's financial data
            user_id: User identifier
        
        Returns:
            Comprehensive response with insights and recommendations
        """
        
        primary_intent = intent_data.get("primary_intent", "general_inquiry")
        
        # Prepare comprehensive prompt based on intent
        prompt = f"""
        {self.nlp_system_prompt}
        
        Process the user's financial query and provide a comprehensive, helpful response:
        
        USER QUERY: "{user_query}"
        
        INTENT ANALYSIS:
        {json.dumps(intent_data, indent=2)}
        
        USER FINANCIAL DATA:
        {json.dumps(user_data, indent=2)}
        
        Based on the intent "{primary_intent}", provide response in this exact JSON format:
        {{
            "response_metadata": {{
                "intent": "{primary_intent}",
                "confidence": 0.95,
                "response_type": "analysis/advice/data/planning",
                "processing_time": "2024-08-30T20:30:00"
            }},
            "conversational_response": "Natural language response to user's query",
            "financial_analysis": {{
                "key_findings": [
                    "Finding 1 based on user's data",
                    "Finding 2 based on user's data"
                ],
                "relevant_metrics": {{
                    "total_spending": 0,
                    "category_breakdown": {{}},
                    "spending_trend": "increasing/stable/decreasing"
                }},
                "insights": [
                    "Specific insight about user's finances",
                    "Pattern identified in spending"
                ]
            }},
            "actionable_recommendations": [
                {{
                    "recommendation": "Specific action user should take",
                    "category": "budgeting/saving/spending",
                    "impact": "high/medium/low",
                    "timeframe": "immediate/short-term/long-term",
                    "expected_benefit": "Specific benefit user will get"
                }}
            ],
            "follow_up_suggestions": [
                "Suggested follow-up question 1",
                "Suggested follow-up question 2"
            ],
            "alerts_and_warnings": [
                {{
                    "type": "budget/spending/savings",
                    "message": "Alert message",
                    "severity": "high/medium/low",
                    "action_required": "What user should do"
                }}
            ],
            "data_visualization_suggestions": [
                "Chart type that would help user understand their finances",
                "Graph showing spending trends"
            ]
        }}
        """
        
        try:
            msg = HumanMessage(content=[{"type": "text", "text": prompt}])
            response = self.llm.invoke([msg])
            parser = JsonOutputParser()
            
            chatbot_response = parser.parse(response.content)
            chatbot_response["user_id"] = user_id
            chatbot_response["original_query"] = user_query
            chatbot_response["generated_at"] = datetime.now().isoformat()
            
            return chatbot_response
            
        except Exception as e:
            return {
                "error": f"Response generation failed: {str(e)}",
                "user_query": user_query,
                "intent": primary_intent,
                "fallback_response": {
                    "conversational_response": "I'm having trouble processing your request right now. Please try rephrasing your question or contact support.",
                    "suggestions": [
                        "How much did I spend this month?",
                        "Show me my budget breakdown",
                        "Give me savings recommendations"
                    ]
                }
            }
    
    def handle_multi_turn_conversation(self, conversation_history: List[Dict], 
                                     current_query: str, user_id: str) -> Dict[str, Any]:
        """
        Handle multi-turn conversations with context awareness.
        
        Args:
            conversation_history: Previous conversation turns
            current_query: Current user query
            user_id: User identifier
        
        Returns:
            Response considering conversation context
        """
        
        # Prepare conversation context
        context_prompt = f"""
        {self.nlp_system_prompt}
        
        Process the current query considering the conversation history:
        
        CONVERSATION HISTORY:
        {json.dumps(conversation_history, indent=2)}
        
        CURRENT QUERY: "{current_query}"
        
        Provide contextual response considering previous conversation in this JSON format:
        {{
            "contextual_response": {{
                "acknowledges_history": true,
                "references_previous": "Reference to previous conversation",
                "conversational_response": "Natural response considering context"
            }},
            "continued_analysis": {{
                "builds_on_previous": "How this builds on previous discussion",
                "new_insights": ["New insight 1", "New insight 2"],
                "comparative_analysis": "How current query relates to previous"
            }},
            "conversation_suggestions": [
                "Natural follow-up question",
                "Related topic to explore"
            ],
            "session_summary": {{
                "topics_covered": ["topic1", "topic2"],
                "key_decisions": ["decision1", "decision2"],
                "action_items": ["action1", "action2"]
            }}
        }}
        """
        
        try:
            msg = HumanMessage(content=[{"type": "text", "text": context_prompt}])
            response = self.llm.invoke([msg])
            parser = JsonOutputParser()
            
            contextual_response = parser.parse(response.content)
            
            # Also get fresh analysis for current query
            current_response = self.process_query(current_query, user_id)
            
            # Combine contextual and current responses
            combined_response = {
                "conversation_context": contextual_response,
                "current_analysis": current_response,
                "response_type": "multi_turn",
                "user_id": user_id,
                "processed_at": datetime.now().isoformat()
            }
            
            return combined_response
            
        except Exception as e:
            return {
                "error": f"Multi-turn processing failed: {str(e)}",
                "fallback_to_single_turn": self.process_query(current_query, user_id)
            }
    
    def extract_financial_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract financial entities from natural language text.
        
        Args:
            text: Natural language text
        
        Returns:
            Dictionary of extracted financial entities
        """
        
        # Regular expressions for common financial entities
        amount_pattern = r'(?:₹|Rs\.?|INR)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        category_pattern = r'\b(food|grocery|transport|fuel|travel|utilities|rent|health|pharmacy|education|entertainment|shopping|electronics|home|services|subscriptions|fees|taxes|office)\b'
        time_pattern = r'\b(today|yesterday|this week|last week|this month|last month|this year|last year|\d+ days?|\d+ weeks?|\d+ months?)\b'
        
        entities = {
            "amounts": re.findall(amount_pattern, text.lower()),
            "categories": re.findall(category_pattern, text.lower()),
            "time_references": re.findall(time_pattern, text.lower()),
            "contains_question": "?" in text,
            "contains_comparison": any(word in text.lower() for word in ["compare", "vs", "versus", "difference", "more than", "less than"]),
            "urgency_indicators": any(word in text.lower() for word in ["urgent", "immediately", "asap", "quickly", "emergency"])
        }
        
        return entities
    
    def generate_quick_response(self, user_query: str, user_id: str) -> Dict[str, Any]:
        """
        Generate quick response for simple queries without full analysis.
        
        Args:
            user_query: User's natural language query
            user_id: User identifier
        
        Returns:
            Quick response dictionary
        """
        
        entities = self.extract_financial_entities(user_query)
        
        prompt = f"""
        {self.nlp_system_prompt}
        
        Provide a quick, helpful response to this financial query:
        
        USER QUERY: "{user_query}"
        EXTRACTED ENTITIES: {json.dumps(entities, indent=2)}
        
        Generate quick response in this JSON format:
        {{
            "quick_response": "Direct answer to user's question",
            "response_type": "quick/simple",
            "confidence": 0.85,
            "requires_data": false,
            "suggested_actions": ["action1", "action2"],
            "follow_up_options": [
                "Get detailed analysis",
                "See full breakdown",
                "Get recommendations"
            ]
        }}
        """
        
        try:
            msg = HumanMessage(content=[{"type": "text", "text": prompt}])
            response = self.llm.invoke([msg])
            parser = JsonOutputParser()
            
            quick_response = parser.parse(response.content)
            quick_response["user_id"] = user_id
            quick_response["generated_at"] = datetime.now().isoformat()
            
            return quick_response
            
        except Exception as e:
            return {
                "error": f"Quick response failed: {str(e)}",
                "fallback_response": "I understand you're asking about your finances. Could you please rephrase your question more specifically?",
                "user_query": user_query
            }
    

    def extract_expenses_from_text(self, text: str) -> Dict[str, Any]:
        """
        Convert natural language text into structured expenses compatible with insert_expenses.
        Output shape: {"items":[{bill_no, expence_name, amount, category, mode, date?}]}
        """
        schema_prompt = """
    You convert casual text into structured expenses for tracking.
    Return ONLY valid JSON with an array under key "items".
    Each item fields:
    - bill_no: string or null
    - expence_name: string
    - amount: number
    - category: one of ["Food","Grocery","Transport","Fuel","Travel","Utilities","Rent","Health","Pharmacy","Education","Entertainment","Shopping","Electronics","Home","Services","Subscriptions","Fees","Taxes","Office","Misc"]
    - mode: one of ["cash","card","upi","wallet","netbanking","bank_transfer","cheque","credit","other"]
    - date: YYYY-MM-DD or null

    Rules:
    - Infer category/mode when possible; else category="Misc", mode="other".
    - bill_no may be null; amount must be positive.
    - Multiple expenses => multiple items.
    - Use ISO date if present; relative words (yesterday/last week) may be left null if unsure.
    Never ask for bank connections; rely on input text only.
    """
        example = """
    Input: "Bought petrol ₹1500 via UPI yesterday; lunch 350 cash; Netflix 499 card on 2024-08-10"
    Output:
    {"items":[
    {"bill_no": null, "expence_name":"Petrol", "amount":1500, "category":"Fuel", "mode":"upi", "date": null},
    {"bill_no": null, "expence_name":"Lunch", "amount":350, "category":"Food", "mode":"cash", "date": null},
    {"bill_no": null, "expence_name":"Netflix", "amount":499, "category":"Subscriptions", "mode":"card", "date":"2024-08-10"}
    ]}
    """
        try:
            msg = HumanMessage(content=[{"type":"text","text": f"{schema_prompt}\n\nInput:\n{text}\n\nOutput JSON only."}])
            response = self.llm.invoke([msg])
            parser = JsonOutputParser()
            out = parser.parse(response.content)
            items = out.get("items") if isinstance(out, dict) else None
            return {"items": items or []}
        except Exception as e:
            return {"items": [], "error": f"parse_failed: {str(e)}"}