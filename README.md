# Smart Expense Tracker with OCR and AI Insights

## Project Overview

This project is a comprehensive Smart Expense Tracker that leverages Optical Character Recognition (OCR) to extract expense details from receipts and bills, stores them, and provides advanced financial insights and a conversational AI chatbot. It's designed to help users manage their finances efficiently and gain a deeper understanding of their spending habits.

## Features

*   **OCR-based Expense Extraction:** Automatically extracts expense name, amount, category, and other details from uploaded images (e.g., JPG, PNG) and PDF documents.
*   **Secure User Authentication:** Robust user registration and login system with password hashing.
*   **Expense Management:** Full CRUD (Create, Read, Update, Delete) operations for managing individual expenses.
*   **AI-Powered Financial Insights:** Generates personalized financial summaries, spending breakdowns, budget suggestions, and anomaly detection using advanced AI models.
*   **Natural Language Chatbot:** Interact with an AI assistant to ask financial questions, get spending analysis, and receive recommendations through natural language.
*   **User Profile Management:** Allows users to set savings goals and financial objectives.
*   **Data Persistence:** Stores all user and expense data securely in a relational database (SQLite by default).
*   **RESTful API:** Provides a well-documented API for seamless integration with various front-end applications.

## Technologies Used

*   **Backend:** Python 3.12+
*   **Web Framework:** FastAPI
*   **Database:** SQLAlchemy (ORM) with SQLite (default), configurable for others.
*   **AI/LLM:** Google Gemini API via LangChain
*   **Data Validation:** Pydantic
*   **Security:** Passlib (for password hashing), Python-jose (for JWT)
*   **OCR/Document Processing:** PDFPlumber, OpenCV-Python
*   **Environment Management:** python-dotenv
*   **Database Migrations:** Alembic

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/smart-expense-tracker.git
    cd smart-expense-tracker
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv env
    # On Windows
    .\env\Scripts\activate
    # On macOS/Linux
    source env/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    Create a `.env` file in the project root directory with the following content. Replace placeholders with your actual values:
    ```env
    GOOGLE_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"
    SECRET_KEY="A_VERY_SECRET_KEY_FOR_JWT_SIGNING"
    DATABASE_URL="sqlite:///expenses.db" # Or your preferred database URL
    CHAT_MODEL_NAME="gemini-1.5-flash" # Or other Gemini models
    ```
    *   You can obtain a `GOOGLE_API_KEY` from the [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   `SECRET_KEY` should be a strong, random string.

## Usage

1.  **Initialize the Database:**
    Run the database migrations to create the necessary tables. If you are starting fresh, you can use:
    ```bash
    alembic upgrade head
    ```
    Alternatively, for initial setup, you can run:
    ```bash
    python database/core.py
    ```

2.  **Run the FastAPI Application:**
    Start the API server using Uvicorn:
    ```bash
    uvicorn api.main:app --reload
    ```
    The API will be accessible at `http://127.0.0.1:8000`.

3.  **Access API Documentation:**
    Once the server is running, you can access the interactive API documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.

4.  **Frontend (Optional):**
    A basic HTML frontend is provided in `Smart Expense Tracker/index.html`. You can open this file directly in your browser to interact with the API, or integrate with your own custom frontend application.

## Project Structure

```
.env
alembic.ini
requirements.txt
...

├───api/
│   ├───input_validation.py
│   └───main.py
├───database/
│   ├───core.py
│   ├───models.py
│   └───repo.py
├───pipeline/
│   ├───financial_insights.py
│   ├───nlp_chatbot.py
│   └───ocr_model.py
├───Smart Expense Tracker/
│   └───index.html
└───system_prompts/
    └───ocr_system_prompt.txt
```

*   `api/`: Contains the FastAPI application, including main routes and input validation schemas.
*   `database/`: Handles database interactions, SQLAlchemy models, and repository logic.
*   `pipeline/`: Houses the core business logic, including OCR extraction, financial insights generation, and NLP chatbot functionalities.
*   `alembic/`: Contains database migration scripts.
*   `Smart Expense Tracker/`: A simple HTML file for a basic frontend.
*   `system_prompts/`: Stores system prompts for AI models.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is open-source and available under the [MIT License](LICENSE). (You might want to create a LICENSE file in your repo)
