from langchain_google_genai import ChatGoogleGenerativeAI
import os
import base64
import cv2
from langchain.schema import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import pdfplumber
import mimetypes



load_dotenv()

class ExpenseExtractor:
    def __init__(self, model_name: str = "gemini-1.5-flash", system_prompt_file: str = "system_prompts/ocr_system_prompt.txt"):
        self.model_name = model_name
        self.system_prompt_file = system_prompt_file

        # ✅ Pass API key explicitly
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        with open(self.system_prompt_file, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    def image_to_base64(self, image_input):
        if isinstance(image_input, str):  # file path
            with open(image_input, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(image_input, (bytes, bytearray)):
            return base64.b64encode(image_input).decode("utf-8")
        else:  # assume numpy array (cv2)
            _, buffer = cv2.imencode(".jpg", image_input)
            return base64.b64encode(buffer).decode("utf-8")

    def pdf_to_text(self, pdf_path: str) -> str:
        """Extract raw text from a PDF file using pdfplumber."""
        text_content = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text_content.append(page.extract_text() or "")
        return "\n".join(text_content).strip()


    def extract_expense(self, input_data, input_type=None):
        """
        Extract structured expense data.
        Args:
            input_data: image file path, numpy image matrix, or PDF file path
            input_type: "image" or "pdf" (optional, auto-detected if None)
        """
        # Auto-detect input_type if not given
        if input_type is None and isinstance(input_data, str):
            mime_type, _ = mimetypes.guess_type(input_data)
            if mime_type == "application/pdf":
                input_type = "pdf"
            else:
                input_type = "image"
        elif input_type is None:
            input_type = "image"

        if input_type == "image":
            img_b64 = self.image_to_base64(input_data)
            msg = HumanMessage(content=[
                {"type": "text", "text": self.system_prompt},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}
            ])

        elif input_type == "pdf":
            pdf_text = self.pdf_to_text(input_data)
            msg = HumanMessage(content=[
                {"type": "text", "text": f"{self.system_prompt}\n\nHere is the extracted PDF text:\n{pdf_text}"}
            ])

        else:
            raise ValueError("Invalid input_type. Use 'image' or 'pdf'.")

        response = self.llm.invoke([msg])
        parser = JsonOutputParser()
        return parser.parse(response.content)

