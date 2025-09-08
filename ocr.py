import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import base64
from langchain_core.output_parsers import JsonOutputParser 


# Initialize Gemini model via LangChain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_expense_from_image(image_path: str):
    # Convert image to base64 for sending
    img_b64 = image_to_base64(image_path)

    # Craft message
    msg = HumanMessage(content=[
        {"type": "text", "text": "Extract all details from this image and return structured JSON the following schema:- name of the product on which the expence was incured like if the reciept is about a purchase or service payment then extract the name of the thing on which the expence was incured (if single product then only one name, if multiple products then divide the json sepraterly based on the products and avoid any nested json output, ), amount, category(you need to categorize the expence based on the bill analysis, mode(choose the mode based on the analysis of the bill like hw the payment was payed like what was the mode of the payment))"},
        {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{img_b64}"
        }
    ])


    response = llm.invoke([msg])
    return response.content

parser = JsonOutputParser()




# Example usage
data = parser.parse(extract_expense_from_image("C:/Users/tohid/OneDrive/Desktop/ocr/bil.png"))

data