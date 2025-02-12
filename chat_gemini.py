#pip install google.generativeai
#pip install python-dotenv
#pip install grpcio==1.66.1
#pip install protobuf==3.20.2
import os
from dotenv import load_dotenv
import google.generativeai as genai
from chat_base import ChatPDFBase

load_dotenv()

class ChatPDF(ChatPDFBase):
    def __init__(self):
        super().__init__()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("A chave da API do Gemini não foi encontrada. Verifique o arquivo .env.")
        genai.configure(api_key)
        self.model_name = "gemini-1.5-flash"
        self.model = genai.GenerativeModel(self.model_name)

    def generate_response(self, prompt):
        response = self.model.generate_content(prompt)
        return response.text