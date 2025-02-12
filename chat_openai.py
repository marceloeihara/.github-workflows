#pip install openai
#pip install -U langchain-openai
#pip install openai --use-deprecated=legacy-resolver
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from chat_base import ChatPDFBase
from langchain.schema import HumanMessage

load_dotenv()

class ChatPDF(ChatPDFBase):
    def __init__(self):
        super().__init__()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("A chave da API do OpenAI não foi encontrada. Verifique o arquivo .env.")
        self.model_name = "gpt-4o-mini-2024-07-18"
        self.model = ChatOpenAI(model=self.model_name, temperature=0.7)

    def generate_response(self, prompt):
        messages = [HumanMessage(content=prompt)]
        # Gere a resposta
        response = self.model(messages)  # O método correto pode ser apenas `self.model(messages)`
        # Acesse o texto da resposta
        return response.content