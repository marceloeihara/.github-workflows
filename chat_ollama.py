from langchain.chat_models import ChatOllama
from chat_base import ChatPDFBase

class ChatPDF(ChatPDFBase):
    def __init__(self, model_name="llama3"):
        super().__init__()
        self.model = ChatOllama(model=model_name)

    def generate_response(self, prompt):
        return self.model.predict(prompt)