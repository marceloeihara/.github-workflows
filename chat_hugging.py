#pip install transformers
#pip install torch torchvision torchaudio
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
from transformers import AutoTokenizer, AutoModelForCausalLM
from chat_base import ChatPDFBase


class ChatPDF(ChatPDFBase):
    def __init__(self, model_name="microsoft/Phi-3.5-mini-instruct"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_response(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        output_ids = self.model.generate(input_ids, max_new_tokens=100)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(prompt,"").strip()