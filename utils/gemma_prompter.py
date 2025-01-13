import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.prompter import Prompter


class GemmaPrompter(Prompter):
    def __init__(self):
        super().__init__("google/gemma-2-27b-it")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto",
                                                          torch_dtype=torch.bfloat16)

    def prompt_model(self, system_msg: str, user_msg: str, question: str) -> str:
        chat = [
            {"role": "user", "content": f"{system_msg}\n{user_msg}\n{question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        outputs = self.model.generate(input_ids=inputs.to(self.model.device), max_new_tokens=500)
        outputs = self.tokenizer.decode(outputs[0])
        return outputs
