import torch
import transformers

from utils.prompter import Prompter


class LlamaPrompter(Prompter):
    def __init__(self, max_new_tokens=10):
        super().__init__("meta-llama/Llama-3.3-70B-Instruct")
        self.max_new_tokens = max_new_tokens
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def prompt_model(self, system_msg: str, user_msg: str, question: str) -> str:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "user", "content": question},
        ]
        outputs = self.pipeline(
            messages,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,  # No randomness if False
            temperature=self.temperature,
            top_p=None
        )
        result = next(item['content'] for item in outputs[0]['generated_text'] if item['role'] == 'assistant')
        return result
