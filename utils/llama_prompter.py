import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

from utils.prompter import Prompter


class LlamaPrompter(Prompter):
    def __init__(self):
        super().__init__("meta-llama/Llama-3.3-70B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # for open-ended generation
        self.prompt_specifiers = ["<s>[INST]<<SYS>>", "<</SYS>>",
                                  "[/INST]"]  # System instructions specifiers for Llama models

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_id = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.generation_pipe = pipeline(
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
        prompt = self.generation_pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        terminators = [
            self.generation_pipe.tokenizer.eos_token_id,
            self.generation_pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = self.generation_pipe(
            prompt,
            max_new_tokens=2000,
            eos_token_id=terminators,
            do_sample=False,  # No randomness
            top_p=1,
        )
        result = outputs[0]["generated_text"][len(prompt):]
        return result
