import torch
import transformers

from utils.prompter import Prompter


class LlamaPrompter(Prompter):
    def __init__(self, max_new_tokens=10):
        super().__init__("Llama-3.3-70B-Instruct")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(f'meta-llama/{self.model_name}')
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # for open-ended generation
        self.max_new_tokens = max_new_tokens
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_id = transformers.AutoModelForCausalLM.from_pretrained(
            f'meta-llama/{self.model_name}',
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.generation_pipe = transformers.pipeline(
            "text-generation",
            model=model_id,
            tokenizer=self.tokenizer,
            trust_remote_code=True,
            device_map="auto",  # finds GPU
        )

    def prompt_model(self, system_msg: str, user_msg: str, question: str) -> str:
        # Instruct version needs specific conversation structure
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
            max_new_tokens=self.max_new_tokens,
            eos_token_id=terminators,
            do_sample=False,  # No randomness
            temperature=None,
            top_p=None
        )
        result = outputs[0]["generated_text"][len(prompt):]
        return result
