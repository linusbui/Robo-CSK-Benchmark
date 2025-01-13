import json
import os

from openai import OpenAI

from utils.prompter import Prompter


class OpenAIPrompter(Prompter):
    def __init__(self):
        super().__init__("o1-2024-12-17")
        json_text = json.load(open(os.path.join("./credentials.json")))
        self.client = OpenAI(
            api_key=json_text["api_key"],
        )

    def prompt_model(self, system_msg: str, user_msg: str, question: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
                {"role": "user", "content": question},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content
