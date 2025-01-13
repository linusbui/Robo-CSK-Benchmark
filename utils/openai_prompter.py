import json
import os

from openai import OpenAI


class OpenAIPrompter:
    def __init__(self):
        json_text = json.load(open(os.path.join("./credentials.json")))
        self.client = OpenAI(
            api_key=json_text["api_key"],
        )
        self.model = "o1-2024-12-17"

    def prompt_openai_model(self, system_msg: str, user_msg: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=1.0,
        )
        return response.choices[0].message.content
