import json
import os

import openai

from model_result import ModelResultTuple


class TidyUpPrompter:
    def __init__(self, model: str):
        json_text = json.load(open(os.path.join("../credentials.json")))
        openai.organization = json_text["organization"]
        openai.api_key = json_text["api_key"]
        self._current_model = model

    def set_model(self, new_model: str):
        self._current_model = new_model

    def check_models(self):
        response = openai.ChatCompletion.create(
            model=self._current_model,
            messages=[
                {"role": "system", "content": "Are you still available?"}
            ],
            temperature=0,
        )
        print(f'The model is available: {response}')

    def prompt_for_locations(self, obj: str) -> ModelResultTuple:
        response = openai.ChatCompletion.create(
            model=self._current_model,
            messages=[
                {"role": "system", "content": "Imagine you are a cognitive robot interacting in an household "
                                              "environment. You are tasked with tidying up your environment. To do "
                                              "so, you need to collect different objects and bring them to the most "
                                              "suitable location."},
                {"role": "system", "content": "In the following I will tell you one household object and you should "
                                              "return an arbitrary number of locations in a typical household, "
                                              "where you would expect this object to reside. These locations can be "
                                              "rooms (e.g. kitchen) or specific parts of that room (e.g. fridge). "
                                              "Please only answer with a comma-separated list of locations, "
                                              "ranked according to their prototypicality. "},
                {"role": "user", "content": f"Object: {obj}\nLocations:"}
            ],
            temperature=0,
        )
        answer = response['choices'][0]['message']['content']
        locations = answer.split(', ')
        return ModelResultTuple(obj, locations)


if __name__ == "__main__":
    prompter = TidyUpPrompter("gpt-3.5-turbo")
    res = prompter.prompt_for_locations("fork")
    print(res)
