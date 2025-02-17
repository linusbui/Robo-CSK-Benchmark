import os
import openai
from openai import OpenAI

import json
import re


class OpenAIChatModel:
    def __init__(self, model_engine):
        self.model_engine = model_engine
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_text = json.load(open(os.path.join(script_dir, '../../utils/', 'credentials.json')))
        self.client = OpenAI(
            api_key=json_text["api_key"],
        )


    def _answer(self, instructions, prompt, old_messages=[]):
        messages = old_messages + [
            {"role": "system", "content": str(instructions)},
            {"role": "user", "content": str(prompt)}
        ]
        completion = self.client.chat.completions.create(
            model=self.model_engine,
            messages=messages,
              temperature=0)
        return completion.choices[0].message.content, messages

    def _extract_steps(self, text):
        first_step_pattern = r"First step:\s*(.*?)(?=Second step)"
        second_step_pattern = r"Second step:\s*(.*$)"
        first_step = re.search(first_step_pattern, text)
        second_step = re.search(second_step_pattern, text)
        first_step = first_step.group(1) if first_step else None
        second_step = second_step.group(1) if second_step else None
        return first_step, second_step

    def _get_order_questions(self, recipe_title, recipe_instructions):
        instructions = ("I will provide you with cooking instructions for the recipe " + recipe_title + ". "
                        "Extract the two most relevant steps from the cooking instructions."
                        "Shorten each step up to a maximum of 5 words. Remove exact time details. "
                        "Return the steps in the order they should be performed:\nFirst step: <first>. Second step: <second> ."
                        "Each step should be comprehensible independently of the other.")
        instructions_str = "; ".join(recipe_instructions)
        prompt = "Instructions: " + instructions_str
        response, old_messages = self._answer(instructions, prompt)

        print(response)
        step_1, step_2 = self._extract_steps(response)
        return step_1, step_2



# Function to extract the cooking instructions from a JSON file
def extract_instructions(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    instructions = []
    title = []
    for recipe in data:
        recipe_instructions = [step['text'] for step in recipe.get('instructions', [])]
        instructions.append(recipe_instructions)
        title.append(recipe.get('title', 'Unknown'))

    return instructions, title


def save_dict_to_file(save_dict, filename):
    # Save the data to a JSON file
    with open(filename, 'w') as file:
        json.dump(save_dict, file, indent=4)
    print(f"Dictionary saved to {filename}")


if __name__ == '__main__':
    chatgpt = OpenAIChatModel("gpt-4o-2024-08-06")
    final_num_order_questions = 200

    for recipe_number in range(1,5):
        json_file = 'Recipe1M+ Dataset/recipes_'+str(recipe_number)+'.json'
        instructions, title = extract_instructions(json_file)

        final_dict = []
        num_order_questions = 0

        for recipe_instructions, recipe_title in zip(instructions, title):
            if num_order_questions < final_num_order_questions:
                print(f"\nProcessing recipe: {recipe_title}")
                print("Number of order questions: ", num_order_questions)
                step_1, step_2 = chatgpt._get_order_questions(recipe_title, recipe_instructions)
                if step_1 and step_2:
                    num_order_questions += 1
                    recipe_dict = {
                        "goal": recipe_title,
                        "step_1": step_1,
                        "step_2": step_2
                        }
                    final_dict.append(recipe_dict)
        save_path = "question_components/questions_recipe_"+str(recipe_number)
        save_dict_to_file(final_dict, save_path + ".json")


        
