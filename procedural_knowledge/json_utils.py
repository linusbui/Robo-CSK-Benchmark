import os
import json

def extract_json(file_path):
    with open(file_path, 'r') as file:
        return [{'goal': recipe.get('goal', 'No goal specified'), 'step_1': recipe.get('step_1', 'No step 1 specified'),
                 'step_2': recipe.get('step_2', 'No step 2 specified')} for recipe in json.load(file)]

def extract_json_multi(file_path):
    with open(file_path, 'r') as file:
        return [{'goal': recipe.get('goal', 'No goal specified'), 'step_1': recipe.get('step_1', 'No step 1 specified'),
                 'step_2': recipe.get('step_2', 'No step 2 specified'),
                 'step_3': recipe.get('step_3', 'No step 3 specified')} for recipe in json.load(file)]

def extract_results_json(file_path):
    with open(file_path, 'r') as file:
        return [
            {
                'title': entry.get('title', 'No title specified'),
                'question': entry.get('question', 'No question specified'),
                'response': entry.get('response', 'No response specified'),
                'correct_response': entry.get('correct_response', 'No correct response specified')
            }
            for entry in json.load(file)
        ]

def save_to_json(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)