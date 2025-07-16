import ast
import json
import random

import pandas as pd
from tqdm import tqdm

aff_data = pd.read_csv('../affordance_data.csv', delimiter=',', on_bad_lines='skip')
affordance_map = {}


def preprocess_affordance_data():
    for idx, row in aff_data.iterrows():
        aff_dict = ast.literal_eval(row['Affordances'])
        for idx2, aff_t in aff_dict.items():
            aff = aff_t[0]
            if aff not in affordance_map:
                affordance_map[aff] = [row['Object']]
            else:
                affordance_map[aff].append(row['Object'])


def get_tool_for_affordance(affordance: str) -> str:
    if affordance not in affordance_map:
        return affordance
    return random.choice(affordance_map[affordance])


def get_unhelpful_tools_for_affordance(affordance: str, amount=4) -> [str]:
    choices = []
    while len(choices) < amount:
        pot_choice = aff_data.sample(n=1)
        aff_dict = ast.literal_eval(pot_choice['Affordances'].iloc[0])
        if affordance not in aff_dict:
            choices.append(pot_choice['Object'].iloc[0])
    return choices


def create_multi_choice_questions():
    preprocess_affordance_data()
    question_data = pd.DataFrame(columns=['Task', 'Affordance', 'Correct_Tool', 'Wrong_Tools'])
    with open("../affordance_task_map.json") as f:
        task_data = json.load(f)
    for affordance, task in tqdm(task_data.items(), f'Creating multiple choice questions'):
        for t in task:
            corr_tool = get_tool_for_affordance(affordance)
            choices = get_unhelpful_tools_for_affordance(affordance)
            random.shuffle(choices)
            new_row = pd.DataFrame([{
                'Task': t,
                'Affordance': affordance,
                'Correct_Tool': corr_tool,
                'Wrong_Tools': choices
            }])
            question_data = pd.concat([question_data, new_row], ignore_index=True)
    question_data.to_csv('../tool_usage_multichoice_questions.csv', index=False)


if __name__ == '__main__':
    create_multi_choice_questions()
