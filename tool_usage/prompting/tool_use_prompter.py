import ast
import json
import random

import pandas as pd

from tool_usage.prompting.tool_use_result import ToolSubstitutionResult
from utils.prompter import Prompter

system_msg = 'Imagine you are a robot in a household environment being confronted with a task and a list of tools.'
user_msg = 'What is the single tool from the given list that you think is most suitable to help you execute your task? Please only answer with the tool you chose.'

aff_data = pd.read_csv('tool_usage/affordance_data.csv', delimiter=',', on_bad_lines='skip')
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


def prompt_all_models(prompters: [Prompter]):
    comb_result = pd.DataFrame(columns=['model', 'acc'])
    for prompter in prompters:
        with open("tool_usage/affordance_task_map.json") as f:
            task_data = json.load(f)
        results = []
        for affordance, task in task_data.items():
            for t in task:
                corr_tool = get_tool_for_affordance(affordance)
                choices = get_unhelpful_tools_for_affordance(affordance) + [corr_tool]
                random.shuffle(choices)
                choices_string = ', '.join([c for c in choices])
                question = f'Task:{t}\nTools: {choices_string}\nYour Choice:'
                res = prompter.prompt_model(system_msg, user_msg, question)
                tup = ToolSubstitutionResult(t, affordance, corr_tool, res, choices)
                results.append(tup)
        write_results_to_file(results, prompter.model_name)
        comb_result = pd.concat([comb_result, calculate_average(results, prompter.model_name)],
                                ignore_index=True)
    comb_result.to_csv('tool_usage/results/model_overview.csv', index=False)


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


def write_results_to_file(results: [ToolSubstitutionResult], model: str):
    dict_list = [re.to_dict() for re in results]
    df = pd.DataFrame(dict_list)
    df.to_csv(f'tool_usage/results/{model.lower()}.csv', index=False)


def calculate_average(results: [ToolSubstitutionResult], model: str):
    average = {met: 0 for met in ['acc']}
    for res in results:
        if res.get_pred_correctness():
            average['acc'] += 1
    new_row = pd.Series(
        {'model': model, 'acc': (average['acc'] / len(results))})
    return new_row.to_frame().T


def execute_prompting(prompters: [Prompter]):
    preprocess_affordance_data()
    prompt_all_models(prompters)
