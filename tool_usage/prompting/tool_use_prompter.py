import ast
import random

import pandas as pd
from tqdm import tqdm

from tool_usage.prompting.tool_use_result import ToolSubstitutionResult
from utils.prompter import Prompter

system_msg = 'Imagine you are a robot in a household environment being confronted with a task and a list of tools.'
user_msg = 'What is the single tool from the given list that you think is most suitable to help you execute your task? Please only answer with the tool you chose.'


def prompt_all_models(prompters: [Prompter]):
    comb_result = pd.DataFrame(columns=['model', 'acc'])
    for prompter in prompters:
        results = []
        questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions.csv', delimiter=',', on_bad_lines='skip')
        questions['Wrong_Tools'] = questions['Wrong_Tools'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(), f'Prompting {prompter.model_name} for the Tool Usage task'):
            task = row['Task']
            affordance = row['Affordance']
            corr_tool = row['Correct_Tool']
            choices = row['Wrong_Tools'] + [corr_tool]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Task: {task}\nTools: {choices_string}\nYour Choice:'
            res = prompter.prompt_model(system_msg, user_msg, question)
            tup = ToolSubstitutionResult(task, affordance, corr_tool, res, choices)
            results.append(tup)
        write_results_to_file(results, prompter.model_name)
        comb_result = pd.concat([comb_result, calculate_average(results, prompter.model_name)], ignore_index=True)
    comb_result.to_csv('tool_usage/results/model_overview.csv', index=False)


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
