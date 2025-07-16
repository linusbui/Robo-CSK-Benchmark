import ast
import random

import pandas as pd
from tqdm import tqdm

from meta_reasoning.prompting.meta_reason_model_result import MetaReasoningMultiChoiceResult
from tidy_up.prompting.tidy_up_result import TidyUpMultiChoiceResult
from utils.prompter import Prompter
from utils.result_writer import add_to_model_overview, write_model_results_to_file

system_msg = 'Imagine you are to create a robot for a specific household task.'
user_msg = 'What is the single hardware configuration from the given list that you think is the most suitable to execute the task? Please only answer with the complete configuration you chose.'


def prompt_all_models(prompters: [Prompter]):
    for prompter in prompters:
        results = []
        questions = pd.read_csv('meta_reasoning/meta_reasoning_multi_questions.csv', delimiter=',', on_bad_lines='skip')
        questions['Wrong_Configurations'] = questions['Wrong_Configurations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Meta-Reasoning task'):
            task = row['Task']
            corr_conf = row['Correct_Configuration']
            choices = row['Wrong_Configurations'] + [corr_conf]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Task: {task}\nConfigurations: {choices_string}\nYour Choice:'
            res = prompter.prompt_model(system_msg, user_msg, question)
            tup = MetaReasoningMultiChoiceResult(task, corr_conf, res, choices)
            results.append(tup)
        write_model_results_to_file(results, prompter.model_name, 'meta_reasoning/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name), 'meta_reasoning/results_multi', False)


def calculate_average(results: [TidyUpMultiChoiceResult], model: str):
    average = {met: 0 for met in ['acc']}
    for res in results:
        if res.get_pred_correctness():
            average['acc'] += 1
    new_row = pd.Series(
        {'model': model, 'acc': (average['acc'] / len(results))})
    return new_row.to_frame().T
