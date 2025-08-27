import ast
import random

import pandas as pd
from tqdm import tqdm

from tidy_up.prompting.tidy_up_result import TidyUpMultiChoiceResult
from utils.prompter import Prompter
from utils.result_writer import add_to_model_overview, write_model_results_to_file

system_msg = 'Imagine you are a robot tidying up a household environment, being confronted with an object and a possible list of locations to put it.'
user_msg = 'What is the single location from the given list that you think is the most suitable place to put the object? Please only answer with the location you chose.'

def prompt_all_models(prompters: [Prompter]):
    for prompter in prompters:
        results = []
        questions = pd.read_csv('tidy_up/tidy_up_multichoice.csv', delimiter=',', on_bad_lines='skip')
        questions['Wrong_Locations'] = questions['Wrong_Locations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Tidy Up task'):
            obj = row['Object']
            corr_loc = row['Correct_Location']
            choices = row['Wrong_Locations'] + [corr_loc]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Object: {obj}\nLocations: {choices_string}\nYour Choice:'
            res = prompter.prompt_model(system_msg, user_msg, question)
            tup = TidyUpMultiChoiceResult(obj, corr_loc, res, choices)
            results.append(tup)
        write_model_results_to_file(results, prompter.model_name, 'tidy_up/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name), 'tidy_up/results_multi', False)


def calculate_average(results: [TidyUpMultiChoiceResult], model: str):
    average = {met: 0 for met in ['acc']}
    for res in results:
        if res.get_pred_correctness():
            average['acc'] += 1
    new_row = pd.Series(
        {'model': model, 'acc': (average['acc'] / len(results))})
    return new_row.to_frame().T