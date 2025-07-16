import ast

import pandas as pd
from tqdm import tqdm

from tidy_up.prompting.tidy_up_result import TidyUpOpenResult
from utils.prompter import Prompter
from utils.result_writer import add_to_model_overview, write_model_results_to_file

system_msg = 'Imagine you are a robot tidying up a household.'
user_msg = 'What are the prototypical locations in a household where the following object can be found? Please only answer with a comma separated & ranked list of locations.'


def prompt_all_models(prompters: [Prompter]):
    for prompter in prompters:
        data = pd.read_csv('tidy_up/tidy_up_data.csv', delimiter=',', on_bad_lines='skip')
        results = []
        for index, row in tqdm(data.iterrows(), f'Prompting {prompter.model_name} for the open Tidy Up task'):
            obj = row['Object']
            locs = row['Locations']
            question = f'Object: {obj}\nLocations:'
            res = prompter.prompt_model(system_msg, user_msg, question)
            tup = TidyUpOpenResult(obj, format_generated_locations(res), format_gold_standard_locations(locs))
            if len(tup.get_predicted_locations()) == 0:
                print(f'Error formatting the generated locations for {obj}: {res}')
                continue
            results.append(tup)
        write_model_results_to_file(results, prompter.model_name)
        add_to_model_overview(calculate_average(results, prompter.model_name), 'tidy_up/results_open', False)


def format_generated_locations(locations: str) -> [str]:
    if locations is None:
        return []
    return [location.strip() for location in locations.lower().split(',')]


def format_gold_standard_locations(locations) -> [str]:
    data_dict = ast.literal_eval(locations)
    return [value[0].lower() for value in data_dict.values()]


def calculate_average(results: [TidyUpOpenResult], model: str):
    average = {met: 0 for met in ['rr', 'ap@1', 'ap@3', 'ap@5', 'rec@1', 'rec@3', 'rec@5']}
    for res in results:
        average['rr'] += res.get_reciprocal_rank()
        average['ap@1'] += res.get_average_precision_at1()
        average['ap@3'] += res.get_average_precision_at3()
        average['ap@5'] += res.get_average_precision_at5()
        average['rec@1'] += res.get_recall_at1()
        average['rec@3'] += res.get_recall_at3()
        average['rec@5'] += res.get_recall_at5()
    new_row = pd.Series(
        {'model': model, 'mrr': (average['rr'] / len(results)), 'map@1': (average['ap@1'] / len(results)),
         'map@3': (average['ap@3'] / len(results)), 'map@5': (average['ap@5'] / len(results)),
         'mrec@1': (average['rec@1'] / len(results)), 'mrec@3': (average['rec@3'] / len(results)),
         'mrec@5': (average['rec@5'] / len(results))})
    return new_row.to_frame().T
