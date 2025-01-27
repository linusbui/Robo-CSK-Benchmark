import ast

import pandas as pd

from tidy_up.prompting.tidy_up_result import TidyUpResult
from utils.prompter import Prompter

system_msg = 'Imagine you are a robot tidying up a household.'
user_msg = 'What are the prototypical locations in a household where the following object can be found? Please only answer with a comma separated & ranked list of locations.'


def prompt_all_models(prompters: [Prompter]):
    comb_result = pd.DataFrame(columns=['model', 'mrr', 'map@1', 'map@3', 'map@5', 'mrec@1', 'mrec@3', 'mrec@5'])
    for prompter in prompters:
        data = pd.read_csv('tidy_up/tidy_up_data.csv', delimiter=',', on_bad_lines='skip')
        results = []
        for index, row in data.iterrows():
            obj = row['Object']
            locs = row['Locations']
            question = f'Object: {obj}\nLocations:'
            res = prompter.prompt_model(system_msg, user_msg, question)
            tup = TidyUpResult(obj, format_generated_locations(res), format_gold_standard_locations(locs))
            results.append(tup)
        write_results_to_file(results, prompter.model_name)
        comb_result = pd.concat([comb_result, calculate_average(results, prompter.model_name)], ignore_index=True)
    comb_result.to_csv('tidy_up/results/model_overview.csv', index=False)


def write_results_to_file(results: [TidyUpResult], model: str):
    dict_list = [re.to_dict() for re in results]
    df = pd.DataFrame(dict_list)
    df.to_csv(f'tidy_up/results/{model.lower()}.csv', index=False)


def format_generated_locations(locations: str) -> [str]:
    return [location.strip() for location in locations.lower().split(',')]


def format_gold_standard_locations(locations) -> [str]:
    data_dict = ast.literal_eval(locations)
    return [value[0].lower() for value in data_dict.values()]


def calculate_average(results: [TidyUpResult], model: str):
    average = {met: 0 for met in ['rr', 'ap@1', 'ap@3', 'ap@5', 'rec@1', 'rec@3', 'rec@5']}
    for res in results:
        average['rr'] += res.get_reciprocal_rank()
        average['ap@1'] += res.get_average_precision_at1()
        average['ap@3'] += res.get_average_precision_at3()
        average['ap@5'] += res.get_average_precision_at5()
        average['rec@1'] += res.get_recall_at1()
        average['rec@3'] += res.get_recall_at3()
        average['rec@5'] += res.get_recall_at5()
    new_row = pd.Series({'model': model, 'mrr': (average['rr'] / len(results)), 'map@1': (average['ap@1'] / len(results)),
                         'map@3': (average['ap@3'] / len(results)), 'map@5': (average['ap@5'] / len(results)),
                         'mrec@1': (average['rec@1'] / len(results)), 'mrec@3': (average['rec@3'] / len(results)),
                         'mrec@5': (average['rec@5'] / len(results))})
    return new_row.to_frame().T
