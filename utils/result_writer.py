from pathlib import Path
import os
import re
import pandas as pd

from utils.eval_result_super import ModelEvaluationResult


def add_to_model_overview(new_entry: pd.DataFrame, folder: str, lower: int, bound, default_res_folder=True):
    # Check for partial run
    if not (lower == 1 and bound == None):
        # encode end of dataset (bound = None) as -1, or return to upper
        if bound == None:
            upper = -1
        else:
            upper = bound + lower
        # add bounds to model name
        new_entry.loc[:, 'model'] = new_entry['model'].apply(lambda x: x + f'_{lower}_{upper}')
    if default_res_folder:
        file = f'{folder}/results/model_overview.csv'
    else:
        file = f'{folder}/model_overview.csv'
    if Path(file).exists():
        model_overview = pd.read_csv(file)
        # Remove old version if the entry already exists
        model = new_entry['model'].values
        model_overview = model_overview[~model_overview['model'].isin(model)]
        result = pd.concat([model_overview, new_entry], ignore_index=True)
    else:
        result = new_entry
    result.to_csv(file, index=False)


def write_model_results_to_file(results: [ModelEvaluationResult], model: str, tech: str, folder: str, lower: int, bound, default_res_folder=True):
    if default_res_folder:
        # create folder if needed
        resdir = f'{folder}/results/{model.lower()}'
        if not os.path.isdir(resdir):
            os.mkdir(resdir)
    else:
        # create folder if needed
        resdir = f'{folder}/{model.lower()}'
        if not os.path.isdir(resdir):
            os.mkdir(resdir)

    file = f'{resdir}/{model.lower()}_{tech}.csv'
    if not (lower == 1 and bound == None):
        # partial run
        # encode end of dataset (bound = None) as -1, or return to upper
        if bound == None:
            upper = -1
        else:
            upper = bound + lower
        file = f'{resdir}/{model.lower()}_{tech}_{lower}_{upper}.csv'
    dict_list = [re.to_dict() for re in results]
    df = pd.DataFrame(dict_list)
    df.to_csv(file, index=False)

    # Check if results of partial runs can be merged
    if not (lower == 1 and bound == None):
        # get (start, end) for all saved partial runs
        numbers = []
        pattern = re.compile(rf'{model.lower()}_{tech}_(\d+)_(-?\d+)')
        for file in os.listdir(resdir):
            match = pattern.match(file)
            if match:
                numbers.append((int(match.group(1)), int(match.group(2))))
        # merge if possible
        if check_merge(numbers):
            # save merged results
            sort = sorted(numbers, key=lambda tup: tup[0])
            dfs = []
            for pair in sort:
                dfs.append(pd.read_csv(f'{resdir}/{model.lower()}_{tech}_{pair[0]}_{pair[1]}.csv'))
            full = pd.concat(dfs, ignore_index=True)
            full.to_csv(f'{resdir}/{model.lower()}_{tech}.csv')
            # delete partial results
            for pair in sort:
                os.remove(f'{resdir}/{model.lower()}_{tech}_{pair[0]}_{pair[1]}.csv')


# For given list of pairs (x_i, y_i) check if continuous path with ending -1 exists
def check_merge(pairs):
    sort = sorted(pairs, key=lambda tup: tup[0])
    for i in range(len(sort)-1):
        if not sort[i][1] == sort[i+1][0]:
            return False
    return True and sort[-1][1] == -1