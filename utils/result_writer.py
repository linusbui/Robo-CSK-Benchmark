from pathlib import Path
import os

import pandas as pd

from utils.eval_result_super import ModelEvaluationResult


def add_to_model_overview(new_entry: pd.DataFrame, folder: str, default_res_folder=True):
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


def write_model_results_to_file(results: [ModelEvaluationResult], model: str, tech: str, folder: str, default_res_folder=True):
    if default_res_folder:
        # create folder if needed
        resdir = f'{folder}/results/{model.lower()}'
        if not os.path.isdir(resdir):
            os.mkdir(resdir)
        file = f'{resdir}/{model.lower()}_{tech}.csv'
    else:
        # create folder if needed
        resdir = f'{folder}/{model.lower()}'
        if not os.path.isdir(resdir):
            os.mkdir(resdir)
        file = f'{resdir}/{model.lower()}_{tech}.csv'
    dict_list = [re.to_dict() for re in results]
    df = pd.DataFrame(dict_list)
    df.to_csv(file, index=False)
