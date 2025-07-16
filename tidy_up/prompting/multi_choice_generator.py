import ast
import random
from typing import List

import pandas as pd
from tqdm import tqdm

obj_data = pd.read_csv('../tidy_up_data.csv', delimiter=',', on_bad_lines='skip')
loc_data = pd.read_csv('../tidy_up_data_reversed.csv', delimiter=',', on_bad_lines='skip')
locations = []


def preprocess_location_data():
    for idx, row in loc_data.iterrows():
        locations.append(row['Location'])


def get_highest_ranked_locations(corr_locs: str) -> List[str]:
    parsed = ast.literal_eval(corr_locs)
    max_value = max(val[1] for val in parsed.values())
    return [name for _, (name, value) in parsed.items() if value == max_value]


def get_wrong_locations_for_object(corr_locs: str, amount=4) -> List[str]:
    choices = []
    while len(choices) < amount:
        pot_choice = random.choice(locations)
        locs = [v[0] for v in ast.literal_eval(corr_locs).values()]
        if pot_choice not in locs:
            choices.append(pot_choice)
    return choices


def create_multi_choice_questions():
    preprocess_location_data()
    question_data = pd.DataFrame(columns=['Object', 'Correct_Location', 'Wrong_Locations'])
    for idx, row in tqdm(obj_data.iterrows(), 'Creating multiple choice questions'):
        obj = row['Object']
        corr_locs = get_highest_ranked_locations(row['Locations'])
        chosen_corr_loc = random.choice(corr_locs)
        choices = get_wrong_locations_for_object(row['Locations'])
        new_row = pd.DataFrame([{
            'Object': obj,
            'Correct_Location': chosen_corr_loc,
            'Wrong_Locations': choices
        }])
        question_data = pd.concat([question_data, new_row], ignore_index=True)
    question_data.to_csv('../tidy_up_multichoice.csv', index=False)


if __name__ == '__main__':
    create_multi_choice_questions()
