import ast
import random

import pandas as pd

obj_data = pd.read_csv('../tidy_up_data.csv', delimiter=',', on_bad_lines='skip')
loc_data = pd.read_csv('../tidy_up_data_reversed.csv', delimiter=',', on_bad_lines='skip')
locations = []


def preprocess_location_data():
    for idx, row in loc_data.iterrows():
        locations.append(row['Location'])


def turn_ranked_locations_into_list(corr_locs: str) -> [str]:
    parsed = ast.literal_eval(corr_locs)
    locs = [v[0] for v in parsed.values()]
    return locs


def get_wrong_locations_for_object(corr_locs: str, amount=4) -> [str]:
    choices = []
    while len(choices) < amount:
        pot_choice = random.choice(locations)
        locs = turn_ranked_locations_into_list(corr_locs)
        if pot_choice not in locs:
            choices.append(pot_choice)
    return choices


def create_multi_choice_questions():
    preprocess_location_data()
    question_data = pd.DataFrame(columns=['Object', 'Correct_Location', 'Wrong_Locations'])
    for idx, row in obj_data.iterrows():
        obj = row['Object']
        corr_locs = turn_ranked_locations_into_list(row['Locations'])
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
