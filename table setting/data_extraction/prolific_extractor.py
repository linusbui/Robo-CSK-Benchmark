import json
import os
import re

import pandas as pd

from preferred_meal_setting import PreferredMealSetting


def _get_id_for_recipe(recipe: str) -> str:
    mapping = pd.read_csv("../data/recipe_id_mapping.csv", delimiter=',', on_bad_lines='skip')
    return mapping.loc[mapping['Name'] == recipe, 'Recipe1M+ ID'].values[0]


def get_user_preferred_settings() -> [PreferredMealSetting]:
    meals = []
    folder_path = "../data/"
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                user_id = data['id']
                for header, answer in data['answers'].items():
                    # get the recipe title and thus also the recipe ID
                    h_data = json.loads(header)
                    recipe_name = h_data['explanation']
                    if recipe_name is None:
                        continue
                    recipe_wo_brackets = re.sub(r'\s*\(.*?\)', '', recipe_name)
                    recipe_id = _get_id_for_recipe(recipe_wo_brackets)
                    # Get the meal if it already exists (same id & same user id)
                    meal = None
                    for m in meals:
                        if recipe_id == m.get_recipe_id() and user_id == m.get_user_id():
                            meal = m
                            meals.remove(meal)
                            break
                    if meal is None:
                        meal = PreferredMealSetting(user_id, recipe_wo_brackets, recipe_id)
                    # add the plate or the utensils
                    question = h_data['question']
                    if question == "What type of plate would you use for eating this meal? (SINGLE CHOICE)":
                        meal.add_plate(answer)
                    else:
                        meal.add_utensils(answer)

                    meals.append(meal)
    return meals


if __name__ == '__main__':
    res = get_user_preferred_settings()
    dict_list = [re.to_dict() for re in sorted(res, key=lambda r: r.get_meal())]
    df = pd.DataFrame(dict_list)
    df.to_csv('../prolific_user_data.csv', index=False)
