import json
import os
import re

import pandas as pd

from preferred_meal_setting import PreferredMealSetting
from utensils_plates import Plate, Utensil


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
                try:
                    data = json.load(file)
                except json.decoder.JSONDecodeError:
                    print(f'DecoderError on file: {file_path}')
                    continue
                user_id = data['id']
                for header, answer in data['answers'].items():
                    # get the recipe title and thus also the recipe ID
                    h_data = json.loads(header)
                    recipe_name = h_data['explanation']
                    if recipe_name is None:
                        continue
                    recipe_wo_brackets = re.sub(r'\s*\(.*?\)', '', recipe_name)
                    recipe_id = _get_id_for_recipe(fix_prolific_data_errors(recipe_wo_brackets))
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


def fix_prolific_data_errors(recipe: str) -> str:
    if recipe == "Vegan Sushi Roll with Avocado (Carrot and Cucumber)":
        return "Vegan Sushi Roll with Avocado, Carrot and Cucumber"
    if recipe == "Eggplant Curry":
        return "Eggplant Curry (Indian)"
    if recipe == "Banh Mi":
        return "Banh Mi (Vietnamese Pulled Pork Sandwich)"
    if recipe == "Tom Kha Gai":
        return "Tom Kha Gai (Thai Spicy Coconut Soup)"
    return recipe


def combine_user_preferences(meals: [PreferredMealSetting]):
    mapping = pd.read_csv("../data/recipe_id_mapping.csv", delimiter=',', on_bad_lines='skip')
    result = pd.DataFrame(
        columns=['recipe_id', 'name', 'hands', 'tongs', 'knife', 'fork', 'skewer', 'chopsticks', 'spoon',
                 'dinner plate', 'dessert plate', 'bowl', 'coupe plate'])

    for idx, meal in mapping.iterrows():
        utensils = {uts: 0 for uts in Utensil}
        plates = {plt: 0 for plt in Plate}
        for m in meals:
            if m.get_recipe_id() != meal["Recipe1M+ ID"]:
                continue
            for u in Utensil:
                if m.needs_specific_utensil(u):
                    utensils[u] += 1
            plates[m.get_plate()] += 1
        new_row = pd.Series({"recipe_id": meal["Recipe1M+ ID"], "name": meal["Name"], "hands": utensils[Utensil.HANDS],
                             "tongs": utensils[Utensil.TONGS], "knife": utensils[Utensil.KNIFE],
                             "fork": utensils[Utensil.FORK], "skewer": utensils[Utensil.SKEWER],
                             "chopsticks": utensils[Utensil.CHOPS], "spoon": utensils[Utensil.SPOON],
                             "dinner plate": plates[Plate.DINNER], "dessert plate": plates[Plate.DESSERT],
                             "bowl": plates[Plate.BOWL], "coupe plate": plates[Plate.COUPE]})
        result = pd.concat([result, new_row.to_frame().T], ignore_index=True)
    result.to_csv('../combined_prolific_data.csv', index=False)


if __name__ == '__main__':
    res = get_user_preferred_settings()
    dict_list = [re.to_dict() for re in sorted(res, key=lambda r: r.get_recipe_id())]
    df = pd.DataFrame(dict_list)
    df.to_csv('../prolific_user_data.csv', index=False)
    combine_user_preferences(res)
