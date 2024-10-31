import pandas as pd
from numpy.matlib import empty

from meal_setting import MealSetting


def get_meal_settings_from_wwd() -> [MealSetting]:
    data = pd.read_csv("../data/WorldWideDishes_2024_June.csv", delimiter=',', on_bad_lines='skip')
    res = []

    for index, row in data.iterrows():
        meal_name = str(row['english_name'])
        if meal_name == "nan":
            continue
        utensils = str(row['utensils'])
        if utensils == "nan":
            continue
        plate = _get_plate_type_from_dish_type(row['type_of_dish'])
        setting = MealSetting(meal_name, 'fork' in utensils, 'spoon' in utensils, 'knife' in utensils, plate)
        res.append(setting)
    return res


def _get_plate_type_from_dish_type(dish_type: str) -> str:
    if 'Main dish' in dish_type:
        return 'Normal Plate'
    if 'Dessert' in dish_type or 'Side dish' in dish_type or 'Starter' in dish_type:
        return 'Small Plate'
    if 'Small plate / bowl for sharing' in dish_type:
        return 'Small Plate or Bowl'
    if 'Soup' in dish_type or 'Sauce' in dish_type or 'Salad' in dish_type:
        return 'Bowl'


if __name__ == '__main__':
    res = get_meal_settings_from_wwd()
    for r in res:
        print(r)
    print(f'Found {len(res)} valid meal settings')
