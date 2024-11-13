import pandas as pd

from meal_setting import MealSetting
from utensils_plates import Utensil, Plate


def get_meal_settings_from_wwd() -> [MealSetting]:
    data = pd.read_csv("../data/WorldWideDishes_2024_June.csv", delimiter=',', on_bad_lines='skip')
    res = []

    for index, row in data.iterrows():
        meal_name = str(row['english_name'])
        if meal_name == "nan":
            continue
        uts = str(row['utensils'])
        if uts == "nan":
            continue
        plate = _get_plate_type_from_dish_type(row['type_of_dish'])
        utensils = _process_utensils(uts)
        setting = MealSetting(meal_name, utensils, plate)
        res.append(setting)
    return res


def _get_plate_type_from_dish_type(dish_type: str) -> str:
    if 'Soup' in dish_type or 'Sauce' in dish_type or 'Salad' in dish_type:
        return Plate.BOWL
    if 'Dessert' in dish_type or 'Side dish' in dish_type or 'Starter' in dish_type:
        return Plate.SMALL
    if 'Small plate / bowl for sharing' in dish_type:
        return Plate.SMALL_OR_BOWL
    if 'Main dish' in dish_type:
        return Plate.NORMAL


def analyse_utensils() -> [str]:
    data = pd.read_csv("../data/WorldWideDishes_2024_June.csv", delimiter=',', on_bad_lines='skip')
    uts = {}

    for index, row in data.iterrows():
        utensils = str(row['utensils'])
        if utensils == "nan":
            continue
        for ut in utensils.split(","):
            ut_fil = ut.strip()
            if ut_fil not in uts:
                uts[ut_fil] = 1
            else:
                uts[ut_fil] += 1
    print(uts)


def _process_utensils(utensils: [str]) -> [str]:
    result = []
    for ut in utensils.split(","):
        ut_proc = ut.strip().lower()
        if "fork" in ut_proc:
            result.append(Utensil.FORK)
        if "knife" in ut_proc:
            result.append(Utensil.KNIFE)
        if "spoon" in ut_proc:
            result.append(Utensil.SPOON)
        if "chopstick" in ut_proc:
            result.append(Utensil.CHOPS)
    return result

if __name__ == '__main__':
    res = get_meal_settings_from_wwd()
    for r in res:
        print(r)
    print(f'Found {len(res)} valid meal settings')
