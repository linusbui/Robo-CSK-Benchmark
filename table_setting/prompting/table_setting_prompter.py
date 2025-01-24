import pandas as pd

from table_setting.data_extraction.utensils_plates import Utensil, Plate
from table_setting.prompting.table_setting_model_result import TableSettingModelResult
from utils.prompter import Prompter

utensils_string = ', '.join([str(utensil) for utensil in Utensil])
plates_string = ', '.join([str(plate) for plate in Plate])
system_msg = 'Imagine you are a robot setting a table for a meal.'
user_msg_cut = f'What are the types of cutlery you would use to eat that meal? Please choose from the following and only answer with your choices: {utensils_string}'
user_msg_plat = f'What is the type of plate you would use to eat that meal? Please choose one from the following and only answer with your choice: {plates_string}'


def prompt_all_models(prompters: [Prompter]):
    comb_result = pd.DataFrame(columns=['model', 'acc', 'jacc'])
    for prompter in prompters:
        data = pd.read_csv('../combined_prolific_data.csv', delimiter=',', on_bad_lines='skip')
        results = []
        for index, row in data.iterrows():
            # setup meal name & get gold standard data
            meal = row['name']
            plate = get_fitting_plate(row)
            utensils = get_utensils(row)
            tup = TableSettingModelResult(meal, plate, utensils)

            # prompt for cutlery
            question = f'Meal: {meal}\nCutlery: '
            res = prompter.prompt_model(system_msg, user_msg_cut, question)
            tup.add_predicted_utensils(transform_utensil_prediction(res))

            # prompt for plate
            question = f'Meal: {meal}\nPlate: '
            res = prompter.prompt_model(system_msg, user_msg_plat, question)
            tup.add_predicted_plate(transform_plate_prediction(res))

            results.append(tup)
        write_results_to_file(results, prompter.model_name)
        comb_result = pd.concat([comb_result, calculate_average(results, prompter.model_name)], ignore_index=True)
    comb_result.to_csv('../results/model_overview.csv', index=False)


def get_fitting_plate(row) -> Plate:
    max_plate = Plate.NONE
    max_val = -1
    columns = ['dinner plate', 'dessert plate', 'bowl', 'coupe plate']
    for c in columns:
        val = row[c]
        if val > max_val:
            max_val = val
            max_plate = transform_plate_prediction(c)
    return max_plate


def get_utensils(row) -> [Utensil]:
    utensils = []
    columns = ['hands', 'tongs', 'knife', 'fork', 'skewer', 'chopsticks', 'spoon']
    thresh = 10
    for c in columns:
        val = row[c]
        if val >= thresh:
            utensils.append(transform_utensil_prediction(c)[0])
    return utensils


def transform_utensil_prediction(pred: str) -> [Utensil]:
    res = []
    for utensil in Utensil:
        if utensil.lower() in pred.lower():
            res.append(utensil)
    if len(res) == 0:
        print(f'Error: "{pred}" contains no valid utensil predictions')
    return res


def transform_plate_prediction(pred: str) -> Plate:
    for plate in Plate:
        plate_type = plate.split(' ')[0]
        if plate_type.lower() in pred.lower():
            return Plate(plate)
    print(f'Error: "{pred}" is not a valid type of Plate')
    return Plate.NONE


def write_results_to_file(results: [TableSettingModelResult], model: str):
    dict_list = [re.to_dict() for re in results]
    df = pd.DataFrame(dict_list)
    df.to_csv(f'../results/{model}.csv', index=False)


def calculate_average(results: [TableSettingModelResult], model: str):
    average = {met: 0 for met in ['acc', 'jacc']}
    for res in results:
        if res.get_plate_pred_correctness():
            average['acc'] += 1
        average['jacc'] += res.get_jaccard_for_utensils()
    new_row = pd.Series(
        {'model': model, 'acc': (average['acc'] / len(results)), 'jacc': (average['jacc'] / len(results))})
    return new_row.to_frame().T
