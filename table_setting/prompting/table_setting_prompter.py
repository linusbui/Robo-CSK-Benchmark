import pandas as pd
from tqdm import tqdm

from table_setting.data_extraction.utensils_plates import Utensil, Plate
from table_setting.prompting.table_setting_model_result import TableSettingModelResult
from utils.prompter import Prompter
from utils.result_writer import add_to_model_overview, write_model_results_to_file
from utils.formatting import majority_vote

utensils_string = ', '.join([str(utensil) for utensil in Utensil])
plates_string = ', '.join([str(plate) for plate in Plate])
system_msg = 'Imagine you are a robot setting a table for a meal.'
user_msg_cut = f'What are the types of cutlery you would use to eat that meal? Please choose from the following and only answer with your choices: {utensils_string}'
user_msg_plat = f'What is the type of plate you would use to eat that meal? Please choose one from the following and only answer with your choice: {plates_string}'


def prompt_all_models(prompters: [Prompter]):
    for prompter in prompters:
        data = pd.read_csv('table_setting/combined_prolific_data_small.csv', delimiter=',', on_bad_lines='skip')
        results = []
        for index, row in tqdm(data.iterrows(), f'Prompting {prompter.model_name} for the Table Setting task'):
            # setup meal name & get gold standard data
            meal = row['name']
            plate = get_fitting_plate(row)
            utensils = get_utensils(row)
            tup = TableSettingModelResult(meal, plate, utensils)

            # prompt for cutlery
            question = f'Meal: {meal}\nCutlery: '
            res = prompter.prompt_model(system_msg, user_msg_cut_meta, question)
            tup.add_predicted_utensils(transform_utensil_prediction_meta(res))

            # prompt for plate
            question = f'Meal: {meal}\nPlate: '
            res = prompter.prompt_model(system_msg, user_msg_plat_meta, question)
            tup.add_predicted_plate(transform_plate_prediction_meta(res))

            results.append(tup)
        write_model_results_to_file(results, prompter.model_name + '_' + technique, 'table_setting')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_' + technique), 'table_setting')


user_msg_cut_rar = f'What are the types of cutlery you would use to eat that meal? Please choose from the following and only answer with your choices: {utensils_string}. Reword and elaborate on the inquiry, then provide your final answer.'
user_msg_plat_rar = f'What is the type of plate you would use to eat that meal? Please choose one from the following and only answer with your choice: {plates_string}. Reword and elaborate on the inquiry, then provide your final answer.'

def prompt_all_models_rar(prompters: [Prompter]):
    for prompter in prompters:
        data = pd.read_csv('table_setting/combined_prolific_data_small.csv', delimiter=',', on_bad_lines='skip')
        results = []
        for index, row in tqdm(data.iterrows(), f'Prompting {prompter.model_name} for the Table Setting task'):
            # setup meal name & get gold standard data
            meal = row['name']
            plate = get_fitting_plate(row)
            utensils = get_utensils(row)
            tup = TableSettingModelResult(meal, plate, utensils)

            # prompt for cutlery
            question = f'Meal: {meal}\nCutlery: '
            res = prompter.prompt_model(system_msg, user_msg_cut_rar, question)
            tup.add_predicted_utensils(transform_utensil_prediction_meta(res))

            # prompt for plate
            question = f'Meal: {meal}\nPlate: '
            res = prompter.prompt_model(system_msg, user_msg_plat_rar, question)
            tup.add_predicted_plate(transform_plate_prediction_meta(res))

            results.append(tup)
        write_model_results_to_file(results, prompter.model_name + '_rar', 'table_setting')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_rar'), 'table_setting')


user_msg_cut_meta = f'''What are the types of cutlery you would use to eat that meal? Please choose from the following {utensils_string}. As you perform this task, follow these steps:
1. Clarify your understanding of the question
2. Make a preliminary identification of the types of cutlery used to eat the meal with.
3. Critically asses your preliminary analysis. If you are unsure about the initial assesment of the types of cutlery used to eat the meal, try to reasses it.
4. Confirm your final decision on the types of cutlery used to eat the meal and explain the reasoning behind your choices.
5. Evaluate your confidence (0-100%) in your analysis and provide an explanation for this confidence level.
Provide the answer in your final response as only your choices.
'''

user_msg_plat_meta = f'''What are the types of cutlery you would use to eat that meal? Please choose from the following {plates_string}. As you perform this task, follow these steps:
1. Clarify your understanding of the question
2. Make a preliminary identification of the plates used to eat the meal with.
3. Critically asses your preliminary analysis. If you are unsure about the initial assesment of the plates used to eat the meal, try to reasses it.
4. Confirm your final decision on the most fitting plate used to eat the meal and explain the reasoning behind your choice.
5. Evaluate your confidence (0-100%) in your analysis and provide an explanation for this confidence level.
Provide the answer in your final response as only your choice.
'''

def prompt_all_models_meta(prompters: [Prompter]):
    for prompter in prompters:
        data = pd.read_csv('table_setting/combined_prolific_data_small.csv', delimiter=',', on_bad_lines='skip')
        results = []
        for index, row in tqdm(data.iterrows(), f'Prompting {prompter.model_name} for the Table Setting task'):
            # setup meal name & get gold standard data
            meal = row['name']
            plate = get_fitting_plate(row)
            utensils = get_utensils(row)
            tup = TableSettingModelResult(meal, plate, utensils)

            # prompt for cutlery
            question = f'Meal: {meal}\nCutlery: '
            res = prompter.prompt_model(system_msg, user_msg_cut_meta, question)
            tup.add_predicted_utensils(transform_utensil_prediction_meta(res))

            # prompt for plate
            question = f'Meal: {meal}\nPlate: '
            res = prompter.prompt_model(system_msg, user_msg_plat_meta, question)
            tup.add_predicted_plate(transform_plate_prediction_meta(res))

            results.append(tup)
        write_model_results_to_file(results, prompter.model_name + '_meta', 'table_setting')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_meta'), 'table_setting')


user_msg_cut_selfcon = f'What are the types of cutlery you would use to eat that meal? Please choose from the following: {utensils_string}. Think step by step before answering with the cutlery of your choosing.'
user_msg_plat_selfcon = f'What is the type of plate you would use to eat that meal? Please choose one from the following: {plates_string}. Think step by step before answering with your chosen plate.'

MAXIT_selfcon = 2
def prompt_all_models_selfcon(prompters: [Prompter]):
    for prompter in prompters:
        data = pd.read_csv('table_setting/combined_prolific_data_small.csv', delimiter=',', on_bad_lines='skip')
        results = []
        for index, row in tqdm(data.iterrows(), f'Prompting {prompter.model_name} for the Table Setting task'):
            # setup meal name & get gold standard data
            meal = row['name']
            plate = get_fitting_plate(row)
            utensils = get_utensils(row)
            tup = TableSettingModelResult(meal, plate, utensils)

            answers_cut = []
            answers_plate = []
            for i in range(MAXIT_selfcon):
                # prompt for cutlery
                question = f'Meal: {meal}\nCutlery: '
                res = prompter.prompt_model(system_msg, user_msg_cut_selfcon, question)
                answers_cut.append(transform_utensil_prediction_selfcon(res))

                # prompt for plate
                question = f'Meal: {meal}\nPlate: '
                res = prompter.prompt_model(system_msg, user_msg_plat_selfcon, question)
                answers_plate.append(transform_plate_prediction_selfcon(res))

            tup.add_predicted_utensils(majority_vote(answers_cut))
            tup.add_predicted_plate(majority_vote(answers_plate))

            results.append(tup)
        write_model_results_to_file(results, prompter.model_name + '_selfcon', 'table_setting')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_selfcon'), 'table_setting')


user_msg_cut_initial = f'What are the types of cutlery you would use to eat that meal? Please choose from the following: {utensils_string}'
user_msg_plat_inital = f'What is the type of plate you would use to eat that meal? Please choose one from the following: {plates_string}'

user_msg_cut_feedback = "Provide Feedback on the answer. If you think the answer contains the right selection of cutlery, end your answer with 'STOP'."
user_msg_plat_feedback = "Provide Feedback on the answer. If you think the answer contains the right plate, end your answer with 'STOP'."
user_msg_refine = 'Improve upon the answer based on the feedback:'

MAXIT_selfref = 2
def prompt_all_models_selfref(prompters: [Prompter]):
    for prompter in prompters:
        data = pd.read_csv('table_setting/combined_prolific_data_small.csv', delimiter=',', on_bad_lines='skip')
        results = []
        for index, row in tqdm(data.iterrows(), f'Prompting {prompter.model_name} for the Table Setting task'):
            # setup meal name & get gold standard data
            meal = row['name']
            plate = get_fitting_plate(row)
            utensils = get_utensils(row)
            tup = TableSettingModelResult(meal, plate, utensils)

            # initial prompt for cutlery
            question = f'Meal: {meal}\nCutlery: '
            initial = prompter.prompt_model(system_msg, user_msg_cut_initial, question)
            question = question + f'\n{initial}\n'

            # Feedback - Refine iterations
            for i in range(MAXIT_selfref):
                # feedback 
                question = question + '\nYour Feedback:'
                feedback = prompter.prompt_model(system_msg, user_msg_cut_feedback, question)
                question = question + f'\n{feedback}'

                if 'STOP' in feedback: break

                # refine
                question = question + '\nImprovement:'
                refine = prompter.prompt_model(system_msg, user_msg_refine, question)
                question = question + f'\n{refine}\n'

            # final answer for cutlery
            user_msg_cut_final = f'Please provide your final answer based on the given feedback-answer iterations. Please choose from the following and only answer with your choices: {utensils_string}'
            question = question + 'Your Choice:'
            final_pred = prompter.prompt_model(system_msg, user_msg_cut_final, question)
            tup.add_predicted_utensils(transform_utensil_prediction_meta(final_pred))

            # initial prompt for plate
            question = f'Meal: {meal}\nPlate: '
            initial = prompter.prompt_model(system_msg, user_msg_plat_inital, question)
            question = question + f'\n{initial}\n'

            # Feedback - Refine iterations
            for i in range(MAXIT_selfref):
                # feedback 
                question = question + '\nYour Feedback:'
                feedback = prompter.prompt_model(system_msg, user_msg_plat_feedback, question)
                question = question + f'\n{feedback}'

                if 'STOP' in feedback: break

                # refine
                question = question + '\nImprovement:'
                refine = prompter.prompt_model(system_msg, user_msg_refine, question)
                question = question + f'\n{refine}\n'

            # final answer for plate
            user_msg_plat_final = f'Please provide your final answer based on the given feedback-answer iterations. Please choose from the following and only answer with your choice: {plates_string}'
            question = question + 'Your Choice:'
            final_pred = prompter.prompt_model(system_msg, user_msg_plat_final, question)
            tup.add_predicted_plate(transform_plate_prediction_meta(final_pred))

            results.append(tup)
        write_model_results_to_file(results, prompter.model_name + '_selfref', 'table_setting')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_selfref'), 'table_setting')


system_msg = 'Imagine you are a robot setting a table for a meal.'
user_msg_cut_principle = 'Your task is to extract the underlying concepts and principles that should be considered when selecting the types of cutlery to eat a given meal with.'
user_msg_plat_principle = 'Your task is to extract the underlying concepts and principles that should be considered when selecting the types of plate to eat a given meal on.'

def prompt_all_models_stepback(prompters: [Prompter]):
    for prompter in prompters:
        data = pd.read_csv('table_setting/combined_prolific_data_small.csv', delimiter=',', on_bad_lines='skip')
        results = []
        for index, row in tqdm(data.iterrows(), f'Prompting {prompter.model_name} for the Table Setting task'):
            # setup meal name & get gold standard data
            meal = row['name']
            plate = get_fitting_plate(row)
            utensils = get_utensils(row)
            tup = TableSettingModelResult(meal, plate, utensils)

            # prompt for cutlery
            # Get higher level principles
            question = f'Meal: {meal}\nPrinciples: '
            principles = prompter.prompt_model(system_msg, user_msg_cut_principle, question)

            # Get answer based on principles
            question = f'Meal: {meal}\nCutlery: '
            user_msg_cut_stepback = f'What are the types of cutlery you would use to eat that meal? Please choose from the following: {utensils_string} and answer the question step by step using the following principles:\n{principles}\n Provide your final answer as only the cutlery of your choosing.'
            
            res = prompter.prompt_model(system_msg, user_msg_cut_stepback, question)
            tup.add_predicted_utensils(transform_utensil_prediction_selfcon(res))

            # prompt for plate
            # Get higher level principles
            question = f'Meal: {meal}\nPrinciples: '
            principles = prompter.prompt_model(system_msg, user_msg_plat_principle, question)

            # Get answer based on principles
            question = f'Meal: {meal}\nPlate: '
            user_msg_plat_stepback = f'What is the type of plate you would use to eat that meal? Please choose one from the following: {plates_string} and answer the question step by step using the following principles:\n{principles}\n Provide your final answer as only your chosen plate.'

            res = prompter.prompt_model(system_msg, user_msg_plat_stepback, question)
            tup.add_predicted_plate(transform_plate_prediction_selfcon(res))

            results.append(tup)
        write_model_results_to_file(results, prompter.model_name + '_stepback', 'table_setting')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_stepback'), 'table_setting')



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


def transform_utensil_prediction_meta(pred: str) -> [Utensil]:
    answ = pred.splitlines()[-1]

    res = []
    for utensil in Utensil:
        if utensil.lower() in answ.lower():
            res.append(utensil)
    if len(res) == 0:
        print(f'Error: "{pred}" contains no valid utensil predictions')
    return res


def transform_plate_prediction_meta(pred: str) -> Plate:
    answ = pred.splitlines()[-1]

    for plate in Plate:
        plate_type = plate.split(' ')[0]
        if plate_type.lower() in answ.lower():
            return Plate(plate)
    print(f'Error: "{pred}" is not a valid type of Plate')
    return Plate.NONE


def transform_utensil_prediction_selfcon(pred: str) -> [Utensil]:
    split = pred.splitlines()
    if len(split) > 1:
        answ = split[-2] + split[-1]
    else:
        answ = split[-1]

    res = []
    for utensil in Utensil:
        if utensil.lower() in answ.lower():
            res.append(utensil)
    if len(res) == 0:
        print(f'Error: "{pred}" contains no valid utensil predictions')
    return res


def transform_plate_prediction_selfcon(pred: str) -> Plate:
    split = pred.splitlines()
    if len(split) > 1:
        answ = split[-2] + split[-1]
    else:
        answ = split[-1]

    for plate in Plate:
        plate_type = plate.split(' ')[0]
        if plate_type.lower() in answ.lower():
            return Plate(plate)
    print(f'Error: "{pred}" is not a valid type of Plate')
    return Plate.NONE


def calculate_average(results: [TableSettingModelResult], model: str):
    average = {met: 0 for met in ['acc', 'jacc']}
    for res in results:
        if res.get_plate_pred_correctness():
            average['acc'] += 1
        average['jacc'] += res.get_jaccard_for_utensils()
    new_row = pd.Series(
        {'model': model, 'acc': (average['acc'] / len(results)), 'jacc': (average['jacc'] / len(results))})
    return new_row.to_frame().T
