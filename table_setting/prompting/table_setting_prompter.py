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
user_msg_feedback = 'Provide Feedback on the answer:'
user_msg_refine = 'Improve upon the answer based on the feedback:'
user_msg_cut_final = 'Please provide your final answer. The answer should only contain the cutlery of your choosing.'
user_msg_plat_final = 'Please provide your final answer. The answer should only contain the one plate of your choosing.'

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
                feedback = prompter.prompt_model(system_msg, user_msg_feedback, question)
                question = question + f'\n{feedback}'

                # refine
                question = question + '\nImprovement:'
                refine = prompter.prompt_model(system_msg, user_msg_refine, question)
                question = question + f'\n{refine}\n'

            # final answer for cutlery
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
                feedback = prompter.prompt_model(system_msg, user_msg_feedback, question)
                question = question + f'\n{feedback}'

                # refine
                question = question + '\nImprovement:'
                refine = prompter.prompt_model(system_msg, user_msg_refine, question)
                question = question + f'\n{refine}\n'

            # final answer for plate
            question = question + 'Your Choice:'
            final_pred = prompter.prompt_model(system_msg, user_msg_plat_final, question)
            tup.add_predicted_plate(transform_plate_prediction_meta(final_pred))

            results.append(tup)
        write_model_results_to_file(results, prompter.model_name + '_selfref', 'table_setting')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_selfref'), 'table_setting')


system_msg = 'Imagine you are a robot setting a table for a meal.'
user_msg_cut_principle = 'Your task is to extract the underlying concepts and principles that should be considered when selecting the types of cutlery to eat a meal with.'
user_msg_plat_principle = 'Your task is to extract the underlying concepts and principles that should be considered when selecting the types of plate to eat a meal on.'

principles_cut = '''
1. **Type of Meal**: Consider the nature of the meal being served. Different meals require different types of cutlery. For example, a steak dinner would typically require a steak knife, while a soup course would need a soup spoon.

2. **Cuisine Specifics**: Different cuisines may have specific cutlery requirements. For instance, Asian meals might require chopsticks, while Western meals typically use forks, knives, and spoons.

3. **Course Sequence**: The sequence of courses in a meal can dictate the cutlery needed. Each course may require specific utensils, such as a salad fork for the salad course or a dessert spoon for dessert.

4. **Functionality**: The cutlery should be appropriate for the food being served. For example, a serrated knife is better for cutting bread, while a butter knife is suitable for spreading.

5. **Etiquette and Formality**: The level of formality of the meal can influence cutlery selection. Formal dining settings may require more specialized cutlery, such as fish forks or oyster forks, while casual settings may use more basic utensils.

6. **Material and Aesthetics**: The material and design of the cutlery can enhance the dining experience. Stainless steel is common for its durability and ease of maintenance, while silver or gold-plated cutlery might be used for more formal occasions.

7. **Cultural Norms**: Be mindful of cultural norms and traditions that may dictate specific cutlery use. For example, in some cultures, it is customary to eat with hands or specific utensils.

8. **Guest Preferences and Needs**: Consider any dietary restrictions or preferences that might affect cutlery choice, such as providing a steak knife for a guest who prefers meat or ensuring there are utensils suitable for left-handed guests.

9. **Sustainability and Environmental Impact**: Consider the environmental impact of the cutlery, opting for reusable or biodegradable options when possible, especially for large gatherings or events.

10. **Safety and Comfort**: Ensure that the cutlery is safe and comfortable to use, with no sharp edges or awkward designs that could cause discomfort or injury.
'''
principles_plat = '''
1. **Functionality and Purpose**:
   - **Type of Meal**: Consider the type of meal being served (e.g., breakfast, lunch, dinner, dessert) and choose plates that accommodate the specific dishes (e.g., soup bowls for soup, dinner plates for main courses).
   - **Portion Size**: Select plates that are appropriately sized for the portion of food being served to avoid overcrowding or excessive empty space.

2. **Material and Durability**:
   - **Material Suitability**: Choose materials that are suitable for the type of food and the dining setting (e.g., porcelain for formal dining, melamine for casual or outdoor settings).
   - **Durability**: Consider the durability of the plate material, especially for everyday use or settings where breakage is a concern.

3. **Aesthetics and Style**:
   - **Design and Color**: Select plates that complement the overall table setting and theme of the meal, considering color, pattern, and design.
   - **Cohesion**: Ensure that the plates match or coordinate with other tableware, such as cutlery, glasses, and napkins, to create a cohesive look.

4. **Cultural and Traditional Considerations**:
   - **Cultural Appropriateness**: Be mindful of cultural norms and traditions that may dictate specific types of plates or presentation styles for certain meals or occasions.

5. **Practicality and Convenience**:
   - **Ease of Handling**: Consider the weight and shape of the plates for ease of handling and serving.
   - **Stackability and Storage**: Choose plates that are easy to stack and store, especially if space is limited.

6. **Environmental Impact**:
   - **Sustainability**: Consider the environmental impact of the plate materials and opt for sustainable or eco-friendly options when possible.

7. **Safety and Health**:
   - **Food Safety**: Ensure that the plates are made from food-safe materials that do not leach harmful substances.
   - **Temperature Resistance**: Consider the plate's ability to withstand temperature changes, especially if the meal involves hot or cold dishes.
'''

user_msg_cut_stepback = f'What are the types of cutlery you would use to eat that meal? Please choose from the following: {utensils_string} and answer the question step by step using the following principles:\n{principles_cut}\n Provide your final answer as only the cutlery of your choosing.'
user_msg_plat_stepback = f'What is the type of plate you would use to eat that meal? Please choose one from the following: {plates_string} and answer the question step by step using the following principles:\n{principles_plat}\n Provide your final answer as only your chosen plate.'


def prompt_all_models_stepback(prompters: [Prompter]):
    for prompter in prompters:
        # Get underlying principles
        # result is prepared in principles
        principles_cut = prompter.prompt_model(system_msg, user_msg_cut_principle, 'Principles Involved:')
        print(principles_cut)
        print('################################################')
        principles_plat = prompter.prompt_model(system_msg, user_msg_plat_principle, 'Principles Involved:')
        print(principles_plat)
        return

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
    answ = split[-2] + split[-1]

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
