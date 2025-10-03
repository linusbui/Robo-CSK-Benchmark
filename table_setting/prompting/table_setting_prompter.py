import pandas as pd
from tqdm import tqdm
import os.path

from table_setting.data_extraction.utensils_plates import Utensil, Plate
from table_setting.prompting.table_setting_model_result import TableSettingModelResult
from utils.prompter import Prompter
from utils.result_writer import add_to_model_overview, write_model_results_to_file
from utils.formatting import transform_prediction, majority_vote
from utils.logging import BasicLogEntry, StepbackLogEntry, write_log_to_file, write_general_log_to_file

utensils_string = ', '.join([str(utensil) for utensil in Utensil])
plates_string = ', '.join([str(plate) for plate in Plate])
system_msg = 'Imagine you are a robot setting a table for a meal.'
user_msg_cut = f'What are the types of cutlery you would use to eat that meal? Please choose from the following and only answer with your choices: {utensils_string}'
user_msg_plat = f'What is the type of plate you would use to eat that meal? Please choose one from the following and only answer with your choice: {plates_string}'

def prompt_all_models(prompters: [Prompter], num_runs: int):
    for prompter in prompters:
        data = pd.read_csv('table_setting/combined_prolific_data.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
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
            tup.add_predicted_utensils(transform_utensil_prediction(res))

            # prompt for plate
            question = f'Meal: {meal}\nPlate: '
            res = prompter.prompt_model(system_msg, user_msg_plat_meta, question)
            tup.add_predicted_plate(transform_plate_prediction(res))

            results.append(tup)
        write_model_results_to_file(results, prompter.model_name, '', 'table_setting')
        add_to_model_overview(calculate_average(results, prompter.model_name), 'table_setting')


user_msg_cut_rar = f'What are the types of cutlery you would use to eat that meal? Please choose from the following and only answer with your choices: {utensils_string}. Reword and elaborate on the inquiry, then provide your final answer.'
user_msg_plat_rar = f'What is the type of plate you would use to eat that meal? Please choose one from the following and only answer with your choice: {plates_string}. Reword and elaborate on the inquiry, then provide your final answer.'

def prompt_all_models_rar(prompters: [Prompter], num_runs: int):
    for prompter in prompters:
        data = pd.read_csv('table_setting/combined_prolific_data.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        results = []
        logs_cut = []
        logs_plat = []
        for index, row in tqdm(data.iterrows(), f'Prompting {prompter.model_name} for the Table Setting task with RaR Prompting'):
            # setup meal name & get gold standard data
            meal = row['name']
            plate = get_fitting_plate(row)
            utensils = get_utensils(row)
            tup = TableSettingModelResult(meal, plate, utensils)

            # prompt for cutlery
            question = f'Meal: {meal}\nCutlery:'
            res = prompter.prompt_model(system_msg, user_msg_cut_rar, question)
            pred_cut = transform_utensil_prediction_new(res)
            tup.add_predicted_utensils(pred_cut)
            log_cut = BasicLogEntry(question, res, pred_cut, utensils)

            # prompt for plate
            question = f'Meal: {meal}\nPlate:'
            res = prompter.prompt_model(system_msg, user_msg_plat_rar, question)
            pred_plat = transform_plate_prediction_new(res)
            tup.add_predicted_plate(pred_plat)
            log_plat = BasicLogEntry(question, res, pred_plat, plate)

            results.append(tup)
            logs_cut.append(log_cut)
            logs_plat.append(log_plat)
        write_model_results_to_file(results, prompter.model_name, 'rar', 'table_setting')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_rar'), 'table_setting')
        write_log_to_file(logs_cut, prompter.model_name, 'cutlery_rar', 'table_setting')
        write_log_to_file(logs_plat, prompter.model_name, 'plate_rar', 'table_setting')


user_msg_cut_meta = f'''What are the types of cutlery you would use to eat that meal? Please choose from the following {utensils_string}. As you perform this task, follow these steps:
1. Clarify your understanding of the question
2. Make a preliminary identification of the types of cutlery used to eat the meal with.
3. Critically asses your preliminary analysis. If you are unsure about the initial assesment of the types of cutlery used to eat the meal, try to reasses it.
4. Confirm your final decision on the types of cutlery used to eat the meal and explain the reasoning behind your choices. Focus on the cutlery that is most likely needed and be decisive in your choices.
5. Evaluate your confidence (0-100%) in your analysis and provide an explanation for this confidence level.
6. Repeat your final choices.
'''

user_msg_plat_meta = f'''What are the type of plate you would use to eat that meal? Please choose from the following {plates_string}. As you perform this task, follow these steps:
1. Clarify your understanding of the question
2. Make a preliminary identification of the plates used to eat the meal with.
3. Critically asses your preliminary analysis. If you are unsure about the initial assesment of the plates used to eat the meal, try to reasses it.
4. Confirm your final decision on the most fitting plate used to eat the meal and explain the reasoning behind your choice.
5. Evaluate your confidence (0-100%) in your analysis and provide an explanation for this confidence level.
6. Repeat your final choice.
'''

def prompt_all_models_meta(prompters: [Prompter], num_runs: int):
    for prompter in prompters:
        data = pd.read_csv('table_setting/combined_prolific_data.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        results = []
        logs_cut = []
        logs_plat = []
        for index, row in tqdm(data.iterrows(), f'Prompting {prompter.model_name} for the Table Setting task with Metacognitive Prompting'):
            # setup meal name & get gold standard data
            meal = row['name']
            plate = get_fitting_plate(row)
            utensils = get_utensils(row)
            tup = TableSettingModelResult(meal, plate, utensils)

            # prompt for cutlery
            question = f'Meal: {meal}'
            res = prompter.prompt_model(system_msg, user_msg_cut_meta, question)
            pred_cut = transform_utensil_prediction_new(res)
            tup.add_predicted_utensils(pred_cut)
            log_cut = BasicLogEntry(question, res, pred_cut, utensils)

            # prompt for plate
            question = f'Meal: {meal}'
            res = prompter.prompt_model(system_msg, user_msg_plat_meta, question)
            pred_plat = transform_plate_prediction_new(res)
            tup.add_predicted_plate(pred_plat)
            log_plat = BasicLogEntry(question, res, pred_plat, plate)

            results.append(tup)
            logs_cut.append(log_cut)
            logs_plat.append(log_plat)
        write_model_results_to_file(results, prompter.model_name, 'meta', 'table_setting')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_meta'), 'table_setting')
        write_log_to_file(logs_cut, prompter.model_name, 'cutlery_meta', 'table_setting')
        write_log_to_file(logs_plat, prompter.model_name, 'plate_meta', 'table_setting')


user_msg_cut_selfcon = f'What are the types of cutlery you would use to eat that meal? Please choose from the following: {utensils_string}. Think step by step before answering with the cutlery of your choosing.'
user_msg_plat_selfcon = f'What is the type of plate you would use to eat that meal? Please choose one from the following: {plates_string}. Think step by step before answering with your chosen plate.'

def prompt_all_models_selfcon(prompters: [Prompter], num_runs: int, n_it: int):
    for prompter in prompters:
        data = pd.read_csv('table_setting/combined_prolific_data.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        results = []
        logs_cut = []
        logs_plat = []
        for index, row in tqdm(data.iterrows(), f'Prompting {prompter.model_name} for the Table Setting task with Self-Consistency Prompting'):
            # setup meal name & get gold standard data
            meal = row['name']
            plate = get_fitting_plate(row)
            utensils = get_utensils(row)
            tup = TableSettingModelResult(meal, plate, utensils)

            question_cut = f'Meal: {meal}'
            question_plat = f'Meal: {meal}'

            log_cut = {'question': question_cut}
            log_plat = {'question': question_plat}

            answers_cut = []
            answers_plate = []
            for i in range(n_it):
                # prompt for cutlery
                res = prompter.prompt_model(system_msg, user_msg_cut_selfcon, question_cut)
                pred_cut = transform_utensil_prediction_new(res)
                answers_cut.append(pred_cut)
                log_cut.update({f'cot_{i}': res,
                                f'answer_{i}': pred_cut})

                # prompt for plate
                res = prompter.prompt_model(system_msg, user_msg_plat_selfcon, question_plat)
                pred_plat = transform_plate_prediction_new(res)
                answers_plate.append(pred_plat)
                log_plat.update({f'cot_{i}': res,
                                 f'answer_{i}': pred_plat})

            final_cut = majority_vote(answers_cut)
            final_plat = majority_vote(answers_plate)
            tup.add_predicted_utensils(final_cut)
            tup.add_predicted_plate(final_plat)
            log_cut.update({'final_answer': final_cut,
                            'correct_answer': utensils})
            log_plat.update({'final_answer': final_plat,
                            'correct_answer': plate})

            results.append(tup)
            logs_cut.append(log_cut)
            logs_plat.append(log_plat)
        write_model_results_to_file(results, prompter.model_name, 'selfcon', 'table_setting')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_selfcon'), 'table_setting')
        write_general_log_to_file(logs_cut, prompter.model_name, 'cutlery_selfcon', 'table_setting')
        write_general_log_to_file(logs_plat, prompter.model_name, 'plate_selfcon', 'table_setting')


user_msg_cut_initial = f'What are the types of cutlery you would use to eat that meal? Please choose from the following: {utensils_string}. Generate your answer in the following format:\nExplanation: <explanation>\nCutlery: <cutlery>'
user_msg_plat_inital = f'What is the type of plate you would use to eat that meal? Please choose one from the following: {plates_string}. Generate your answer in the following format:\nExplanation: <explanation>\nPlate: <plate>'

system_msg_cut_feedback = f'You are given an answer to a multiple choice question regarding setting a table. The possible answers are: {utensils_string}'
system_msg_plat_feedback = f'You are given an answer to a multiple choice question regarding setting a table. The possible answers are: {plates_string}'
user_msg_feedback = "Provide Feedback on the answer. The feedback should only evaluate the correctness of the answer. At the end, score the answer from 1 to 5. A score of 5 means that the answer is the right choice."

system_msg_refine = 'You are given an answer to a multiple choice question regarding setting a table and corresponding feedback.'
user_msg_cut_refine = 'Improve upon the answer based on the feedback. Remember that the answer has to be chosen from the given list. Generate your answer in the following format:\nExplanation: <explanation>\nCutlery: <cutlery>'
user_msg_plat_refine = 'Improve upon the answer based on the feedback. Remember that the answer has to be chosen from the given list. Generate your answer in the following format:\nExplanation: <explanation>\nPlate: <plate>'

def prompt_all_models_selfref(prompters: [Prompter], num_runs: int, n_it: int):
    for prompter in prompters:
        data = pd.read_csv('table_setting/combined_prolific_data.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        results = []
        logs_cut = []
        logs_plat = []
        for index, row in tqdm(data.iterrows(), f'Prompting {prompter.model_name} for the Table Setting task with Self-Refine Prompting' ):
            # setup meal name & get gold standard data
            meal = row['name']
            plate = get_fitting_plate(row)
            utensils = get_utensils(row)
            tup = TableSettingModelResult(meal, plate, utensils)

            # initial prompt for cutlery
            question = f'Meal: {meal}\nCutlery:'
            answer = prompter.prompt_model(system_msg, user_msg_cut_initial, question)

            # full conversation for logging, LLM only sees last answer
            full_conv = question + f'\n{answer}'

            # Feedback - Refine iterations
            for i in range(n_it):
                # feedback 
                f_question = question + f'\n{answer}' + '\nFeedback:'
                feedback = prompter.prompt_model(system_msg_cut_feedback, user_msg_feedback, f_question)
                full_conv = full_conv + '\nFeedback:' + f'\n{feedback}'

                # Stopping condition
                ind = feedback.lower().find('score')
                if not ind == -1:
                    end = feedback[ind:]
                    if '4' in end or '5' in end:
                        break

                # refine
                r_question = f_question + f'\n{feedback}'
                answer = prompter.prompt_model(system_msg_refine, user_msg_cut_refine, r_question)
                full_conv = full_conv + '\nImprovement:' + f'\n{answer}'

            # final answer for cutlery
            pred_cut = transform_utensil_prediction_new(answer)
            tup.add_predicted_utensils(pred_cut)
            log_cut = BasicLogEntry(question, full_conv, pred_cut, utensils)

            # initial prompt for plate
            question = f'Meal: {meal}\nPlate:'
            answer = prompter.prompt_model(system_msg, user_msg_plat_inital, question)

            # full conversation for logging, LLM only sees last answer
            full_conv = question + f'\n{answer}'

            # Feedback - Refine iterations
            for i in range(n_it):
                # feedback 
                f_question = question + f'\n{answer}' + '\nFeedback:'
                feedback = prompter.prompt_model(system_msg_plat_feedback, user_msg_feedback, f_question)
                full_conv = full_conv + '\nFeedback:' + f'\n{feedback}'

                # Stopping condition
                ind = feedback.lower().find('score')
                if not ind == -1:
                    end = feedback[ind:]
                    if '4' in end or '5' in end:
                        break

                # refine
                r_question = f_question + f'\n{feedback}'
                answer = prompter.prompt_model(system_msg_refine, user_msg_plat_refine, r_question)
                full_conv = full_conv + '\nImprovement:' + f'\n{answer}'

            # final answer for plate
            pred_plat = transform_plate_prediction_new(answer)
            tup.add_predicted_plate(pred_plat)
            log_plat = BasicLogEntry(question, full_conv, pred_plat, plate)

            results.append(tup)
            logs_cut.append(log_cut)
            logs_plat.append(log_plat)
        write_model_results_to_file(results, prompter.model_name, 'selfref', 'table_setting')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_selfref'), 'table_setting')
        write_log_to_file(logs_cut, prompter.model_name, 'cutlery_selfref', 'table_setting')
        write_log_to_file(logs_plat, prompter.model_name, 'plate_selfref', 'table_setting')


system_msg_cut_principle = 'You are given the title of a meal. Your task is to extract the underlying concepts and principles involved in choosing the right cutlery to eat that meal with.'
system_msg_plat_principle = 'You are given the title of a meal. Your task is to extract the underlying concepts and principles involved in choosing the right plate to eat that meal on.'
user_msg__principle = 'Only answer with the 5 most important concepts and principles.'

def prompt_all_models_stepback(prompters: [Prompter], num_runs: int):
    for prompter in prompters:
        data = pd.read_csv('table_setting/combined_prolific_data.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        results = []
        logs_cut = []
        logs_plat = []
        for index, row in tqdm(data.iterrows(), f'Prompting {prompter.model_name} for the Table Setting task with Stepback Prompting'):
            # setup meal name & get gold standard data
            meal = row['name']
            plate = get_fitting_plate(row)
            utensils = get_utensils(row)
            tup = TableSettingModelResult(meal, plate, utensils)

            # prompt for cutlery
            # Get higher level principles
            p_question = f'Meal: {meal}\nPrinciples:'
            principles = prompter.prompt_model(system_msg_cut_principle, user_msg__principle, p_question)

            # Get answer based on principles
            question = f'Meal: {meal}'
            user_msg_cut_stepback = f'What are the types of cutlery you would use to eat that meal? Please choose from the following: {utensils_string} and answer the question step by step using the following principles:\n{principles}\nEnd the answer with your chosen cutlery.'
            
            res = prompter.prompt_model(system_msg, user_msg_cut_stepback, question)
            pred_cut = transform_utensil_prediction_new(res)
            tup.add_predicted_utensils(pred_cut)
            log_cut = StepbackLogEntry(p_question, principles, question, res, pred_cut, utensils)

            # prompt for plate
            # Get higher level principles
            question = f'Meal: {meal}\nPrinciples:'
            principles = prompter.prompt_model(system_msg_plat_principle, user_msg__principle, question)

            # Get answer based on principles
            question = f'Meal: {meal}'
            user_msg_plat_stepback = f'What is the type of plate you would use to eat that meal? Please choose one from the following: {plates_string} and answer the question step by step using the following principles:\n{principles}\nEnd the answer with your chosen plate.'

            res = prompter.prompt_model(system_msg, user_msg_plat_stepback, question)
            pred_plat = transform_plate_prediction_new(res)
            tup.add_predicted_plate(pred_plat)
            log_plat = StepbackLogEntry(p_question, principles, question, res, pred_plat, plate)

            results.append(tup)
            logs_cut.append(log_cut)
            logs_plat.append(log_plat)
        write_model_results_to_file(results, prompter.model_name, 'stepback', 'table_setting')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_stepback'), 'table_setting')
        write_log_to_file(logs_cut, prompter.model_name, 'cutlery_stepback', 'table_setting')
        write_log_to_file(logs_plat, prompter.model_name, 'plate_stepback', 'table_setting')


system_msg_example = 'You are helping to create questions regarding household environments.'
user_msg_cut_example = 'For the given choice of cutlery, generate a a meal that can be eaten with that cutlery. Answer only with the meal.'
user_msg_plat_example = 'For the given plate, generate a a meal that can be eaten with that cutlery. Answer only with the meal.'

def prompt_all_models_sgicl(prompters: [Prompter], num_runs: int, n_ex: int):
    for prompter in prompters:
        # Generate examples if needed
        ex_file = f'table_setting/examples/table_setting_examples_{prompter.model_name}.csv'
        if not os.path.isfile(ex_file):
            results = []
            questions = pd.read_csv('table_setting/combined_prolific_data.csv', delimiter=',', on_bad_lines='skip', nrows=15)
            for index, row in tqdm(questions.iterrows(),
                                f'Prompting {prompter.model_name} to generate Table Setting task examples'):
                meal = row['name']
                plate = get_fitting_plate(row)
                utensils = ', '.join([str(utensil) for utensil in get_utensils(row)])

                question_cut = f'Cutlery: {utensils}\nGenerate a meal: {meal}\nGenerate a meal:'
                question_plat = f'Plate: {plate}\nGenerate a meal: {meal}\nGenerate a meal:'

                pred_meal_cut = prompter.prompt_model(system_msg_example, user_msg_cut_example, question_cut)
                pred_meal_plat = prompter.prompt_model(system_msg_example, user_msg_plat_example, question_plat)
                entry = {
                    'Meal_Cutlery': pred_meal_cut,
                    'Cutlery': utensils,
                    'Meal_Plate': pred_meal_plat,
                    'Plate': plate
                }
                results.append(entry)
            df = pd.DataFrame(results)
            df.to_csv(ex_file, index=False)
            print('Finished generating examples')

        # Load examples
        examples = pd.read_csv(ex_file, delimiter=',', on_bad_lines='skip', nrows=n_ex)
        ex_cut_str = ''
        ex_plat_str = ''
        for index, row in examples.iterrows():
            meal_cut = row['Meal_Cutlery']
            utensils = row['Cutlery']
            meal_plat = row['Meal_Plate']
            plate = row['Plate']
            ex_cut_str = ex_cut_str + f'Meal: {meal_cut}\nCutlery: {utensils}\n'
            ex_plat_str = ex_plat_str + f'Meal: {meal_plat}\nPlate: {plate}\n'

        # few shot prompting
        data = pd.read_csv('table_setting/combined_prolific_data.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        results = []
        logs_cut = []
        logs_plat = []
        for index, row in tqdm(data.iterrows(), f'Prompting {prompter.model_name} for the Table Setting task with SG-ICL Prompting'):
            # setup meal name & get gold standard data
            meal = row['name']
            plate = get_fitting_plate(row)
            utensils = get_utensils(row)
            tup = TableSettingModelResult(meal, plate, utensils)

            # prompt for cutlery
            question = f'Here are a few examples:\n{ex_cut_str}Meal: {meal}\nCutlery: '
            res = prompter.prompt_model(system_msg, user_msg_cut_meta, question)
            pred_cut = transform_utensil_prediction_new(res)
            tup.add_predicted_utensils(pred_cut)
            log_cut = BasicLogEntry(question, res, pred_cut, utensils)

            # prompt for plate
            question = f'Here are a few examples:\n{ex_plat_str}Meal: {meal}\nPlate: '
            res = prompter.prompt_model(system_msg, user_msg_plat_meta, question)
            pred_plat = transform_plate_prediction_new(res)
            tup.add_predicted_plate(pred_plat)
            log_plat = BasicLogEntry(question, res, pred_plat, utensils)

            results.append(tup)
            logs_cut.append(log_cut)
            logs_plat.append(log_plat)
        write_model_results_to_file(results, prompter.model_name, 'sgicl', 'table_setting')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_sgicl'), 'table_setting')
        write_log_to_file(logs_cut, prompter.model_name, 'cutlery_sgicl', 'table_setting')
        write_log_to_file(logs_plat, prompter.model_name, 'plate_sgicl', 'table_setting')


system_msg_rewrite = 'You are helping in rewriting answers to questions regarding household environments.'
user_msg_rewrite = 'Rewrite the given answer by swapping key points with wrong facts leading to a wrong final answer. Keep the overall structure the same.'

def prompt_all_models_contr(prompters: [Prompter], num_runs: int, n_ex: int, n_cot: int):
    for prompter in prompters:
        # Generate cutlery examples if needed
        ex_cut_file = f'table_setting/examples/table_setting_multichoice_cot_cut_examples_{prompter.model_name}.csv'
        if not os.path.isfile(ex_cut_file):
            results = []
            log_cut = pd.read_csv(f'table_setting/logs/{prompter.model_name}/{prompter.model_name}_cutlery_selfcon.csv', delimiter=',', on_bad_lines='skip', nrows=10)
            for index, row in tqdm(log_cut.iterrows(),
                                f'Prompting {prompter.model_name} to generate Table Setting task cutlery examples'):
                corr_cut = row['correct_answer']
                # get correct cot
                cot_right = ''
                for i in range(n_cot):
                    answ = row[f'answer_{i}']
                    if answ == corr_cut:
                        cot_right = row[f'cot_{i}']
                        break
                if cot_right == '': continue

                question = f'Right answer:\n{cot_right}\nWrong answer:'
                cot_wrong = prompter.prompt_model(system_msg_rewrite, user_msg_rewrite, question)
                entry = {
                    'question': row['question'],
                    'cot_right': cot_right,
                    'cot_wrong': cot_wrong
                }
                results.append(entry)
            df = pd.DataFrame(results)
            df.to_csv(ex_cut_file, index=False)
            print('Finished generating cutlery examples')
        
        # Generate plate examples if needed
        ex_plat_file = f'table_setting/examples/table_setting_multichoice_cot_plat_examples_{prompter.model_name}.csv'
        if not os.path.isfile(ex_plat_file):
            results = []
            log_cut = pd.read_csv(f'table_setting/logs/{prompter.model_name}/{prompter.model_name}_plate_selfcon.csv', delimiter=',', on_bad_lines='skip', nrows=10)
            for index, row in tqdm(log_cut.iterrows(),
                                f'Prompting {prompter.model_name} to generate Table Setting task plate examples'):
                corr_plat = row['correct_answer']
                # get correct cot
                cot_right = ''
                for i in range(n_cot):
                    answ = row[f'answer_{i}']
                    if answ == corr_plat:
                        cot_right = row[f'cot_{i}']
                        break
                if cot_right == '': continue

                question = f'Right answer:\n{cot_right}\nWrong answer:'
                cot_wrong = prompter.prompt_model(system_msg_rewrite, user_msg_rewrite, question)
                entry = {
                    'question': row['question'],
                    'cot_right': cot_right,
                    'cot_wrong': cot_wrong
                }
                results.append(entry)
            df = pd.DataFrame(results)
            df.to_csv(ex_plat_file, index=False)
            print('Finished generating plate examples')
        
        # Load examples
        examples = pd.read_csv(ex_cut_file, delimiter=',', on_bad_lines='skip', nrows=n_ex)
        ex_cut_str = ''
        for index, row in examples.iterrows():
            question = row['question']
            cot_right = row['cot_right']
            cot_wrong = row['cot_wrong']
            ex_cut_str = ex_cut_str + f'Question: {question}\n\nRight Explanation: {cot_right}\n\nWrong Explanation: {cot_wrong}\n\n'

        examples = pd.read_csv(ex_plat_file, delimiter=',', on_bad_lines='skip', nrows=n_ex)
        ex_plat_str = ''
        for index, row in examples.iterrows():
            question = row['question']
            cot_right = row['cot_right']
            cot_wrong = row['cot_wrong']
            ex_plat_str = ex_plat_str + f'Question: {question}\n\nRight Explanation: {cot_right}\n\nWrong Explanation: {cot_wrong}\n\n'
        
        # few shot prompting
        data = pd.read_csv('table_setting/combined_prolific_data.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        results = []
        logs_cut = []
        logs_plat = []
        for index, row in tqdm(data.iterrows(), f'Prompting {prompter.model_name} for the Table Setting task with Contrastive CoT Prompting'):
            # setup meal name & get gold standard data
            meal = row['name']
            plate = get_fitting_plate(row)
            utensils = get_utensils(row)
            tup = TableSettingModelResult(meal, plate, utensils)

            # prompt for cutlery
            question = f'Here are a few examples:\n{ex_cut_str}Meal: {meal}\nCutlery:'
            res = prompter.prompt_model(system_msg, user_msg_cut_meta, question)
            pred_cut = transform_utensil_prediction_new(res)
            tup.add_predicted_utensils(pred_cut)
            log_cut = BasicLogEntry(question, res, pred_cut, utensils)

            # prompt for plate
            question = f'Here are a few examples:\n{ex_plat_str}Meal: {meal}\nPlate:'
            res = prompter.prompt_model(system_msg, user_msg_plat_meta, question)
            pred_plat = transform_plate_prediction_new(res)
            tup.add_predicted_plate(pred_plat)
            log_plat = BasicLogEntry(question, res, pred_plat, utensils)

            results.append(tup)
            logs_cut.append(log_cut)
            logs_plat.append(log_plat)
        write_model_results_to_file(results, prompter.model_name, 'contr', 'table_setting')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_contr'), 'table_setting')
        write_log_to_file(logs_cut, prompter.model_name, 'cutlery_contr', 'table_setting')
        write_log_to_file(logs_plat, prompter.model_name, 'plate_contr', 'table_setting')


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


# Answer tends to be not in a single line, but in a block seperated by newlines
# Search lines until first time no answer is found -> block finished
def transform_utensil_prediction_new(pred: str) -> [Utensil]:
    # LLM response line by line
    split = pred.splitlines()
    split.reverse()

    res = set()
    found = False
    # Scan for possible answer
    for i in range(len(split)):
        line_found = False
        for utensil in Utensil:
            if utensil.lower() in split[i].lower():
                found = True
                line_found = True
                res.add(utensil)
        # Stop if line after last hit does not contain answer
        if found and not line_found:
            break
        
    if len(res) == 0:
        print('No viable utensils found!')
    return list(res)


def transform_plate_prediction_new(pred: str) -> Plate:
    choices = [plate for plate in Plate]
    plate = transform_prediction(pred, choices)
    if plate == 'None':
        return Plate.NONE
    else:
        return Plate(plate)


def calculate_average(results: [TableSettingModelResult], model: str):
    average = {met: 0 for met in ['acc', 'jacc']}
    for res in results:
        if res.get_plate_pred_correctness():
            average['acc'] += 1
        average['jacc'] += res.get_jaccard_for_utensils()
    new_row = pd.Series(
        {'model': model, 'acc': (average['acc'] / len(results)), 'jacc': (average['jacc'] / len(results))})
    return new_row.to_frame().T
