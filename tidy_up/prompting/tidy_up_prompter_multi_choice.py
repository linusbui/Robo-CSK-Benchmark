import ast
import random

import pandas as pd
from tqdm import tqdm

from tidy_up.prompting.tidy_up_result import TidyUpMultiChoiceResult
from utils.prompter import Prompter
from utils.result_writer import add_to_model_overview, write_model_results_to_file
from utils.formatting import transform_prediction_meta_single, majority_vote

system_msg = 'Imagine you are a robot tidying up a household environment, being confronted with an object and a possible list of locations to put it.'
user_msg = 'What is the single location from the given list that you think is the most suitable place to put the object? Please only answer with the location you chose.'

def prompt_all_models(prompters: [Prompter]):
    for prompter in prompters:
        results = []
        questions = pd.read_csv('tidy_up/tidy_up_multichoice_small.csv', delimiter=',', on_bad_lines='skip')
        questions['Wrong_Locations'] = questions['Wrong_Locations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Tidy Up task'):
            obj = row['Object']
            corr_loc = row['Correct_Location']
            choices = row['Wrong_Locations'] + [corr_loc]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Object: {obj}\nLocations: {choices_string}\nYour Choice:'
            res = prompter.prompt_model(system_msg, user_msg, question)
            pred_loc = transform_prediction_meta_single(res, choices)
            tup = TidyUpMultiChoiceResult(obj, corr_loc, pred_loc, choices)
            results.append(tup)
        write_model_results_to_file(results, prompter.model_name, 'tidy_up/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name), 'tidy_up/results_multi', False)


user_msg_rar = 'What is the single location from the given list that you think is the most suitable place to put the object? Reword and elaborate on the inquiry, then answer only with the location you chose.'

def prompt_all_models_rar(prompters: [Prompter]):
    for prompter in prompters:
        results = []
        questions = pd.read_csv('tidy_up/tidy_up_multichoice_small.csv', delimiter=',', on_bad_lines='skip')
        questions['Wrong_Locations'] = questions['Wrong_Locations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Tidy Up task'):
            obj = row['Object']
            corr_loc = row['Correct_Location']
            choices = row['Wrong_Locations'] + [corr_loc]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Object: {obj}\nLocations: {choices_string}\nYour Choice:'
            res = prompter.prompt_model(system_msg, user_msg_rar, question)
            pred_loc = transform_prediction_meta_single(res, choices)
            tup = TidyUpMultiChoiceResult(obj, corr_loc, pred_loc, choices)
            results.append(tup)
        write_model_results_to_file(results, prompter.model_name + '_rar', 'tidy_up/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_rar'), 'tidy_up/results_multi', False)


user_msg_meta = '''
What is the single location from the given list that you think is the most suitable place to put the object? As you perform this task, follow these steps:
1. Clarify your understanding of the question.
2. Make a preliminary identification of the single most suitable location for the object.
3. Critically asses your preliminary analysis. If you are unsure about the initial assesment of the most suitable location of the object, try to reasses it.
4. Confirm your final decision on the most suitable place to put the object.
5. Evaluate your confidence (0-100%) in your analysis and provide an explanation for this confidence level.
Provide the answer in your final response as only the location you choose.
'''

def prompt_all_models_meta(prompters: [Prompter]):
    for prompter in prompters:
        results = []
        questions = pd.read_csv('tidy_up/tidy_up_multichoice_small.csv', delimiter=',', on_bad_lines='skip')
        questions['Wrong_Locations'] = questions['Wrong_Locations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Tidy Up task'):
            obj = row['Object']
            corr_loc = row['Correct_Location']
            choices = row['Wrong_Locations'] + [corr_loc]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Object: {obj}\nLocations: {choices_string}\nYour Choice:'
            res = prompter.prompt_model(system_msg, user_msg_meta, question)
            pred_loc = transform_prediction_meta_single(res, choices)
            tup = TidyUpMultiChoiceResult(obj, corr_loc, pred_loc, choices)
            results.append(tup)
        write_model_results_to_file(results, prompter.model_name + '_meta', 'tidy_up/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_meta'), 'tidy_up/results_multi', False)


user_msg_selfcon = 'What is the single location from the given list that you think is the most suitable place to put the object? Think step by step before answering with the single location of your choosing.'

MAXIT_selfcon = 2
def prompt_all_models_selfcon(prompters: [Prompter]):
    for prompter in prompters:
        results = []
        questions = pd.read_csv('tidy_up/tidy_up_multichoice_small.csv', delimiter=',', on_bad_lines='skip')
        questions['Wrong_Locations'] = questions['Wrong_Locations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Tidy Up task'):
            obj = row['Object']
            corr_loc = row['Correct_Location']
            choices = row['Wrong_Locations'] + [corr_loc]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Object: {obj}\nLocations: {choices_string}\nYour Choice:'

            answers = []
            for i in range(MAXIT_selfcon):
                res = prompter.prompt_model(system_msg, user_msg_selfcon, question)
                pred_loc = transform_prediction_meta_single(res, choices)
                answers.append(pred_loc)
            tup = TidyUpMultiChoiceResult(obj, corr_loc, majority_vote(answers), choices)
            results.append(tup)
        write_model_results_to_file(results, prompter.model_name + '_selfcon', 'tidy_up/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_selfcon'), 'tidy_up/results_multi', False)


user_msg_initial = 'What is the single location from the given list that you think is the most suitable place to put the object?'
user_msg_feedback = 'Provide Feedback on the answer:'
user_msg_refine = 'Improve upon the answer based on the feedback:'
user_msg_final = 'Please provide your final answer. The answer should only contain the single location of your choosing.'

MAXIT_selfref = 2
def prompt_all_models_selfref(prompters: [Prompter]):
    for prompter in prompters:
        results = []
        questions = pd.read_csv('tidy_up/tidy_up_multichoice_small.csv', delimiter=',', on_bad_lines='skip')
        questions['Wrong_Locations'] = questions['Wrong_Locations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Tidy Up task'):
            obj = row['Object']
            corr_loc = row['Correct_Location']
            choices = row['Wrong_Locations'] + [corr_loc]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])

            question = f'Object: {obj}\nLocations: {choices_string}\nYour Choice:'
            initial = prompter.prompt_model(system_msg, user_msg_initial, question)
            question = question + f'\n{initial}\n'

            for i in range(MAXIT_selfref):
                # feedback 
                question = question + '\nYour Feedback:'
                feedback = prompter.prompt_model(system_msg, user_msg_feedback, question)
                question = question + f'\n{feedback}'

                # refine
                question = question + '\nImprovement:'
                refine = prompter.prompt_model(system_msg, user_msg_refine, question)
                question = question + f'\n{refine}\n'

            # final answer
            question = question + 'Your Choice:'
            final_pred = prompter.prompt_model(system_msg, user_msg_final, question)

            pred_loc = transform_prediction_meta_single(final_pred, choices)
            tup = TidyUpMultiChoiceResult(obj, corr_loc, pred_loc, choices)
            results.append(tup)
        write_model_results_to_file(results, prompter.model_name + '_selfref', 'tidy_up/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_selfref'), 'tidy_up/results_multi', False)


user_msg_principle = 'Your task is to extract the underlying concepts and principles that should be considered when choosing the most suitable place to put the object from of a given list.'

def prompt_all_models_stepback(prompters: [Prompter]):
    for prompter in prompters:
        results = []
        questions = pd.read_csv('tidy_up/tidy_up_multichoice_small.csv', delimiter=',', on_bad_lines='skip')
        questions['Wrong_Locations'] = questions['Wrong_Locations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Tidy Up task'):
            obj = row['Object']
            corr_loc = row['Correct_Location']
            choices = row['Wrong_Locations'] + [corr_loc]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])

            # Get higher level principles
            question = f'Object: {obj}\nLocations: {choices_string}\nPrinciples:'
            principles = prompter.prompt_model(system_msg, user_msg_principle, question)

            # Get answer based on principles
            user_msg_stepback = f'What is the single location from the given list that you think is the most suitable place to put the object? Answer the question step by step using the following principles:\n{principles}\n Provide your final answer as only the single location you choose.'
            question = f'Object: {obj}\nLocations: {choices_string}\nYour Choice:'

            res = prompter.prompt_model(system_msg, user_msg_stepback, question)

            pred_loc = transform_prediction_meta_single(res, choices)
            tup = TidyUpMultiChoiceResult(obj, corr_loc, pred_loc, choices)
            results.append(tup)
        write_model_results_to_file(results, prompter.model_name + '_stepback', 'tidy_up/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_stepback'), 'tidy_up/results_multi', False)



def calculate_average(results: [TidyUpMultiChoiceResult], model: str):
    average = {met: 0 for met in ['acc']}
    for res in results:
        if res.get_pred_correctness():
            average['acc'] += 1
    new_row = pd.Series(
        {'model': model, 'acc': (average['acc'] / len(results))})
    return new_row.to_frame().T