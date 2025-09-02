import ast
import random

import pandas as pd
from tqdm import tqdm

from meta_reasoning.prompting.meta_reason_model_result import MetaReasoningMultiChoiceResult
from tidy_up.prompting.tidy_up_result import TidyUpMultiChoiceResult
from utils.prompter import Prompter
from utils.result_writer import add_to_model_overview, write_model_results_to_file
from utils.formatting import transform_prediction_meta_single, transform_prediction_selfcon_single, majority_vote

system_msg = 'Imagine you are to create a robot for a specific household task.'
user_msg = 'What is the single hardware configuration from the given list that you think is the most suitable to execute the task? Please only answer with the complete configuration you chose.'

def prompt_all_models(prompters: [Prompter]):
    for prompter in prompters:
        results = []
        questions = pd.read_csv('meta_reasoning/meta_reasoning_multi_questions_small.csv', delimiter=',', on_bad_lines='skip')
        questions['Wrong_Configurations'] = questions['Wrong_Configurations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Meta-Reasoning task'):
            task = row['Task']
            corr_conf = row['Correct_Configuration']
            choices = row['Wrong_Configurations'] + [corr_conf]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Task: {task}\nConfigurations: {choices_string}\nYour Choice:'
            res = prompter.prompt_model(system_msg, user_msg, question)
            pred_conf = transform_prediction_meta_single(res, choices)
            tup = MetaReasoningMultiChoiceResult(task, corr_conf, pred_conf, choices)
            results.append(tup)
        write_model_results_to_file(results, prompter.model_name, 'meta_reasoning/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name), 'meta_reasoning/results_multi', False)


user_msg_rar = 'What is the single hardware configuration from the given list that you think is the most suitable to execute the task? Reword and elaborate on the inquiry, then answer only with the complete configuration you choose.'

def prompt_all_models_rar(prompters: [Prompter]):
    for prompter in prompters:
        results = []
        questions = pd.read_csv('meta_reasoning/meta_reasoning_multi_questions_small.csv', delimiter=',', on_bad_lines='skip')
        questions['Wrong_Configurations'] = questions['Wrong_Configurations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Meta-Reasoning task'):
            task = row['Task']
            corr_conf = row['Correct_Configuration']
            choices = row['Wrong_Configurations'] + [corr_conf]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Task: {task}\nConfigurations: {choices_string}\nYour Choice:'
            res = prompter.prompt_model(system_msg, user_msg_rar, question)
            pred_conf = transform_prediction_meta_single(res, choices)
            tup = MetaReasoningMultiChoiceResult(task, corr_conf, pred_conf, choices)
            results.append(tup)
        write_model_results_to_file(results, prompter.model_name + '_rar', 'meta_reasoning/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_rar'), 'meta_reasoning/results_multi', False)


user_msg_meta = '''What is the single hardware configuration from the given list that you think is the most suitable to execute the task? As you perform this task, follow these steps:
1. Clarify your understanding of the question.
2. Make a preliminary identification of the right configuration based on degrees of freedom, grippers and mobility.
3. Critically asses your preliminary analysis. If you are unsure about the initial configuration for the robot, try to reasses it.
4. Confirm your final decision on the right robot configuration for the task and explain the reasoning behind your choices.
5. Evaluate your confidence (0-100%) in your analysis and provide an explanation for this confidence level.
Provide the answer in your final response as the complete configuration you chose.
'''

def prompt_all_models_meta(prompters: [Prompter]):
    for prompter in prompters:
        results = []
        questions = pd.read_csv('meta_reasoning/meta_reasoning_multi_questions_small.csv', delimiter=',', on_bad_lines='skip')
        questions['Wrong_Configurations'] = questions['Wrong_Configurations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Meta-Reasoning task'):
            task = row['Task']
            corr_conf = row['Correct_Configuration']
            choices = row['Wrong_Configurations'] + [corr_conf]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Task: {task}\nConfigurations: {choices_string}\nYour Choice:'
            res = prompter.prompt_model(system_msg, user_msg_meta, question)
            pred_conf = transform_prediction_meta_single(res, choices)
            tup = MetaReasoningMultiChoiceResult(task, corr_conf, pred_conf, choices)
            results.append(tup)
        write_model_results_to_file(results, prompter.model_name + '_meta', 'meta_reasoning/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_meta'), 'meta_reasoning/results_multi', False)


user_msg_selfcon = 'What is the single hardware configuration from the given list that you think is the most suitable to execute the task? Think step by step before answering with the complete configuration you chose.'

MAXIT_selfcon = 2
def prompt_all_models_selfcon(prompters_selfcon: [Prompter], prompter_extract: [Prompter]):
    for prompter, prompter_res in zip(prompters_selfcon, prompter_extract):
        results = []
        questions = pd.read_csv('meta_reasoning/meta_reasoning_multi_questions_small.csv', delimiter=',', on_bad_lines='skip')
        questions['Wrong_Configurations'] = questions['Wrong_Configurations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Meta-Reasoning task'):
            task = row['Task']
            corr_conf = row['Correct_Configuration']
            choices = row['Wrong_Configurations'] + [corr_conf]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Task: {task}\nConfigurations: {choices_string}\nYour Choice:'

            # Sample answers over multiple paths
            answers = []
            for i in range(MAXIT_selfcon):
                res = prompter.prompt_model(system_msg, user_msg_selfcon, question)
                # extract final answer:
                user_msg_extract = f'Your task is to determine the final answer of a given LLM response. The final answer should only contain one of the following configurations: {choices}'
                pred_conf = prompter_res.prompt_model(system_msg, user_msg_extract, res)
                answers.append(pred_conf)
            final_pred = majority_vote(answers)
            tup = MetaReasoningMultiChoiceResult(task, corr_conf, final_pred, choices)
            results.append(tup)
        write_model_results_to_file(results, prompter.model_name + '_selfcon', 'meta_reasoning/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_selfcon'), 'meta_reasoning/results_multi', False)


user_msg_initial = 'What is the single hardware configuration from the given list that you think is the most suitable to execute the task?'
user_msg_feedback = 'Provide Feedback on the answer:'
user_msg_refine = 'Improve upon the answer based on the feedback:'
user_msg_final = 'Please provide your final answer. The answer should only contain the configuration of your choosing.'

MAXIT_selfref = 2
def prompt_all_models_selfref(prompters: [Prompter]):
    for prompter in prompters:
        results = []
        questions = pd.read_csv('meta_reasoning/meta_reasoning_multi_questions_small.csv', delimiter=',', on_bad_lines='skip')
        questions['Wrong_Configurations'] = questions['Wrong_Configurations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Meta-Reasoning task'):
            task = row['Task']
            corr_conf = row['Correct_Configuration']
            choices = row['Wrong_Configurations'] + [corr_conf]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Task: {task}\nConfigurations: {choices_string}\nYour answer:'

            # inital answer
            initial = prompter.prompt_model(system_msg, user_msg_initial, question)
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

            # final answer
            question = question + 'Your Choice:'
            final_pred = prompter.prompt_model(system_msg, user_msg_final, question)
            pred_conf = transform_prediction_meta_single(final_pred, choices)

            tup = MetaReasoningMultiChoiceResult(task, corr_conf, pred_conf, choices)
            results.append(tup)
        write_model_results_to_file(results, prompter.model_name + '_selfref', 'meta_reasoning/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_selfref'), 'meta_reasoning/results_multi', False)


user_msg_principle = 'Your task is to extract the underlying concepts and principles that should be considered when selecting hardware configurations from a given list.'

def prompt_all_models_stepback(prompters: [Prompter]):
    for prompter in prompters:
        results = []
        questions = pd.read_csv('meta_reasoning/meta_reasoning_multi_questions_small.csv', delimiter=',', on_bad_lines='skip')
        questions['Wrong_Configurations'] = questions['Wrong_Configurations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Meta-Reasoning task'):
            task = row['Task']
            corr_conf = row['Correct_Configuration']
            choices = row['Wrong_Configurations'] + [corr_conf]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])

            # Get higher level principles
            question = f'Task: {task}\nConfigurations: {choices_string}\nPrinciples:'
            principles = prompter.prompt_model(system_msg, user_msg_principle, question)
            
            # Get answer based on principles
            user_msg_stepback= f'What is the single hardware configuration from the given list that you think is the most suitable to execute the task? Answer the question step by step using the following principles:\n{principles}\n Provide your final answer as only the complete configuration of your choosing.'
            question = f'Task: {task}\nConfigurations: {choices_string}\nYour Choice:'
            res = prompter.prompt_model(system_msg, user_msg_stepback, question)
            
            pred_conf = transform_prediction_selfcon_single(res, choices)
            tup = MetaReasoningMultiChoiceResult(task, corr_conf, pred_conf, choices)
            results.append(tup)
        write_model_results_to_file(results, prompter.model_name + '_stepback', 'meta_reasoning/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_stepback'), 'meta_reasoning/results_multi', False)


def calculate_average(results: [TidyUpMultiChoiceResult], model: str):
    average = {met: 0 for met in ['acc']}
    for res in results:
        if res.get_pred_correctness():
            average['acc'] += 1
    new_row = pd.Series(
        {'model': model, 'acc': (average['acc'] / len(results))})
    return new_row.to_frame().T
