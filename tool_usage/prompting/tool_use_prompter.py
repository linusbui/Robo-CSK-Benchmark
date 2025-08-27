import ast
import random

import pandas as pd
from tqdm import tqdm

from tool_usage.prompting.tool_use_result import ToolSubstitutionResult
from utils.prompter import Prompter
from utils.result_writer import add_to_model_overview, write_model_results_to_file
from utils.formatting import transform_prediction_meta_single, majority_vote


system_msg = 'Imagine you are a robot in a household environment being confronted with a task and a list of tools.'
user_msg = 'What is the single tool from the given list that you think is most suitable to help you execute your task? Please only answer with the tool you chose.'

def prompt_all_models(prompters: [Prompter]):
    for prompter in prompters:
        results = []
        questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions_small.csv', delimiter=',', on_bad_lines='skip')
        questions['Wrong_Tools'] = questions['Wrong_Tools'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(), f'Prompting {prompter.model_name} for the Tool Usage task'):
            task = row['Task']
            affordance = row['Affordance']
            corr_tool = row['Correct_Tool']
            choices = row['Wrong_Tools'] + [corr_tool]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Task: {task}\nTools: {choices_string}\nYour Choice:'
            res = prompter.prompt_model(system_msg, user_msg, question)
            pred_tool = transform_prediction_meta_single(res, choices)
            tup = ToolSubstitutionResult(task, affordance, corr_tool, pred_tool, choices)
            results.append(tup)
        write_model_results_to_file(results, prompter.model_name, 'tool_usage')
        add_to_model_overview(calculate_average(results, prompter.model_name), 'tool_usage')


user_msg_rar = 'What is the single tool from the given list that you think is most suitable to help you execute your task? Please only answer with the tool you chose. Reword and elaborate on the inquiry, then provide your final answer.'

def prompt_all_models_rar(prompters: [Prompter]):
    for prompter in prompters:
        results = []
        questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions_small.csv', delimiter=',', on_bad_lines='skip')
        questions['Wrong_Tools'] = questions['Wrong_Tools'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(), f'Prompting {prompter.model_name} for the Tool Usage task'):
            task = row['Task']
            affordance = row['Affordance']
            corr_tool = row['Correct_Tool']
            choices = row['Wrong_Tools'] + [corr_tool]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Task: {task}\nTools: {choices_string}\nYour Choice:'
            res = prompter.prompt_model(system_msg, user_msg_rar, question)
            pred_tool = transform_prediction_meta_single(res, choices)
            tup = ToolSubstitutionResult(task, affordance, corr_tool, pred_tool, choices)
            results.append(tup)
        write_model_results_to_file(results, prompter.model_name + '_rar', 'tool_usage')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_rar'), 'tool_usage')


user_msg_meta = '''
What is the single tool from the given list that you think is most suitable to help you execute your task? As you perform this task, follow these steps:
1. Clarify your understanding of the question.
2. Make a preliminary identification of the single tool that may be most suitable to help you execute your task.
3. Critically asses your preliminary analysis. If you are unsure about the initial assesment of right tool used for the task, try to reasses it.
4. Confirm your final decision on the most suitable tool to help you execute your task.
5. Evaluate your confidence (0-100%) in your analysis and provide an explanation for this confidence level.
Provide the answer in your final response as only the tool you choose.
'''

def prompt_all_models_meta(prompters: [Prompter]):
    for prompter in prompters:
        results = []
        questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions_small.csv', delimiter=',', on_bad_lines='skip')
        questions['Wrong_Tools'] = questions['Wrong_Tools'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(), f'Prompting {prompter.model_name} for the Tool Usage task'):
            task = row['Task']
            affordance = row['Affordance']
            corr_tool = row['Correct_Tool']
            choices = row['Wrong_Tools'] + [corr_tool]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Task: {task}\nTools: {choices_string}\nYour Choice:'
            res = prompter.prompt_model(system_msg, user_msg_meta, question)
            pred_tool = transform_prediction_meta_single(res, choices)
            tup = ToolSubstitutionResult(task, affordance, corr_tool, pred_tool, choices)
            results.append(tup)
        write_model_results_to_file(results, prompter.model_name + '_meta', 'tool_usage')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_meta'), 'tool_usage')


user_msg_selfcon = 'What is the single tool from the given list that you think is most suitable to help you execute your task? Think step by step before answering with the single tool of your choosing.'

MAXIT_selfcon = 2
def prompt_all_models_selfcon(prompters: [Prompter]):
    for prompter in prompters:
        results = []
        questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions_small.csv', delimiter=',', on_bad_lines='skip')
        questions['Wrong_Tools'] = questions['Wrong_Tools'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(), f'Prompting {prompter.model_name} for the Tool Usage task'):
            task = row['Task']
            affordance = row['Affordance']
            corr_tool = row['Correct_Tool']
            choices = row['Wrong_Tools'] + [corr_tool]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Task: {task}\nTools: {choices_string}\nYour Choice:'

            answers = []
            for i in range(MAXIT_selfcon):
                res = prompter.prompt_model(system_msg, user_msg_selfcon, question)
                pred_tool = transform_prediction_meta_single(res, choices)
                answers.append(pred_tool)
            tup = ToolSubstitutionResult(task, affordance, corr_tool, majority_vote(answers), choices)
            results.append(tup)
        write_model_results_to_file(results, prompter.model_name + '_selfcon', 'tool_usage')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_selfcon'), 'tool_usage')


user_msg_initial = 'What is the single tool from the given list that you think is most suitable to help you execute your task? Please only answer with the tool you chose.'
user_msg_feedback = 'Provide Feedback on the answer:'
user_msg_refine = 'Improve upon the answer based on the feedback:'
user_msg_final = 'Please provide your final answer. The answer should only contain the single tool of your choosing.'

MAXIT_selfref = 2
def prompt_all_models_selfref(prompters: [Prompter]):
    for prompter in prompters:
        results = []
        questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions_small.csv', delimiter=',', on_bad_lines='skip')
        questions['Wrong_Tools'] = questions['Wrong_Tools'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(), f'Prompting {prompter.model_name} for the Tool Usage task'):
            task = row['Task']
            affordance = row['Affordance']
            corr_tool = row['Correct_Tool']
            choices = row['Wrong_Tools'] + [corr_tool]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])

            # initial prompt
            question = f'Task: {task}\nTools: {choices_string}\nYour Choice:'
            initial = prompter.prompt_model(system_msg, user_msg_initial, question)

            question = question + f'\n{initial}\n'

            # Refine - Feedback loop
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

            pred_tool = transform_prediction_meta_single(final_pred, choices)
            tup = ToolSubstitutionResult(task, affordance, corr_tool, pred_tool, choices)

            results.append(tup)
        write_model_results_to_file(results, prompter.model_name + '_selfref', 'tool_usage')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_selfref'), 'tool_usage')



def calculate_average(results: [ToolSubstitutionResult], model: str):
    average = {met: 0 for met in ['acc']}
    for res in results:
        if res.get_pred_correctness():
            average['acc'] += 1
    new_row = pd.Series(
        {'model': model, 'acc': (average['acc'] / len(results))})
    return new_row.to_frame().T
