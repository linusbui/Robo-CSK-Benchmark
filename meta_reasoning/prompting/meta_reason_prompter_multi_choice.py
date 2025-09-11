import ast
import random

import pandas as pd
from tqdm import tqdm
import os.path

from meta_reasoning.prompting.meta_reason_model_result import MetaReasoningMultiChoiceResult
from tidy_up.prompting.tidy_up_result import TidyUpMultiChoiceResult
from utils.prompter import Prompter
from utils.result_writer import add_to_model_overview, write_model_results_to_file
from utils.formatting import transform_prediction_meta_single, transform_prediction_selfcon_single, majority_vote
from utils.logging import BasicLogEntry, StepbackLogEntry, SgiclLogEntry, write_log_to_file, write_general_log_to_file

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
            tup = MetaReasoningMultiChoiceResult(task, corr_conf, res, choices)
            results.append(tup)
        write_model_results_to_file(results, prompter.model_name, 'meta_reasoning/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name), 'meta_reasoning/results_multi', False)


user_msg_rar = 'What is the single hardware configuration from the given list that you think is the most suitable to execute the task? Reword and elaborate on the inquiry, then answer only with the complete configuration you choose.'

def prompt_all_models_rar(prompters: [Prompter], num_runs: int):
    for prompter in prompters:
        results = []
        logs = []
        questions = pd.read_csv('meta_reasoning/meta_reasoning_multi_questions.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        questions['Wrong_Configurations'] = questions['Wrong_Configurations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Meta-Reasoning task with RaR Prompting'):
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
            log = BasicLogEntry(question, res, pred_conf, corr_conf)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name + '_rar', 'meta_reasoning/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_rar'), 'meta_reasoning/results_multi', False)
        write_log_to_file(logs, prompter.model_name + '_rar', 'meta_reasoning')


user_msg_meta = '''What is the single hardware configuration from the given list that you think is the most suitable to execute the task? As you perform this task, follow these steps:
1. Clarify your understanding of the question.
2. Make a preliminary identification of the right configuration based on degrees of freedom, grippers and mobility.
3. Critically asses your preliminary analysis. If you are unsure about the initial configuration for the robot, try to reasses it.
4. Confirm your final decision on the right robot configuration for the task and explain the reasoning behind your choices.
5. Evaluate your confidence (0-100%) in your analysis and provide an explanation for this confidence level.
Provide the answer in your final response as the complete configuration you chose.
'''

def prompt_all_models_meta(prompters: [Prompter], num_runs: int):
    for prompter in prompters:
        results = []
        logs = []
        questions = pd.read_csv('meta_reasoning/meta_reasoning_multi_questions.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        questions['Wrong_Configurations'] = questions['Wrong_Configurations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Meta-Reasoning task with Metacognitive Prompting'):
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
            log = BasicLogEntry(question, res, pred_conf, corr_conf)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name + '_meta', 'meta_reasoning/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_meta'), 'meta_reasoning/results_multi', False)
        write_log_to_file(logs, prompter.model_name + '_meta', 'meta_reasoning')


user_msg_selfcon = 'What is the single hardware configuration from the given list that you think is the most suitable to execute the task? Think step by step before answering with the complete configuration you chose.'

MAXIT_selfcon = 2
def prompt_all_models_selfcon(prompters_selfcon: [Prompter], prompter_extract: [Prompter], num_runs: int):
    for prompter, prompter_res in zip(prompters_selfcon, prompter_extract):
        results = []
        logs = []
        questions = pd.read_csv('meta_reasoning/meta_reasoning_multi_questions.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        questions['Wrong_Configurations'] = questions['Wrong_Configurations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Meta-Reasoning task with Self-Consistency Prompting'):
            task = row['Task']
            corr_conf = row['Correct_Configuration']
            choices = row['Wrong_Configurations'] + [corr_conf]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Task: {task}\nConfigurations: {choices_string}\nYour Choice:'

            log = {'question': question}
            # Sample answers over multiple paths
            answers = []
            for i in range(MAXIT_selfcon):
                res = prompter.prompt_model(system_msg, user_msg_selfcon, question)
                # try to extract result classicaly
                pred_conf = transform_prediction_selfcon_single(res, choices)
                if pred_conf == 'None':
                    # extract final answer with LLM
                    user_msg_extract = f'Your task is to determine the final answer of a given LLM response. The final answer should only contain one of the following configurations: {choices}'
                    pred_conf = prompter_res.prompt_model(system_msg, user_msg_extract, res)
                answers.append(pred_conf)
                log.update({f'cot_{i}': res,
                            f'answer_{i}': pred_conf})
            final_pred = majority_vote(answers)
            tup = MetaReasoningMultiChoiceResult(task, corr_conf, final_pred, choices)
            results.append(tup)
            log.update({'final_answer': final_pred,
                         'correct_answer': corr_conf})
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name + '_selfcon', 'meta_reasoning/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_selfcon'), 'meta_reasoning/results_multi', False)
        write_general_log_to_file(logs, prompter.model_name + '_selfcon', 'meta_reasoning')


user_msg_initial = 'What is the single hardware configuration from the given list that you think is the most suitable to execute the task?'
user_msg_feedback = "Provide Feedback on the answer. At the end, score the answer from 1 to 5. 1 means that the answer is completely wrong, 4 or above means that the answer is right."
user_msg_refine = 'Improve upon the answer based on the feedback:'

MAXIT_selfref = 2
def prompt_all_models_selfref(prompters: [Prompter], num_runs: int):
    for prompter in prompters:
        results = []
        logs = []
        questions = pd.read_csv('meta_reasoning/meta_reasoning_multi_questions.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        questions['Wrong_Configurations'] = questions['Wrong_Configurations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Meta-Reasoning task with Self-Refine Prompting'):
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

                # Stopping condition
                ind = feedback.lower().find('score')
                if not ind == -1:
                    end = feedback[ind:]
                    if '4' in end or '5' in end:
                        break

                # refine
                question = question + '\nImprovement:'
                refine = prompter.prompt_model(system_msg, user_msg_refine, question)
                question = question + f'\n{refine}\n'

            # final answer
            user_msg_final = f'Please provide your final answer based on the given feedback-answer iterations. The answer should only contain one of the following configurations: {choices}'
            question = question + '\nYour Choice:'
            final_pred = prompter.prompt_model(system_msg, user_msg_final, question)
            pred_conf = transform_prediction_meta_single(final_pred, choices)

            tup = MetaReasoningMultiChoiceResult(task, corr_conf, pred_conf, choices)
            results.append(tup)
            log = BasicLogEntry(question, final_pred, pred_conf, corr_conf)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name + '_selfref', 'meta_reasoning/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_selfref'), 'meta_reasoning/results_multi', False)
        write_log_to_file(logs, prompter.model_name + '_selfref', 'meta_reasoning')


user_msg_principle = 'Your task is to extract the underlying concepts and principles that should be considered when selecting hardware configurations from a given list.'

def prompt_all_models_stepback(prompters: [Prompter], num_runs: int):
    for prompter in prompters:
        results = []
        logs = []
        questions = pd.read_csv('meta_reasoning/meta_reasoning_multi_questions.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        questions['Wrong_Configurations'] = questions['Wrong_Configurations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Meta-Reasoning task with Stepback Prompting'):
            task = row['Task']
            corr_conf = row['Correct_Configuration']
            choices = row['Wrong_Configurations'] + [corr_conf]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])

            # Get higher level principles
            p_question = f'Task: {task}\nConfigurations: {choices_string}\nPrinciples:'
            principles = prompter.prompt_model(system_msg, user_msg_principle, p_question)
            
            # Get answer based on principles
            user_msg_stepback= f'What is the single hardware configuration from the given list that you think is the most suitable to execute the task? Answer the question step by step using the following principles:\n{principles}\n Provide your final answer as only the complete configuration of your choosing.'
            question = f'Task: {task}\nConfigurations: {choices_string}\nYour Choice:'
            res = prompter.prompt_model(system_msg, user_msg_stepback, question)
            
            pred_conf = transform_prediction_selfcon_single(res, choices)
            tup = MetaReasoningMultiChoiceResult(task, corr_conf, pred_conf, choices)
            results.append(tup)
            log = StepbackLogEntry(p_question, principles, question, res, pred_conf, corr_conf)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name + '_stepback', 'meta_reasoning/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_stepback'), 'meta_reasoning/results_multi', False)
        write_log_to_file(logs, prompter.model_name + '_stepback', 'meta_reasoning')


system_msg_example = 'You are helping to create questions regarding household environments.'
user_msg_example = 'For the given hardware configuration, generate a task that can be executed by a robot with that configuration. Answer in one short sentence only.'

NUM_EXAMPLES = 8
def prompt_all_models_sgicl(prompters: [Prompter], num_runs: int):
    for prompter in prompters:
        # Generate examples if needed
        ex_file = f'meta_reasoning/examples/meta_reasoning_multi_questions_examples_{prompter.model_name}.csv'
        if not os.path.isfile(ex_file):
            results = []
            questions = pd.read_csv('meta_reasoning/meta_reasoning_multi_questions.csv', delimiter=',', on_bad_lines='skip', nrows=15)
            for index, row in tqdm(questions.iterrows(),
                                f'Prompting {prompter.model_name} to generate Meta-Rasoning task examples'):
                task = row['Task']
                corr_conf = row['Correct_Configuration']
                question = f'Tool: {corr_conf}\nGenerate a task: {task}\nGenerate a task:'
                pred_task = prompter.prompt_model(system_msg_example, user_msg_example, question)
                entry = {
                    'Task': pred_task,
                    'Configuration': corr_conf
                }
                results.append(entry)
            df = pd.DataFrame(results)
            df.to_csv(ex_file, index=False)
            print('Finished generating examples')

        # Load examples
        examples = pd.read_csv(ex_file, delimiter=',', on_bad_lines='skip', nrows=NUM_EXAMPLES)
        ex_str = ''
        for index, row in examples.iterrows():
            task = row['Task']
            conf = row['Configuration']
            ex_str = ex_str + f'Task: {task}\nConfiguration: {conf}\n'

        # few shot prompting
        results = []
        logs = []
        questions = pd.read_csv('meta_reasoning/meta_reasoning_multi_questions.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        questions['Wrong_Configurations'] = questions['Wrong_Configurations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Meta-Reasoning task with SG-ICL Prompting'):
            task = row['Task']
            corr_conf = row['Correct_Configuration']
            choices = row['Wrong_Configurations'] + [corr_conf]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Here are a few examples:\n{ex_str}Task: {task}\nConfigurations: {choices_string}\nYour Choice:'
            res = prompter.prompt_model(system_msg, user_msg, question)
            tup = MetaReasoningMultiChoiceResult(task, corr_conf, res, choices)
            results.append(tup)
            log = SgiclLogEntry(question, res, corr_conf)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name + '_sgicl', 'meta_reasoning/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_sgicl'), 'meta_reasoning/results_multi', False)
        write_log_to_file(logs, prompter.model_name + '_sgicl', 'meta_reasoning')


system_msg_rewrite = 'You are helping in rewriting answers to questions regarding household environments.'
user_msg_rewrite = 'Rewrite the given answer by swapping key points with wrong facts leading to a wrong final answer. Keep the overall structure the same.'

NUM_COT = 4
def prompt_all_models_contr(prompters: [Prompter], num_runs: int):
    for prompter in prompters:
        # Generate examples if needed
        ex_file = f'meta_reasoning/examples/meta_reasoning_multi_cot_examples_{prompter.model_name}.csv'
        if not os.path.isfile(ex_file):
            results = []
            log = pd.read_csv(f'meta_reasoning/logs/{prompter.model_name}_selfcon.csv', delimiter=',', on_bad_lines='skip', nrows=10)
            for index, row in tqdm(log.iterrows(),
                                f'Prompting {prompter.model_name} to generate Meta-Rasoning task examples'):
                corr_conf = row['correct_answer']
                # get correct cot
                cot_right = ''
                for i in range(MAXIT_selfcon):
                    answ = row[f'answer_{i}']
                    if answ == corr_conf:
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
            df.to_csv(ex_file, index=False)
            print('Finished generating examples')

        # Load examples
        examples = pd.read_csv(ex_file, delimiter=',', on_bad_lines='skip', nrows=NUM_COT)
        ex_str = ''
        for index, row in examples.iterrows():
            question = row['question']
            cot_right = row['cot_right']
            cot_wrong = row['cot_wrong']
            ex_str = ex_str + f'Question: {question}\nRight Explanation: {cot_right}\nWrong Explanation: {cot_wrong}\n'
        
        # few shot prompting
        results = []
        logs = []
        questions = pd.read_csv('meta_reasoning/meta_reasoning_multi_questions.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        questions['Wrong_Configurations'] = questions['Wrong_Configurations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Meta-Reasoning task with Contrastive CoT Prompting'):
            task = row['Task']
            corr_conf = row['Correct_Configuration']
            choices = row['Wrong_Configurations'] + [corr_conf]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Here are a few examples:\n{ex_str}Task: {task}\nConfigurations: {choices_string}\nYour Choice:'
            res = prompter.prompt_model(system_msg, user_msg, question)
            tup = MetaReasoningMultiChoiceResult(task, corr_conf, res, choices)
            results.append(tup)
            log = SgiclLogEntry(question, res, corr_conf)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name + '_contr', 'meta_reasoning/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_contr'), 'meta_reasoning/results_multi', False)
        write_log_to_file(logs, prompter.model_name + '_contr', 'meta_reasoning')


def calculate_average(results: [TidyUpMultiChoiceResult], model: str):
    average = {met: 0 for met in ['acc']}
    for res in results:
        if res.get_pred_correctness():
            average['acc'] += 1
    new_row = pd.Series(
        {'model': model, 'acc': (average['acc'] / len(results))})
    return new_row.to_frame().T