import ast
import random

import pandas as pd
from tqdm import tqdm
import os.path

from tool_usage.prompting.tool_use_result import ToolSubstitutionResult
from utils.prompter import Prompter
from utils.result_writer import add_to_model_overview, write_model_results_to_file
from utils.formatting import transform_prediction_meta_single, majority_vote
from utils.logging import BasicLogEntry, StepbackLogEntry, SgiclLogEntry, write_log_to_file, write_general_log_to_file

system_msg = 'Imagine you are a robot in a household environment being confronted with a task and a list of tools.'
user_msg = 'What is the single tool from the given list that you think is most suitable to help you execute your task? Please only answer with the tool you chose.'

def prompt_all_models(prompters: [Prompter], num_runs:int):
    for prompter in prompters:
        results = []
        questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
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
            tup = ToolSubstitutionResult(task, affordance, corr_tool, res, choices)
            results.append(tup)
        write_model_results_to_file(results, prompter.model_name, '', 'tool_usage')
        add_to_model_overview(calculate_average(results, prompter.model_name), 'tool_usage')


user_msg_rar = 'What is the single tool from the given list that you think is most suitable to help you execute your task? Please only answer with the tool you chose. Reword and elaborate on the inquiry, then provide your final answer.'

def prompt_all_models_rar(prompters: [Prompter], num_runs: int):
    for prompter in prompters:
        results = []
        logs = []
        questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        questions['Wrong_Tools'] = questions['Wrong_Tools'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(), f'Prompting {prompter.model_name} for the Tool Usage task with RaR Prompting'):
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
            log = BasicLogEntry(question, res, pred_tool, corr_tool)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name, 'rar', 'tool_usage')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_rar'), 'tool_usage')
        write_log_to_file(logs, prompter.model_name, 'rar', 'tool_usage')


user_msg_meta = '''
What is the single tool from the given list that you think is most suitable to help you execute your task? As you perform this task, follow these steps:
1. Clarify your understanding of the question.
2. Make a preliminary identification of the single tool that may be most suitable to help you execute your task.
3. Critically asses your preliminary analysis. If you are unsure about the initial assesment of right tool used for the task, try to reasses it.
4. Confirm your final decision on the most suitable tool to help you execute your task.
5. Evaluate your confidence (0-100%) in your analysis and provide an explanation for this confidence level.
Provide the answer in your final response as only the tool you choose.
'''

def prompt_all_models_meta(prompters: [Prompter], num_runs: int):
    for prompter in prompters:
        results = []
        logs = []
        questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        questions['Wrong_Tools'] = questions['Wrong_Tools'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(), f'Prompting {prompter.model_name} for the Tool Usage task with Metacognitive Prompting'):
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
            log = BasicLogEntry(question, res, pred_tool, corr_tool)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name, 'meta', 'tool_usage')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_meta'), 'tool_usage')
        write_log_to_file(logs, prompter.model_name, 'meta', 'tool_usage')


user_msg_selfcon = 'What is the single tool from the given list that you think is most suitable to help you execute your task? Think step by step before answering with the single tool of your choosing.'

MAXIT_selfcon = 2
def prompt_all_models_selfcon(prompters: [Prompter], num_runs: int):
    for prompter in prompters:
        results = []
        logs = []
        questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        questions['Wrong_Tools'] = questions['Wrong_Tools'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(), f'Prompting {prompter.model_name} for the Tool Usage task with Self-Consistency Prompting'):
            task = row['Task']
            affordance = row['Affordance']
            corr_tool = row['Correct_Tool']
            choices = row['Wrong_Tools'] + [corr_tool]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Task: {task}\nTools: {choices_string}\nYour Choice:'

            log = {'question': question}
            answers = []
            for i in range(MAXIT_selfcon):
                res = prompter.prompt_model(system_msg, user_msg_selfcon, question)
                pred_tool = transform_prediction_meta_single(res, choices)
                answers.append(pred_tool)
                log.update({f'cot_{i}': res,
                            f'answer_{i}': pred_tool})
            final_pred = majority_vote(answers)
            tup = ToolSubstitutionResult(task, affordance, corr_tool, final_pred, choices)
            results.append(tup)
            log.update({'final_answer': final_pred,
                        'correct_answer': corr_tool})
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name, 'selfcon', 'tool_usage')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_selfcon'), 'tool_usage')
        write_general_log_to_file(logs, prompter.model_name, 'selfcon', 'tool_usage')


user_msg_initial = 'What is the single tool from the given list that you think is most suitable to help you execute your task? Please only answer with the tool you chose.'
user_msg_feedback = "Provide Feedback on the answer. At the end, score the answer from 1 to 5. 1 means that the answer is completely wrong, 4 or above means that the answer is right."
user_msg_refine = 'Improve upon the answer based on the feedback:'

MAXIT_selfref = 2
def prompt_all_models_selfref(prompters: [Prompter], num_runs: int):
    for prompter in prompters:
        results = []
        logs = []
        questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        questions['Wrong_Tools'] = questions['Wrong_Tools'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(), f'Prompting {prompter.model_name} for the Tool Usage task with Self-Refine Prompting'):
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
            user_msg_final = f'Please provide your final answer based on the given feedback-answer iterations. The answer should only contain one of the following tools: {choices}'
            question = question + '\nYour Choice:'
            final_pred = prompter.prompt_model(system_msg, user_msg_final, question)
            pred_tool = transform_prediction_meta_single(final_pred, choices)

            tup = ToolSubstitutionResult(task, affordance, corr_tool, pred_tool, choices)
            results.append(tup)
            log = BasicLogEntry(question, final_pred, pred_tool, corr_tool)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name, 'selfref', 'tool_usage')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_selfref'), 'tool_usage')
        write_log_to_file(logs, prompter.model_name, 'selfref', 'tool_usage')


system_msg_principle = 'You are given a multiple-choice question. Your task is to extract the underlying concepts and principles involved in choosing the right answer.'
user_msg_principle = 'Only answer with the 5 most important concepts and principles.'

def prompt_all_models_stepback(prompters: [Prompter], num_runs: int):
    for prompter in prompters:
        results = []
        logs = []
        questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        questions['Wrong_Tools'] = questions['Wrong_Tools'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(), f'Prompting {prompter.model_name} for the Tool Usage task with Stepback Prompting'):
            task = row['Task']
            affordance = row['Affordance']
            corr_tool = row['Correct_Tool']
            choices = row['Wrong_Tools'] + [corr_tool]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])

            # Get higher level principles
            p_question = f'Task: {task}\nTools: {choices_string}\nPrinciples:'
            principles = prompter.prompt_model(system_msg_principle, user_msg_principle, p_question)

            # Get answer based on principles
            user_msg_stepback = f'What is the single tool from the given list that you think is most suitable to help you execute your task? Answer the question step by step using the following principles:\n{principles}\nEnd your answer with the tool you chose.'
            question = f'Task: {task}\nTools: {choices_string}\nYour Choice:'

            res = prompter.prompt_model(system_msg, user_msg_stepback, question)

            pred_tool = transform_prediction_meta_single(res, choices)
            tup = ToolSubstitutionResult(task, affordance, corr_tool, pred_tool, choices)
            results.append(tup)
            log = StepbackLogEntry(p_question, principles, question, res, pred_tool, corr_tool)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name, 'stepback', 'tool_usage')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_stepback'), 'tool_usage')
        write_log_to_file(logs, prompter.model_name, 'stepback', 'tool_usage')


system_msg_example = 'You are helping to create questions regarding household environments.'
user_msg_example = 'For the given tool, generate a task that is executed using that tool. Answer in one short sentence only.'

NUM_EXAMPLES = 8
def prompt_all_models_sgicl(prompters: [Prompter], num_runs: int):
    for prompter in prompters:
        # Generate examples if needed
        ex_file = f'tool_usage/examples/tool_usage_multichoice_examples_{prompter.model_name}.csv'
        if not os.path.isfile(ex_file):
            results = []
            questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions.csv', delimiter=',', on_bad_lines='skip', nrows=15)
            for index, row in tqdm(questions.iterrows(),
                                f'Prompting {prompter.model_name} to generate Tool Usage task examples'):
                task = row['Task']
                corr_tool = row['Correct_Tool']
                question = f'Tool: {corr_tool}\nGenerate a task: {task}\nGenerate a task:'
                pred_task = prompter.prompt_model(system_msg_example, user_msg_example, question)
                entry = {
                    'Task': pred_task,
                    'Tool': corr_tool
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
            tool = row['Tool']
            ex_str = ex_str + f'Task: {task}\nTool: {tool}\n'

        # few shot prompting
        results = []
        logs = []
        questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        questions['Wrong_Tools'] = questions['Wrong_Tools'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(), f'Prompting {prompter.model_name} for the Tool Usage task with SG-ICL Prompting'):
            task = row['Task']
            affordance = row['Affordance']
            corr_tool = row['Correct_Tool']
            choices = row['Wrong_Tools'] + [corr_tool]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Here are a few examples:\n{ex_str}Task: {task}\nTools: {choices_string}\nYour Choice:'
            res = prompter.prompt_model(system_msg, user_msg, question)
            tup = ToolSubstitutionResult(task, affordance, corr_tool, res, choices)
            results.append(tup)
            log = SgiclLogEntry(question, res, corr_tool)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name, 'sgicl', 'tool_usage')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_sgicl'), 'tool_usage')
        write_log_to_file(logs, prompter.model_name, 'sgicl', 'tool_usage')


system_msg_rewrite = 'You are helping in rewriting answers to questions regarding household environments.'
user_msg_rewrite = 'Rewrite the given answer by swapping key points with wrong facts leading to a wrong final answer. Keep the overall structure the same.'

NUM_COT = 4
def prompt_all_models_contr(prompters: [Prompter], num_runs: int):
    for prompter in prompters:
        # Generate examples if needed
        ex_file = f'tool_usage/examples/tool_usage_multichoice_cot_examples_{prompter.model_name}.csv'
        if not os.path.isfile(ex_file):
            results = []
            log = pd.read_csv(f'tool_usage/logs/{prompter.model_name}/{prompter.model_name}_selfcon.csv', delimiter=',', on_bad_lines='skip', nrows=10)
            for index, row in tqdm(log.iterrows(),
                                f'Prompting {prompter.model_name} to generate Tool Usage task examples'):
                corr_tool = row['correct_answer']
                # get correct cot
                cot_right = ''
                for i in range(MAXIT_selfcon):
                    answ = row[f'answer_{i}']
                    if answ == corr_tool:
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
        questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        questions['Wrong_Tools'] = questions['Wrong_Tools'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(), f'Prompting {prompter.model_name} for the Tool Usage task with SG-ICL Prompting'):
            task = row['Task']
            affordance = row['Affordance']
            corr_tool = row['Correct_Tool']
            choices = row['Wrong_Tools'] + [corr_tool]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Here are a few examples:\n{ex_str}Task: {task}\nTools: {choices_string}\nYour Choice:'
            res = prompter.prompt_model(system_msg, user_msg, question)
            tup = ToolSubstitutionResult(task, affordance, corr_tool, res, choices)
            results.append(tup)
            log = SgiclLogEntry(question, res, corr_tool)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name, 'contr', 'tool_usage')
        add_to_model_overview(calculate_average(results, prompter.model_name + '_contr'), 'tool_usage')
        write_log_to_file(logs, prompter.model_name, 'contr', 'tool_usage')

def calculate_average(results: [ToolSubstitutionResult], model: str):
    average = {met: 0 for met in ['acc']}
    for res in results:
        if res.get_pred_correctness():
            average['acc'] += 1
    new_row = pd.Series(
        {'model': model, 'acc': (average['acc'] / len(results))})
    return new_row.to_frame().T
