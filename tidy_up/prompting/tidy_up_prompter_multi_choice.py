import ast
import random

import pandas as pd
from tqdm import tqdm
import os.path

from tidy_up.prompting.tidy_up_result import TidyUpMultiChoiceResult
from utils.prompter import Prompter
from utils.result_writer import add_to_model_overview, write_model_results_to_file
from utils.formatting import transform_prediction, majority_vote
from utils.logging import BasicLogEntry, StepbackLogEntry, write_log_to_file, write_general_log_to_file

system_msg = 'Imagine you are a robot tidying up a household environment, being confronted with an object and a possible list of locations to put it.'
user_msg = 'What is the single location from the given list that you think is the most suitable place to put the object? Please only answer with the location you chose.'

def prompt_all_models(prompters: [Prompter], num_runs:int):
    for prompter in prompters:
        results = []
        questions = pd.read_csv('tidy_up/tidy_up_multichoice.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
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
            tup = TidyUpMultiChoiceResult(obj, corr_loc, res, choices)
            results.append(tup)
        write_model_results_to_file(results, prompter.model_name, '', 'tidy_up/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name), 'tidy_up/results_multi', False)


user_msg_rar = 'What is the single location from the given list that you think is the most suitable place to put the object? Reword and elaborate on the inquiry, then answer only with the location you chose.'

def prompt_all_models_rar(prompters: [Prompter], num_runs: int):
    for prompter in prompters:
        results = []
        logs = []
        questions = pd.read_csv('tidy_up/tidy_up_multichoice.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        questions['Wrong_Locations'] = questions['Wrong_Locations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Tidy Up task with RaR Prompting'):
            obj = row['Object']
            corr_loc = row['Correct_Location']
            choices = row['Wrong_Locations'] + [corr_loc]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Object: {obj}\nLocations: {choices_string}\nYour Choice:'
            res = prompter.prompt_model(system_msg, user_msg_rar, question)
            pred_loc = transform_prediction(res, choices)
            tup = TidyUpMultiChoiceResult(obj, corr_loc, pred_loc, choices)
            results.append(tup)
            log = BasicLogEntry(question, res, pred_loc, corr_loc)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name, 'rar', 'tidy_up/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_rar'), 'tidy_up/results_multi', False)
        write_log_to_file(logs, prompter.model_name, 'rar', 'tidy_up')


user_msg_meta = '''
What is the single location from the given list that you think is the most suitable place to put the object? As you perform this task, follow these steps:
1. Clarify your understanding of the question.
2. Make a preliminary identification of the single most suitable location for the object.
3. Critically asses your preliminary analysis. If you are unsure about the initial assesment of the most suitable location of the object, try to reasses it.
4. Confirm your final decision on the most suitable place to put the object.
5. Evaluate your confidence (0-100%) in your analysis and provide an explanation for this confidence level.
6. Repeat your final choice.
'''

def prompt_all_models_meta(prompters: [Prompter], num_runs: int):
    for prompter in prompters:
        results = []
        logs = []
        questions = pd.read_csv('tidy_up/tidy_up_multichoice.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        questions['Wrong_Locations'] = questions['Wrong_Locations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Tidy Up task with Metacognitive Prompting'):
            obj = row['Object']
            corr_loc = row['Correct_Location']
            choices = row['Wrong_Locations'] + [corr_loc]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Object: {obj}\nLocations: {choices_string}'
            res = prompter.prompt_model(system_msg, user_msg_meta, question)
            pred_loc = transform_prediction(res, choices)
            tup = TidyUpMultiChoiceResult(obj, corr_loc, pred_loc, choices)
            results.append(tup)
            log = BasicLogEntry(question, res, pred_loc, corr_loc)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name, 'meta', 'tidy_up/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_meta'), 'tidy_up/results_multi', False)
        write_log_to_file(logs, prompter.model_name, 'meta', 'tidy_up')


user_msg_selfcon = 'What is the single location from the given list that you think is the most suitable place to put the object? Think step by step before answering with the single location of your choosing.'

def prompt_all_models_selfcon(prompters: [Prompter], num_runs: int, n_it: int):
    for prompter in prompters:
        results = []
        logs = []
        questions = pd.read_csv('tidy_up/tidy_up_multichoice.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        questions['Wrong_Locations'] = questions['Wrong_Locations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Tidy Up task with Self-Consistency Prompting'):
            obj = row['Object']
            corr_loc = row['Correct_Location']
            choices = row['Wrong_Locations'] + [corr_loc]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Object: {obj}\nLocations: {choices_string}'

            log = {'question': question}
            answers = []
            for i in range(n_it):
                res = prompter.prompt_model(system_msg, user_msg_selfcon, question)
                pred_loc = transform_prediction(res, choices)
                answers.append(pred_loc)
                log.update({f'cot_{i}': res,
                            f'answer_{i}': pred_loc})
            final_pred = majority_vote(answers)
            tup = TidyUpMultiChoiceResult(obj, corr_loc, final_pred, choices)
            results.append(tup)
            log.update({'final_answer': final_pred,
                        'correct_answer': corr_loc})
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name, 'selfcon', 'tidy_up/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_selfcon'), 'tidy_up/results_multi', False)
        write_general_log_to_file(logs, prompter.model_name, 'selfcon', 'tidy_up')


user_msg_initial = 'What is the single location from the given list that you think is the most suitable place to put the object? Generate your answer in the following format:\nExplanation: <explanation>\nLocation: <location>'
user_msg_feedback = "Provide Feedback on the answer. The feedback should only evaluate the correctness of the answer. Remember that the answer has to be chosen from the given list. At the end, score the answer from 1 to 5. A score of 5 means that the answer is the right choice."
system_msg_refine = 'You are given an answer to a multiple choice question regarding tidying up a household and corresponding feedback.'
user_msg_refine = 'Improve upon the answer based on the feedback. Remember that the answer has to be chosen from the given list. Generate your answer in the following format:\nExplanation: <explanation>\nLocation: <location>'

def prompt_all_models_selfref(prompters: [Prompter], num_runs: int, n_it: int):
    for prompter in prompters:
        results = []
        logs = []
        questions = pd.read_csv('tidy_up/tidy_up_multichoice.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        questions['Wrong_Locations'] = questions['Wrong_Locations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Tidy Up task with Self-Refine Prompting'):
            obj = row['Object']
            corr_loc = row['Correct_Location']
            choices = row['Wrong_Locations'] + [corr_loc]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])

            # initial prompt
            question = f'Object: {obj}\nLocations: {choices_string}'
            answer = prompter.prompt_model(system_msg, user_msg_initial, question)

            # full conversation for logging, LLM only sees last answer
            full_conv = question + f'\n{answer}'

            # Refine - Feedback loop
            for i in range(n_it):
                # feedback 
                f_question = question + f'\n{answer}' + '\nFeedback:'
                system_msg_feedback = f'You are given the answer to a multiple choice question regarding tidying up a household. The possible answers are: {choices_string}'
                feedback = prompter.prompt_model(system_msg_feedback, user_msg_feedback, f_question)
                full_conv = full_conv + '\nFeedback:' + f'\n{feedback}'

                # Stopping condition
                ind = feedback.lower().find('score')
                if not ind == -1:
                    end = feedback[ind:]
                    if '4' in end or '5' in end:
                        break

                # refine
                r_question = f_question + f'\n{feedback}'
                answer = prompter.prompt_model(system_msg_refine, user_msg_refine, r_question)
                full_conv = full_conv + '\nImprovement:' + f'\n{answer}'

            # final answer
            pred_loc = transform_prediction(answer, choices)
            tup = TidyUpMultiChoiceResult(obj, corr_loc, pred_loc, choices)
            results.append(tup)
            log = BasicLogEntry(question, full_conv, pred_loc, corr_loc)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name, 'selfref', 'tidy_up/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_selfref'), 'tidy_up/results_multi', False)
        write_log_to_file(logs, prompter.model_name, 'selfref', 'tidy_up')


system_msg_principle = 'You are given a multiple-choice question. Your task is to extract the underlying concepts and principles involved in choosing the right answer.'
user_msg_principle = 'Only answer with the 5 most important concepts and principles.'

def prompt_all_models_stepback(prompters: [Prompter], num_runs: int):
    for prompter in prompters:
        results = []
        logs = []
        questions = pd.read_csv('tidy_up/tidy_up_multichoice.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        questions['Wrong_Locations'] = questions['Wrong_Locations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Tidy Up task with Stepback Prompting'):
            obj = row['Object']
            corr_loc = row['Correct_Location']
            choices = row['Wrong_Locations'] + [corr_loc]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])

            # Get higher level principles
            p_question = f'Object: {obj}\nLocations: {choices_string}\nPrinciples:'
            principles = prompter.prompt_model(system_msg_principle, user_msg_principle, p_question)

            # Get answer based on principles
            user_msg_stepback = f'What is the single location from the given list that you think is the most suitable place to put the object? Answer the question step by step using the following principles:\n{principles}\nEnd your answer with the single location you chose.'
            question = f'Object: {obj}\nLocations: {choices_string}'

            res = prompter.prompt_model(system_msg, user_msg_stepback, question)

            pred_loc = transform_prediction(res, choices)
            tup = TidyUpMultiChoiceResult(obj, corr_loc, pred_loc, choices)
            results.append(tup)
            log = StepbackLogEntry(p_question, principles, question, res, pred_loc, corr_loc)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name, 'stepback', 'tidy_up/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_stepback'), 'tidy_up/results_multi', False)
        write_log_to_file(logs, prompter.model_name, 'stepback', 'tidy_up')


system_msg_example = 'You are helping to create questions regarding household environments.'
user_msg_example = 'For the given location, generate an object typically found in that location. Answer only with the object.'

def prompt_all_models_sgicl(prompters: [Prompter], num_runs: int, n_ex: int):
    for prompter in prompters:
        # Generate examples if needed
        ex_file = f'tidy_up/examples/tidy_up_multichoice_examples_{prompter.model_name}.csv'
        if not os.path.isfile(ex_file):
            results = []
            questions = pd.read_csv('tidy_up/tidy_up_multichoice.csv', delimiter=',', on_bad_lines='skip', nrows=15)
            for index, row in tqdm(questions.iterrows(),
                                f'Prompting {prompter.model_name} to generate Tidy Up task examples'):
                obj = row['Object']
                corr_loc = row['Correct_Location']
                question = f'Location: {corr_loc}\nGenerate an object: {obj}\nGenerate an object:'
                pred_obj = prompter.prompt_model(system_msg_example, user_msg_example, question)
                entry = {
                    'Object': pred_obj,
                    'Location': corr_loc
                }
                results.append(entry)
            df = pd.DataFrame(results)
            df.to_csv(ex_file, index=False)
            print('Finished generating examples')

        # Load examples
        examples = pd.read_csv(ex_file, delimiter=',', on_bad_lines='skip', nrows=n_ex)
        ex_str = ''
        for index, row in examples.iterrows():
            obj = row['Object']
            loc = row['Location']
            ex_str = ex_str + f'Object: {obj}\nLocation: {loc}\n'

        # few shot prompting
        results = []
        logs = []
        questions = pd.read_csv('tidy_up/tidy_up_multichoice.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        questions['Wrong_Locations'] = questions['Wrong_Locations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Tidy Up task with SG-ICL Prompting'):
            obj = row['Object']
            corr_loc = row['Correct_Location']
            choices = row['Wrong_Locations'] + [corr_loc]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])

            question = f'Here are a few examples:\n{ex_str}Object: {obj}\nLocations: {choices_string}\nYour Choice:'
            res = prompter.prompt_model(system_msg, user_msg, question)
            pred_loc = transform_prediction(res, choices)
            tup = TidyUpMultiChoiceResult(obj, corr_loc, pred_loc, choices)
            results.append(tup)
            log = BasicLogEntry(question, res, pred_loc, corr_loc)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name, 'sgicl', 'tidy_up/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_sgicl'), 'tidy_up/results_multi', False)
        write_log_to_file(logs, prompter.model_name, 'sgicl', 'tidy_up')


system_msg_rewrite = 'You are helping in rewriting answers to questions regarding household environments.'
user_msg_rewrite = 'Rewrite the given answer by swapping key points with wrong facts leading to a wrong final answer. Keep the overall structure the same.'

def prompt_all_models_contr(prompters: [Prompter], num_runs: int, n_ex: int, n_cot: int):
    for prompter in prompters:
        # Generate examples if needed
        ex_file = f'tidy_up/examples/tidy_up_multichoice_cot_examples_{prompter.model_name}.csv'
        if not os.path.isfile(ex_file):
            results = []
            log = pd.read_csv(f'tidy_up/logs/{prompter.model_name}/{prompter.model_name}_selfcon.csv', delimiter=',', on_bad_lines='skip', nrows=10)
            for index, row in tqdm(log.iterrows(),
                                f'Prompting {prompter.model_name} to generate Tidy Up task examples'):
                corr_loc = row['correct_answer']
                # get correct cot
                cot_right = ''
                for i in range(n_cot):
                    answ = row[f'answer_{i}']
                    if answ == corr_loc:
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
        examples = pd.read_csv(ex_file, delimiter=',', on_bad_lines='skip', nrows=n_ex)
        ex_str = ''
        for index, row in examples.iterrows():
            question = row['question']
            cot_right = row['cot_right']
            cot_wrong = row['cot_wrong']
            ex_str = ex_str + f'Question: {question}\n\nRight Explanation: {cot_right}\n\nWrong Explanation: {cot_wrong}\n\n'

        # few shot prompting
        results = []
        logs = []
        questions = pd.read_csv('tidy_up/tidy_up_multichoice.csv', delimiter=',', on_bad_lines='skip', nrows=num_runs)
        questions['Wrong_Locations'] = questions['Wrong_Locations'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(),
                               f'Prompting {prompter.model_name} for the multiple choice Tidy Up task with SG-ICL Prompting'):
            obj = row['Object']
            corr_loc = row['Correct_Location']
            choices = row['Wrong_Locations'] + [corr_loc]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])

            question = f'Here are a few examples:\n{ex_str}Object: {obj}\nLocations: {choices_string}\nYour Choice:'
            res = prompter.prompt_model(system_msg, user_msg, question)
            pred_loc = transform_prediction(res, choices)
            tup = TidyUpMultiChoiceResult(obj, corr_loc, pred_loc, choices)
            results.append(tup)
            log = BasicLogEntry(question, res, pred_loc, corr_loc)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name + 'contr', 'tidy_up/results_multi', False)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_contr'), 'tidy_up/results_multi', False)
        write_log_to_file(logs, prompter.model_name, 'contr', 'tidy_up')


def calculate_average(results: [TidyUpMultiChoiceResult], model: str):
    average = {met: 0 for met in ['acc']}
    for res in results:
        if res.get_pred_correctness():
            average['acc'] += 1
    new_row = pd.Series(
        {'model': model, 'acc': (average['acc'] / len(results))})
    return new_row.to_frame().T
