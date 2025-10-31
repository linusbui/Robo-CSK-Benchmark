import ast
import random

import pandas as pd
from tqdm import tqdm
import os.path

import dspy
from dspy.teleprompt import MIPROv2
from typing import Literal

from tool_usage.prompting.tool_use_result import ToolSubstitutionResult
from utils.prompter import Prompter
from utils.result_writer import add_to_model_overview, write_model_results_to_file
from utils.formatting import transform_prediction, majority_vote
from utils.logging import BasicLogEntry, StepbackLogEntry, write_log_to_file, write_general_log_to_file

system_msg = 'Imagine you are a robot in a household environment being confronted with a task and a list of tools.'
user_msg = 'What is the single tool from the given list that you think is most suitable to help you execute your task? Please only answer with the tool you chose.'

def prompt_all_models(prompters: [Prompter], lower: int, bound):
    for prompter in prompters:
        results = []
        questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions.csv', delimiter=',', on_bad_lines='skip', skiprows=range(1, lower), nrows=bound)
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
        write_model_results_to_file(results, prompter.model_name, '', 'tool_usage', lower, bound)
        add_to_model_overview(calculate_average(results, prompter.model_name), 'tool_usage', lower, bound)


user_msg_rar = 'What is the single tool from the given list that you think is most suitable to help you execute your task? Please only answer with the tool you chose. Reword and elaborate on the inquiry, then provide your final answer.'

def prompt_all_models_rar(prompters: [Prompter], lower: int, bound):
    for prompter in prompters:
        results = []
        logs = []
        questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions.csv', delimiter=',', on_bad_lines='skip', skiprows=range(1, lower), nrows=bound)
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
            pred_tool = transform_prediction(res, choices)
            tup = ToolSubstitutionResult(task, affordance, corr_tool, pred_tool, choices)
            results.append(tup)
            log = BasicLogEntry(question, res, pred_tool, corr_tool)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name, 'rar', 'tool_usage', lower, bound)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_rar'), 'tool_usage', lower, bound)
        write_log_to_file(logs, prompter.model_name, 'rar', 'tool_usage', lower, bound)


user_msg_meta = '''
What is the single tool from the given list that you think is most suitable to help you execute your task? As you perform this task, follow these steps:
1. Clarify your understanding of the question.
2. Make a preliminary identification of the single tool that may be most suitable to help you execute your task.
3. Critically asses your preliminary analysis. If you are unsure about the initial assesment of right tool used for the task, try to reasses it.
4. Confirm your final decision on the most suitable tool to help you execute your task.
5. Evaluate your confidence (0-100%) in your analysis and provide an explanation for this confidence level.
6. Repeat your final choice.
'''

def prompt_all_models_meta(prompters: [Prompter], lower: int, bound):
    for prompter in prompters:
        results = []
        logs = []
        questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions.csv', delimiter=',', on_bad_lines='skip', skiprows=range(1, lower), nrows=bound)
        questions['Wrong_Tools'] = questions['Wrong_Tools'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(), f'Prompting {prompter.model_name} for the Tool Usage task with Metacognitive Prompting'):
            task = row['Task']
            affordance = row['Affordance']
            corr_tool = row['Correct_Tool']
            choices = row['Wrong_Tools'] + [corr_tool]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Task: {task}\nTools: {choices_string}'
            res = prompter.prompt_model(system_msg, user_msg_meta, question)
            pred_tool = transform_prediction(res, choices)
            tup = ToolSubstitutionResult(task, affordance, corr_tool, pred_tool, choices)
            results.append(tup)
            log = BasicLogEntry(question, res, pred_tool, corr_tool)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name, 'meta', 'tool_usage', lower, bound)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_meta'), 'tool_usage', lower, bound)
        write_log_to_file(logs, prompter.model_name, 'meta', 'tool_usage', lower, bound)


user_msg_selfcon = 'What is the single tool from the given list that you think is most suitable to help you execute your task? Think step by step before answering with the single tool of your choosing.'

def prompt_all_models_selfcon(prompters: [Prompter], n_it: int, lower: int, bound):
    for prompter in prompters:
        results = []
        logs = []
        questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions.csv', delimiter=',', on_bad_lines='skip', skiprows=range(1, lower), nrows=bound)
        questions['Wrong_Tools'] = questions['Wrong_Tools'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(), f'Prompting {prompter.model_name} for the Tool Usage task with Self-Consistency Prompting'):
            task = row['Task']
            affordance = row['Affordance']
            corr_tool = row['Correct_Tool']
            choices = row['Wrong_Tools'] + [corr_tool]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Task: {task}\nTools: {choices_string}'

            log = {'question': question}
            answers = []
            for i in range(n_it):
                res = prompter.prompt_model(system_msg, user_msg_selfcon, question)
                pred_tool = transform_prediction(res, choices)
                answers.append(pred_tool)
                log.update({f'cot_{i}': res,
                            f'answer_{i}': pred_tool})
            final_pred = majority_vote(answers)
            tup = ToolSubstitutionResult(task, affordance, corr_tool, final_pred, choices)
            results.append(tup)
            log.update({'final_answer': final_pred,
                        'correct_answer': corr_tool})
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name, 'selfcon', 'tool_usage', lower, bound)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_selfcon'), 'tool_usage', lower, bound)
        write_general_log_to_file(logs, prompter.model_name, 'selfcon', 'tool_usage', lower, bound)


user_msg_initial = 'What is the single tool from the given list that you think is most suitable to help you execute your task? Generate your answer in the following format:\nExplanation: <explanation>\nTool: <tool>'
user_msg_feedback = "Provide Feedback on the answer. The feedback should only evaluate the correctness of the answer. At the end, score the answer from 1 to 5. A score of 5 means that the answer is the right choice."
system_msg_refine = 'You are given an answer to a multiple choice question regarding the use of tools in a household environment and corresponding feedback.'
user_msg_refine = 'Improve upon the answer based on the feedback. Remember that the answer has to be chosen from the given list. Generate your answer in the following format:\nExplanation: <explanation>\nTool: <tool>'

def prompt_all_models_selfref(prompters: [Prompter], n_it: int, lower: int, bound):
    for prompter in prompters:
        results = []
        logs = []
        questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions.csv', delimiter=',', on_bad_lines='skip', skiprows=range(1, lower), nrows=bound)
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
            answer = prompter.prompt_model(system_msg, user_msg_initial, question)

            # full conversation for logging, LLM only sees last answer
            full_conv = question + f'\n{answer}'

            # Refine - Feedback loop
            for i in range(n_it):
                # feedback 
                f_question = question + f'\n{answer}' + '\nFeedback:'
                system_msg_feedback = f'You are given an answer to a multiple choice question regarding the use of tools in a household environment. The possible answers are: {choices_string}'
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
            pred_tool = transform_prediction(answer, choices)
            tup = ToolSubstitutionResult(task, affordance, corr_tool, pred_tool, choices)
            results.append(tup)
            log = BasicLogEntry(question, full_conv, pred_tool, corr_tool)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name, 'selfref', 'tool_usage', lower, bound)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_selfref'), 'tool_usage', lower, bound)
        write_log_to_file(logs, prompter.model_name, 'selfref', 'tool_usage', lower, bound)


system_msg_principle = 'You are given a multiple-choice question. Your task is to extract the underlying concepts and principles involved in choosing the right answer.'
user_msg_principle = 'Only answer with the 5 most important concepts and principles.'

def prompt_all_models_stepback(prompters: [Prompter], lower: int, bound):
    for prompter in prompters:
        results = []
        logs = []
        questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions.csv', delimiter=',', on_bad_lines='skip', skiprows=range(1, lower), nrows=bound)
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
            question = f'Task: {task}\nTools: {choices_string}'

            res = prompter.prompt_model(system_msg, user_msg_stepback, question)

            pred_tool = transform_prediction(res, choices)
            tup = ToolSubstitutionResult(task, affordance, corr_tool, pred_tool, choices)
            results.append(tup)
            log = StepbackLogEntry(p_question, principles, question, res, pred_tool, corr_tool)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name, 'stepback', 'tool_usage', lower, bound)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_stepback'), 'tool_usage', lower, bound)
        write_log_to_file(logs, prompter.model_name, 'stepback', 'tool_usage', lower, bound)


system_msg_example = 'You are helping to create questions regarding household environments.'
user_msg_example = 'For the given tool, generate a task that is executed using that tool. Answer only with the task.'
user_msg_sgicl = 'You are given a task, a list of tools and some examples. What is the single tool from the given list that you think is most suitable to help you execute your task? Please only answer with the tool you chose.'

def prompt_all_models_sgicl(prompters: [Prompter], n_ex: int, lower: int, bound):
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
        examples = pd.read_csv(ex_file, delimiter=',', on_bad_lines='skip', nrows=n_ex)
        ex_str = ''
        for index, row in examples.iterrows():
            task = row['Task']
            tool = row['Tool']
            ex_str = ex_str + f'Task: {task}\nTool: {tool}\n'

        # few shot prompting
        results = []
        logs = []
        questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions.csv', delimiter=',', on_bad_lines='skip', skiprows=range(1, lower), nrows=bound)
        questions['Wrong_Tools'] = questions['Wrong_Tools'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(), f'Prompting {prompter.model_name} for the Tool Usage task with SG-ICL Prompting'):
            task = row['Task']
            affordance = row['Affordance']
            corr_tool = row['Correct_Tool']
            choices = row['Wrong_Tools'] + [corr_tool]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Here are a few examples:\n{ex_str}\nTask: {task}\nTools: {choices_string}\nYour Choice:'
            res = prompter.prompt_model(system_msg, user_msg_sgicl, question)
            pred_tool = transform_prediction(res, choices)
            tup = ToolSubstitutionResult(task, affordance, corr_tool, pred_tool, choices)
            results.append(tup)
            log = BasicLogEntry(question, res, pred_tool, corr_tool)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name, 'sgicl', 'tool_usage', lower, bound)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_sgicl'), 'tool_usage', lower, bound)
        write_log_to_file(logs, prompter.model_name, 'sgicl', 'tool_usage', lower, bound)


system_msg_rewrite = 'You are helping in rewriting answers to questions regarding household environments.'
user_msg_rewrite = 'Rewrite the given answer by swapping key points with wrong facts leading to a wrong final answer. Keep the overall structure the same.'
user_msg_contr = 'You are given a task and a list of tools. Additionaly you are given some right and wrong answers to similar questions. What is the single tool from the given list that you think is most suitable to help you execute your task? Generate your answer in the following format:\nExplanation: <explanation>\nTool: <tool>.'

def prompt_all_models_contr(prompters: [Prompter], n_ex: int, n_cot: int, lower: int, bound):
    for prompter in prompters:
        # Generate examples if needed
        ex_file = f'tool_usage/examples/tool_usage_multichoice_cot_examples_{prompter.model_name}.csv'
        if not os.path.isfile(ex_file):
            results = []
            log = pd.read_csv(f'tool_usage/logs/{prompter.model_name}/{prompter.model_name}_selfcon.csv', delimiter=',', on_bad_lines='skip', nrows=5)
            for index, row in tqdm(log.iterrows(),
                                f'Prompting {prompter.model_name} to generate Tool Usage task examples'):
                corr_tool = row['correct_answer']
                # get correct cot
                cot_right = ''
                for i in range(n_cot):
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
        questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions.csv', delimiter=',', on_bad_lines='skip', skiprows=range(1, lower), nrows=bound)
        questions['Wrong_Tools'] = questions['Wrong_Tools'].apply(ast.literal_eval)
        for index, row in tqdm(questions.iterrows(), f'Prompting {prompter.model_name} for the Tool Usage task with Contrastive CoT Prompting'):
            task = row['Task']
            affordance = row['Affordance']
            corr_tool = row['Correct_Tool']
            choices = row['Wrong_Tools'] + [corr_tool]
            random.shuffle(choices)
            choices_string = ', '.join([c for c in choices])
            question = f'Here are a few examples:\n{ex_str}Task: {task}\nTools: {choices_string}\nYour Choice:'
            res = prompter.prompt_model(system_msg, user_msg_contr, question)
            pred_tool = transform_prediction(res, choices)
            tup = ToolSubstitutionResult(task, affordance, corr_tool, pred_tool, choices)
            results.append(tup)
            log = BasicLogEntry(question, res, pred_tool, corr_tool)
            logs.append(log)
        write_model_results_to_file(results, prompter.model_name, 'contr', 'tool_usage', lower, bound)
        add_to_model_overview(calculate_average(results, prompter.model_name + '_contr'), 'tool_usage', lower, bound)
        write_log_to_file(logs, prompter.model_name, 'contr', 'tool_usage', lower, bound)


def prompt_dspy(lm: dspy.LM, mode: str):
    dspy.configure(lm=lm)
    save_path = 'tool_usage/results/dspy_optimized'
    res_path = 'tool_usage/results/dspy_eval.csv'
    n_train = 50

    if mode == 'scratch':
        instruct = ''
    if mode == 'role' or mode == 'base':
        instruct = 'Imagine you are a robot in a household environment being confronted with a task and a list of tools. What is the single tool from the given list that you think is most suitable to help you execute your task? Please only answer with the tool you chose.'

    # Dataset creation
    trainset = []
    testset = []
    questions = pd.read_csv('tool_usage/tool_usage_multichoice_questions.csv', delimiter=',', on_bad_lines='skip')
    questions['Wrong_Tools'] = questions['Wrong_Tools'].apply(ast.literal_eval)
    for index, row in questions.iterrows():
        task = row['Task']
        corr_tool = row['Correct_Tool']
        choices = row['Wrong_Tools'] + [corr_tool]
        random.shuffle(choices)
        choices_string = ', '.join([c for c in choices])
        question = f'Task: {task}\nTools: {choices_string}'
        testset.append(dspy.Example(question=question, answer=corr_tool).with_inputs('question'))
        if index < n_train:
            trainset.append(dspy.Example(question=question, answer=corr_tool).with_inputs('question'))

    def one_word_answer(args, pred):
        return 1.0 if len(pred.answer.split()) == 1 else 0.0

    class ToolUsage(dspy.Module):
        def __init__(self):
            classifier = dspy.Predict(
                dspy.Signature(
                    'question -> answer: str',
                    instructions = instruct
                )
            )
            self.classifier = dspy.Refine(
                module = classifier,
                N = 3,
                reward_fn = one_word_answer,
                threshold = 1.0
            )

        def forward(self, question):
            pred = self.classifier(question=question)
            return pred
    
    # show current saved program
    if mode == 'show':
        if os.path.isdir(save_path):
            loaded_program = dspy.load(save_path)
            print('\nSummary of dspy-optimized prompts:')
            for name, pred in loaded_program.named_predictors():
                print(f"Predictor: {name}")
                print("Prompt:", pred.signature.instructions)
        else:
            print('No saved model was found!')
        return
    
    # evaluate current saved program
    if mode == 'eval':
        if os.path.isdir(save_path):
            loaded_program = dspy.load(save_path)
            eval = dspy.Evaluate(devset=testset, metric=dspy.evaluate.metrics.answer_exact_match, num_threads = 16, display_progress=True, display_table=True)

            print('\nSummary of dspy-optimized prompts:')
            for name, pred in loaded_program.named_predictors():
                print(f"Predictor: {name}")
                print("Prompt:", pred.signature.instructions)
            print('\nEvaluating cutlery prompts')
            eval(loaded_program)
        else:
            print('No saved model was found!')
        return
    
    # evaluate DSPy program with RoboCSKBench prompt in signature
    if mode == 'base':
        module = ToolUsage()
        eval = dspy.Evaluate(devset=testset, metric=dspy.evaluate.metrics.answer_exact_match, num_threads = 16, display_progress=True, save_as_csv=res_path)
        # Evaluate optimized program
        print('\nEvaluating basic program')
        eval(module)

    # Optimize and evaluate program, either from basic signature or from signature with instructions  
    if mode == 'scratch' or mode == 'role':
        # Optimizing
        teleprompter = MIPROv2(
            metric = dspy.evaluate.metrics.answer_exact_match,
            auto = 'medium'
        )

        print('Optimizing with miprov2')

        optimized_program = teleprompter.compile(
            ToolUsage(),
            trainset = trainset,
            max_bootstrapped_demos = 0,
            max_labeled_demos = 0
        )

        optimized_program.save(save_path, save_program = True)

        # Give model summary
        print('\nSummary of dspy-optimized prompts for cutlery:')
        for name, pred in optimized_program.named_predictors():
            print(f"Predictor: {name}")
            print("Prompt:", pred.signature.instructions)

        # Evaluate optimized program
        print("\nEvaluating optimized program")
        eval = dspy.Evaluate(devset=testset, metric=dspy.evaluate.metrics.answer_exact_match, num_threads = 16, display_progress=True, save_as_csv=res_path, display_table=True)
        eval(optimized_program)


def calculate_average(results: [ToolSubstitutionResult], model: str):
    average = {met: 0 for met in ['acc']}
    for res in results:
        if res.get_pred_correctness():
            average['acc'] += 1
    new_row = pd.Series(
        {'model': model, 'acc': (average['acc'] / len(results))})
    return new_row.to_frame().T
