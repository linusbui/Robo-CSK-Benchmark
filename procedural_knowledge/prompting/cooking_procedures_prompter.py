import pandas as pd
import random
import os.path
from tqdm import tqdm

from procedural_knowledge.json_utils import extract_json, extract_json_multi, extract_json_multi_limited, save_to_json
from procedural_knowledge.prompting import evaluate_prompters
from tidy_up.prompting.tidy_up_prompter_open import user_msg
from utils.prompter import Prompter
from utils.formatting import transform_prediction, majority_vote
from utils.logging import BasicLogEntry, StepbackLogEntry, write_log_to_file, write_general_log_to_file

def prompt_all_models_binary(prompters: [Prompter]):
    system_msg = "Imagine you are a robot tasked with determining the temporal order of two steps from one recipe. "
    system_msg_before = "Based on the recipe title and the two steps provided, identify whether one action occurred before another. "
    system_msg_after = "Based on the recipe title and the two steps provided, identify whether one action occurred after another. "
    user_msg = "Answer only with 'Yes' or 'No'."

    for prompter in prompters:
        for recipe_number in range(1, 5):
            json_file = f'procedural_knowledge/data_generation/question_components_binary/questions_recipe_' + str(recipe_number) + '.json'
            recipe_components = extract_json(json_file)

            all_yes_before_answers, all_no_before_answers = [], []
            all_yes_after_answers, all_no_after_answers = [], []
            for recipe in recipe_components:
                title = recipe['goal']
                step_1 = recipe['step_1'].rstrip(". ").strip()
                step_2 = recipe['step_2'].rstrip(". ").strip()

                for steps, correct_response, answers in [
                    ((step_1, step_2), "Yes", all_yes_before_answers),
                    ((step_2, step_1), "No", all_no_before_answers),
                ]:
                    question = f"In the recipe '{title}', did '{steps[0]}' occur before '{steps[1]}'?"
                    response = prompter.prompt_model(system_msg + system_msg_before, user_msg, question)
                    answers.append({
                        'title': title,
                        'question': system_msg + system_msg_before + user_msg + question,
                        'response': response,
                        'correct_response': correct_response
                    })
                    # Save the result to json after every step if an error happens
                    save_to_json(f'procedural_knowledge/results_binary/Yes/{model.model_name}/before_{recipe_number}.json', all_yes_before_answers)
                    save_to_json(f'procedural_knowledge/results_binary/No/{model.model_name}/before_{recipe_number}.json', all_no_before_answers)

                for steps, correct_response, answers in [
                    ((step_1, step_2), "Yes", all_yes_after_answers),
                    ((step_2, step_1), "No", all_no_after_answers),
                ]:
                    question = f"In the recipe '{title}', did '{steps[1]}' occur after '{steps[0]}'?"
                    response = prompter.prompt_model(system_msg + system_msg_after, user_msg, question)
                    answers.append({
                        'title': title,
                        'question': system_msg + system_msg_after + user_msg + question,
                        'response': response,
                        'correct_response': correct_response
                    })
                save_to_json(f'procedural_knowledge/results_binary/Yes/{model.model_name}/after_{recipe_number}.json', all_yes_after_answers)
                save_to_json(f'procedural_knowledge/results_binary/No/{model.model_name}/after_{recipe_number}.json', all_no_after_answers)
    evaluate_prompters.evaluate_binary(prompters)


def prompt_all_models_multi(prompters: [Prompter], num_runs: int):
    def get_answer_before(prompter, recipe_title, step_question, other_steps):
        system_msg = (
            "Imagine you are a robot tasked with determining the temporal order of steps in a recipe. "
            "Based on the recipe title and the provided steps, identify which step occurred before another. "
        )
        user_msg = "Answer only with your chosen step."

        question = (
                f"In the recipe '{recipe_title}', which step occurs before '{step_question}'?\n"
                f"\nOptions:\n" + "\n".join(f"- {step}" for step in other_steps))
        return prompter.prompt_model(system_msg, user_msg, question), question

    def get_answer_after(prompter, recipe_title, step_question, other_steps):
        system_msg = (
            "Imagine you are a robot tasked with determining the temporal order of steps in a recipe. "
            "Based on the recipe title and the provided steps, identify  which step occurred after another. "
        )
        user_msg = "Answer only with your chosen step."
        question = (
                f"In the recipe '{recipe_title}', which steps occurs after '{step_question}'?"
                f"\nOptions:\n" + "\n".join(f"- {step}" for step in other_steps))
        return prompter.prompt_model(system_msg, user_msg, question), question

    for prompter in prompters:
        for recipe_number in range(1, 5):
            json_file = f'procedural_knowledge/data_generation/question_components_multi/questions_recipe_' + str(
                recipe_number) + '.json'
            recipe_components = extract_json_multi_limited(json_file, num_runs)
            before_answers, after_answers = [], []

            for recipe in recipe_components:
                title = recipe['goal']
                step_1 = recipe['step_1'].rstrip(". ").strip()
                step_2 = recipe['step_2'].rstrip(". ").strip()
                step_3 = recipe['step_3'].rstrip(". ").strip()

                before_steps = [step_1, step_3]
                available_indices = [i for i in range(len(recipe_components)) if i != recipe_number]
                unique_step_3s = set(before_steps)  # track what's already added
                random.shuffle(available_indices)  # randomize the order
                for idx in available_indices:
                    if len(unique_step_3s) >= 5:  # step_1 + step_3 + 3 unique additions
                        break
                    other_recipe = recipe_components[idx]
                    candidate_step = other_recipe['step_3'].rstrip(". ").strip()
                    if candidate_step not in unique_step_3s:
                        before_steps.append(candidate_step)
                        unique_step_3s.add(candidate_step)
                random.shuffle(before_steps)
                response, question = get_answer_before(prompter, title, step_2, before_steps)
                before_answers.append({
                    'title': title,
                    'question': question,
                    'response': response,
                    'correct_response': step_1
                })
                save_to_json(f'procedural_knowledge/results_multi/before/{prompter.model_name}/{recipe_number}.json', before_answers)

                after_steps = [step_1, step_3]
                available_indices = [i for i in range(len(recipe_components)) if i != recipe_number]
                unique_step_1s = set(after_steps)  # track what's already added
                random.shuffle(available_indices)  # randomize the order
                for idx in available_indices:
                    if len(unique_step_1s) >= 5:  # step_1 + step_3 + 3 unique additions
                        break
                    other_recipe = recipe_components[idx]
                    candidate_step = other_recipe['step_1'].rstrip(". ").strip()
                    if candidate_step not in unique_step_1s:
                        after_steps.append(candidate_step)
                        unique_step_1s.add(candidate_step)

                random.shuffle(after_steps)
                response, question = get_answer_after(prompter, title, step_2, after_steps)
                after_answers.append({
                    'title': title,
                    'question': question,
                    'response': response,
                    'correct_response': step_3
                })
                save_to_json(f'procedural_knowledge/results_multi/after/{prompter.model_name}/{recipe_number}.json', after_answers)
    evaluate_prompters.evaluate_multi(prompters, '')


def prompt_all_models_multi_rar(prompters: [Prompter], num_runs: int):
    def get_answer_before(prompter, recipe_title, step_question, other_steps):
        system_msg = (
            "Imagine you are a robot tasked with determining the temporal order of steps in a recipe. "
            "Based on the recipe title and the provided steps, identify which step occurred before another. "
        )
        user_msg = "Reword and elaborate on the inquiry, then answer only with your chosen step."

        question = (
                f"In the recipe '{recipe_title}', which step occurs before '{step_question}'?\n"
                f"\nOptions:\n" + "\n".join(f"- {step}" for step in other_steps))
        answer = prompter.prompt_model(system_msg, user_msg, question)
        return transform_prediction(answer, [step for step in other_steps]), question, answer

    def get_answer_after(prompter, recipe_title, step_question, other_steps):
        system_msg = (
            "Imagine you are a robot tasked with determining the temporal order of steps in a recipe. "
            "Based on the recipe title and the provided steps, identify which step occurred after another. "
        )
        user_msg = "Reword and elaborate on the inquiry, then answer only with your chosen step."
        question = (
                f"In the recipe '{recipe_title}', which steps occurs after '{step_question}'?"
                f"\nOptions:\n" + "\n".join(f"- {step}" for step in other_steps))
        answer = prompter.prompt_model(system_msg, user_msg, question)
        return transform_prediction(answer, [step for step in other_steps]), question, answer

    for prompter in prompters:
        logs_before = []
        logs_after = []
        for recipe_number in tqdm(range(1, 5), f'Prompting {prompter.model_name} for the multiple choice Procedural Knowledge task with RaR Prompting'):
            json_file = f'procedural_knowledge/data_generation/question_components_multi/questions_recipe_' + str(
                recipe_number) + '.json'
            recipe_components = extract_json_multi_limited(json_file, num_runs)
            before_answers, after_answers = [], []

            for recipe in tqdm(recipe_components, f'Prompting recipes {recipe_number}'):
                title = recipe['goal']
                step_1 = recipe['step_1'].rstrip(". ").strip()
                step_2 = recipe['step_2'].rstrip(". ").strip()
                step_3 = recipe['step_3'].rstrip(". ").strip()

                before_steps = [step_1, step_3]
                available_indices = [i for i in range(len(recipe_components)) if i != recipe_number]
                unique_step_3s = set(before_steps)  # track what's already added
                random.shuffle(available_indices)  # randomize the order
                for idx in available_indices:
                    if len(unique_step_3s) >= 5:  # step_1 + step_3 + 3 unique additions
                        break
                    other_recipe = recipe_components[idx]
                    candidate_step = other_recipe['step_3'].rstrip(". ").strip()
                    if candidate_step not in unique_step_3s:
                        before_steps.append(candidate_step)
                        unique_step_3s.add(candidate_step)
                random.shuffle(before_steps)
                response, question, full_response = get_answer_before(prompter, title, step_2, before_steps)
                before_answers.append({
                    'title': title,
                    'question': question,
                    'response': response,
                    'correct_response': step_1
                })
                save_to_json(f'procedural_knowledge/results_multi/before/{prompter.model_name}_rar/{recipe_number}.json', before_answers)
                log_before = BasicLogEntry(question, full_response, response, step_1)
                logs_before.append(log_before)

                after_steps = [step_1, step_3]
                available_indices = [i for i in range(len(recipe_components)) if i != recipe_number]
                unique_step_1s = set(after_steps)  # track what's already added
                random.shuffle(available_indices)  # randomize the order
                for idx in available_indices:
                    if len(unique_step_1s) >= 5:  # step_1 + step_3 + 3 unique additions
                        break
                    other_recipe = recipe_components[idx]
                    candidate_step = other_recipe['step_1'].rstrip(". ").strip()
                    if candidate_step not in unique_step_1s:
                        after_steps.append(candidate_step)
                        unique_step_1s.add(candidate_step)

                random.shuffle(after_steps)
                response, question, full_response = get_answer_after(prompter, title, step_2, after_steps)
                after_answers.append({
                    'title': title,
                    'question': question,
                    'response': response,
                    'correct_response': step_3
                })
                save_to_json(f'procedural_knowledge/results_multi/after/{prompter.model_name}_rar/{recipe_number}.json', after_answers)
                log_after = BasicLogEntry(question, full_response, response, step_3)
                logs_after.append(log_after)
    evaluate_prompters.evaluate_multi(prompters, '_rar')
    write_log_to_file(logs_before, prompter.model_name, 'before_rar', 'procedural_knowledge')
    write_log_to_file(logs_after, prompter.model_name, 'after_rar', 'procedural_knowledge')


def prompt_all_models_multi_meta(prompters: [Prompter], num_runs: int):
    def get_answer_before(prompter, recipe_title, step_question, other_steps):
        system_msg = (
            "Imagine you are a robot tasked with determining the temporal order of steps in a recipe. "
            "Based on the recipe title and the provided steps, identify which step occurred before another. "
        )
        user_msg = '''As you perform this task, follow these steps:
        1. Clarify your understanding of the question.
        2. Make a preliminary identification of the step that occured before.
        3. Critically asses your preliminary analysis. If you are unsure about the initial assesment of the sequence of the steps, try to reasses it.
        4. Confirm your final decision on the step that occured before.
        5. Evaluate your confidence (0-100%) in your analysis and provide an explanation for this confidence level. 
        6. Repeat your final choice. Only include your chosen step.
        '''

        question = (
                f"In the recipe '{recipe_title}', which step occurs before '{step_question}'?\n"
                f"\nOptions:\n" + "\n".join(f"- {step}" for step in other_steps))
        answer = prompter.prompt_model(system_msg, user_msg, question)
        return  transform_prediction(answer, [step for step in other_steps]), question, answer

    def get_answer_after(prompter, recipe_title, step_question, other_steps):
        system_msg = (
            "Imagine you are a robot tasked with determining the temporal order of steps in a recipe. "
            "Based on the recipe title and the provided steps, identify which step occurred after another. "
        )
        user_msg = '''As you perform this task, follow these steps:
        1. Clarify your understanding of the question.
        2. Make a preliminary identification of the step that occured before.
        3. Critically asses your preliminary analysis. If you are unsure about the initial assesment of the sequence of the steps, try to reasses it.
        4. Confirm your final decision on the step that occured before.
        5. Evaluate your confidence (0-100%) in your analysis and provide an explanation for this confidence level. 
        6. Repeat your final choice. Only include your chosen step.
        '''
        question = (
                f"In the recipe '{recipe_title}', which steps occurs after '{step_question}'?"
                f"\nOptions:\n" + "\n".join(f"- {step}" for step in other_steps))
        answer = prompter.prompt_model(system_msg, user_msg, question)
        return  transform_prediction(answer, [step for step in other_steps]), question, answer

    for prompter in prompters:
        logs_before = []
        logs_after = []
        for recipe_number in tqdm(range(1, 5), f'Prompting {prompter.model_name} for the multiple choice Procedural Knowledge task with Metacognitive Prompting'):
            json_file = f'procedural_knowledge/data_generation/question_components_multi/questions_recipe_' + str(
                recipe_number) + '.json'
            recipe_components = extract_json_multi_limited(json_file, num_runs)
            before_answers, after_answers = [], []

            for recipe in tqdm(recipe_components, f'Prompting recipes {recipe_number}'):
                title = recipe['goal']
                step_1 = recipe['step_1'].rstrip(". ").strip()
                step_2 = recipe['step_2'].rstrip(". ").strip()
                step_3 = recipe['step_3'].rstrip(". ").strip()

                before_steps = [step_1, step_3]
                available_indices = [i for i in range(len(recipe_components)) if i != recipe_number]
                unique_step_3s = set(before_steps)  # track what's already added
                random.shuffle(available_indices)  # randomize the order
                for idx in available_indices:
                    if len(unique_step_3s) >= 5:  # step_1 + step_3 + 3 unique additions
                        break
                    other_recipe = recipe_components[idx]
                    candidate_step = other_recipe['step_3'].rstrip(". ").strip()
                    if candidate_step not in unique_step_3s:
                        before_steps.append(candidate_step)
                        unique_step_3s.add(candidate_step)
                random.shuffle(before_steps)
                response, question, full_response = get_answer_before(prompter, title, step_2, before_steps)
                before_answers.append({
                    'title': title,
                    'question': question,
                    'response': response,
                    'correct_response': step_1
                })
                save_to_json(f'procedural_knowledge/results_multi/before/{prompter.model_name}_meta/{recipe_number}.json', before_answers)
                log_before = BasicLogEntry(question, full_response, response, step_1)
                logs_before.append(log_before)

                after_steps = [step_1, step_3]
                available_indices = [i for i in range(len(recipe_components)) if i != recipe_number]
                unique_step_1s = set(after_steps)  # track what's already added
                random.shuffle(available_indices)  # randomize the order
                for idx in available_indices:
                    if len(unique_step_1s) >= 5:  # step_1 + step_3 + 3 unique additions
                        break
                    other_recipe = recipe_components[idx]
                    candidate_step = other_recipe['step_1'].rstrip(". ").strip()
                    if candidate_step not in unique_step_1s:
                        after_steps.append(candidate_step)
                        unique_step_1s.add(candidate_step)

                random.shuffle(after_steps)
                response, question, full_response = get_answer_after(prompter, title, step_2, after_steps)
                after_answers.append({
                    'title': title,
                    'question': question,
                    'response': response,
                    'correct_response': step_3
                })
                save_to_json(f'procedural_knowledge/results_multi/after/{prompter.model_name}_meta/{recipe_number}.json', after_answers)
                log_after = BasicLogEntry(question, full_response, response, step_3)
                logs_after.append(log_after)
    evaluate_prompters.evaluate_multi(prompters, '_meta')
    write_log_to_file(logs_before, prompter.model_name, 'before_meta', 'procedural_knowledge')
    write_log_to_file(logs_after, prompter.model_name, 'after_meta', 'procedural_knowledge')


def prompt_all_models_multi_selfcon(prompters: [Prompter], num_runs: int, n_it: int):
    def get_answer_before(prompter, recipe_title, step_question, other_steps):
        system_msg = "Imagine you are a robot tasked with determining the temporal order of steps in a recipe. "
        user_msg = 'First think step by step to make sure you have the right answer. Then provide the final answer as only your chosen step.'

        question = (
                f"In the recipe '{recipe_title}', which step occurs before '{step_question}'?\n"
                f"\nOptions:\n" + "\n".join(f"- {step}" for step in other_steps))
        answer = prompter.prompt_model(system_msg, user_msg, question)
        return transform_prediction(answer, [step for step in other_steps]), question, answer

    def get_answer_after(prompter, recipe_title, step_question, other_steps):
        system_msg = "Imagine you are a robot tasked with determining the temporal order of steps in a recipe. "
        user_msg = 'First think step by step to make sure you have the right answer. Then provide the final answer as only your chosen step.'

        question = (
                f"In the recipe '{recipe_title}', which steps occurs after '{step_question}'?"
                f"\nOptions:\n" + "\n".join(f"- {step}" for step in other_steps))
        answer = prompter.prompt_model(system_msg, user_msg, question)
        return transform_prediction(answer, [step for step in other_steps]), question, answer

    for prompter in prompters:
        logs_before = []
        logs_after = []
        for recipe_number in tqdm(range(1, 5), f'Prompting {prompter.model_name} for the multiple choice Procedural Knowledge task with Self-Consistency Prompting'):
            json_file = f'procedural_knowledge/data_generation/question_components_multi/questions_recipe_' + str(
                recipe_number) + '.json'
            recipe_components = extract_json_multi_limited(json_file, num_runs)
            before_answers, after_answers = [], []

            for recipe in tqdm(recipe_components, f'Prompting recipes {recipe_number}'):
                title = recipe['goal']
                step_1 = recipe['step_1'].rstrip(". ").strip()
                step_2 = recipe['step_2'].rstrip(". ").strip()
                step_3 = recipe['step_3'].rstrip(". ").strip()

                before_steps = [step_1, step_3]
                available_indices = [i for i in range(len(recipe_components)) if i != recipe_number]
                unique_step_3s = set(before_steps)  # track what's already added
                random.shuffle(available_indices)  # randomize the order
                for idx in available_indices:
                    if len(unique_step_3s) >= 5:  # step_1 + step_3 + 3 unique additions
                        break
                    other_recipe = recipe_components[idx]
                    candidate_step = other_recipe['step_3'].rstrip(". ").strip()
                    if candidate_step not in unique_step_3s:
                        before_steps.append(candidate_step)
                        unique_step_3s.add(candidate_step)
                random.shuffle(before_steps)

                answers = []
                log_before = {}
                for i in range(n_it):
                    response, question, full_response = get_answer_before(prompter, title, step_2, before_steps)
                    answers.append(response)
                    log_before.update({'question': question,
                                       f'cot_{i}': full_response,
                                       f'answer_{i}': response})
                final_answer = majority_vote(answers)
                before_answers.append({
                    'title': title,
                    'question': question,
                    'response': final_answer,
                    'correct_response': step_1
                })
                save_to_json(f'procedural_knowledge/results_multi/before/{prompter.model_name}_selfcon/{recipe_number}.json', before_answers)
                log_before.update({'final_answer': final_answer,
                                   'correct_answer': step_1})
                logs_before.append(log_before)

                after_steps = [step_1, step_3]
                available_indices = [i for i in range(len(recipe_components)) if i != recipe_number]
                unique_step_1s = set(after_steps)  # track what's already added
                random.shuffle(available_indices)  # randomize the order
                for idx in available_indices:
                    if len(unique_step_1s) >= 5:  # step_1 + step_3 + 3 unique additions
                        break
                    other_recipe = recipe_components[idx]
                    candidate_step = other_recipe['step_1'].rstrip(". ").strip()
                    if candidate_step not in unique_step_1s:
                        after_steps.append(candidate_step)
                        unique_step_1s.add(candidate_step)

                random.shuffle(after_steps)
                
                answers = []
                log_after = {}
                for i in range(n_it):
                    response, question, full_response = get_answer_after(prompter, title, step_2, before_steps)
                    answers.append(response)
                    log_after.update({'question': question,
                                       f'cot_{i}': full_response,
                                       f'answer_{i}': response})
                final_answer = majority_vote(answers)
                after_answers.append({
                    'title': title,
                    'question': question,
                    'response': final_answer,
                    'correct_response': step_3
                })
                save_to_json(f'procedural_knowledge/results_multi/after/{prompter.model_name}_selfcon/{recipe_number}.json', after_answers)
                log_after.update({'final_answer': final_answer,
                                   'correct_answer': step_3})
                logs_after.append(log_after)

    evaluate_prompters.evaluate_multi(prompters, '_selfcon')
    write_general_log_to_file(logs_before, prompter.model_name, 'before_selfcon', 'procedural_knowledge')
    write_general_log_to_file(logs_after, prompter.model_name, 'after_selfcon', 'procedural_knowledge')

def prompt_all_models_multi_selfref(prompters: [Prompter], num_runs: int, n_it: int):
    def get_answer_before(prompter, recipe_title, step_question, other_steps):
        system_msg = (
            "Imagine you are a robot tasked with determining the temporal order of steps in a recipe. "
            "Based on the recipe title and the provided steps, identify which step occurred before another. "
        )
        user_msg_initial = 'Generate your answer in the following format:\nExplanation: <explanation>\nStep before: <step>'

        system_msg_feedback = f'You are given an answer to a multiple choice question regarding the order of steps in a cooking recipe. The possible answers are:\n' + "\n".join(f"- {step}" for step in other_steps)
        user_msg_feedback = "Provide Feedback on the answer. The feedback should only evaluate the correctness of the answer. At the end, score the answer from 1 to 5. A score of 5 means that the answer is the right choice."
        
        system_msg_refine = 'You are given an answer to a multiple choice question regarding the order of steps in a cooking recipe and corresponding feedback.'
        user_msg_refine = 'Improve upon the answer based on the feedback. Remember that the answer has to be chosen from the given options. Generate your answer in the following format:\nExplanation: <explanation>\nStep before: <step>'

        # initial prompt
        question = (
                f"In the recipe '{recipe_title}', which step occurs before '{step_question}'?\n"
                f"\nOptions:\n" + "\n".join(f"- {step}" for step in other_steps))
        answer = prompter.prompt_model(system_msg, user_msg_initial, question)

        # full conversation for logging, LLM only sees last answer
        full_conv = question + f'\n{answer}'

        # Refine - Feedback loop
        for i in range(n_it):
            # feedback 
            f_question = question + f'\n{answer}' + '\nFeedback:'
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
        return transform_prediction(answer, [step for step in other_steps]), question, full_conv

    def get_answer_after(prompter, recipe_title, step_question, other_steps):
        system_msg = (
            "Imagine you are a robot tasked with determining the temporal order of steps in a recipe. "
            "Based on the recipe title and the provided steps, identify which step occurred after another. "
        )
        user_msg_initial = 'Generate your answer in the following format:\nExplanation: <explanation>\nStep after: <step>'

        system_msg_feedback = f'You are given an answer to a multiple choice question regarding the order of steps in a cooking recipe. The possible answers are:\n' + "\n".join(f"- {step}" for step in other_steps)
        user_msg_feedback = "Provide Feedback on the answer. The feedback should only evaluate the correctness of the answer. At the end, score the answer from 1 to 5. A score of 5 means that the answer is the right choice."
        
        system_msg_refine = 'You are given an answer to a multiple choice question regarding the order of steps in a cooking recipe and corresponding feedback.'
        user_msg_refine = 'Improve upon the answer based on the feedback. Remember that the answer has to be chosen from the given options. Generate your answer in the following format:\nExplanation: <explanation>\nStep after: <step>'
        
        # initial prompt
        question = (
                f"In the recipe '{recipe_title}', which steps occurs after '{step_question}'?"
                f"\nOptions:\n" + "\n".join(f"- {step}" for step in other_steps))
        answer = prompter.prompt_model(system_msg, user_msg_initial, question)

        # full conversation for logging, LLM only sees last answer
        full_conv = question + f'\n{answer}'

        # Refine - Feedback loop
        for i in range(n_it):
            # feedback 
            f_question = question + f'\n{answer}' + '\nFeedback:'
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
        return transform_prediction(answer, [step for step in other_steps]), question, full_conv

    for prompter in prompters:
        logs_before = []
        logs_after = []
        for recipe_number in tqdm(range(1, 5), f'Prompting {prompter.model_name} for the multiple choice Procedural Knowledge task with Self-Refine Prompting'):
            json_file = f'procedural_knowledge/data_generation/question_components_multi/questions_recipe_' + str(
                recipe_number) + '.json'
            recipe_components = extract_json_multi_limited(json_file, num_runs)
            before_answers, after_answers = [], []

            for recipe in tqdm(recipe_components, f'Prompting recipes {recipe_number}'):
                title = recipe['goal']
                step_1 = recipe['step_1'].rstrip(". ").strip()
                step_2 = recipe['step_2'].rstrip(". ").strip()
                step_3 = recipe['step_3'].rstrip(". ").strip()

                before_steps = [step_1, step_3]
                available_indices = [i for i in range(len(recipe_components)) if i != recipe_number]
                unique_step_3s = set(before_steps)  # track what's already added
                random.shuffle(available_indices)  # randomize the order
                for idx in available_indices:
                    if len(unique_step_3s) >= 5:  # step_1 + step_3 + 3 unique additions
                        break
                    other_recipe = recipe_components[idx]
                    candidate_step = other_recipe['step_3'].rstrip(". ").strip()
                    if candidate_step not in unique_step_3s:
                        before_steps.append(candidate_step)
                        unique_step_3s.add(candidate_step)
                random.shuffle(before_steps)
                response, question, full_response = get_answer_before(prompter, title, step_2, before_steps)
                before_answers.append({
                    'title': title,
                    'question': question,
                    'response': response,
                    'correct_response': step_1
                })
                save_to_json(f'procedural_knowledge/results_multi/before/{prompter.model_name}_selfref/{recipe_number}.json', before_answers)
                log_before = BasicLogEntry(question, full_response, response, step_1)
                logs_before.append(log_before)


                after_steps = [step_1, step_3]
                available_indices = [i for i in range(len(recipe_components)) if i != recipe_number]
                unique_step_1s = set(after_steps)  # track what's already added
                random.shuffle(available_indices)  # randomize the order
                for idx in available_indices:
                    if len(unique_step_1s) >= 5:  # step_1 + step_3 + 3 unique additions
                        break
                    other_recipe = recipe_components[idx]
                    candidate_step = other_recipe['step_1'].rstrip(". ").strip()
                    if candidate_step not in unique_step_1s:
                        after_steps.append(candidate_step)
                        unique_step_1s.add(candidate_step)

                random.shuffle(after_steps)
                response, question, full_response = get_answer_after(prompter, title, step_2, after_steps)
                after_answers.append({
                    'title': title,
                    'question': question,
                    'response': response,
                    'correct_response': step_3
                })
                save_to_json(f'procedural_knowledge/results_multi/after/{prompter.model_name}_selfref/{recipe_number}.json', after_answers)
                log_after = BasicLogEntry(question, full_response, response, step_3)
                logs_after.append(log_after)

    evaluate_prompters.evaluate_multi(prompters, '_selfref')
    write_log_to_file(logs_before, prompter.model_name, 'before_selfref', 'procedural_knowledge')
    write_log_to_file(logs_after, prompter.model_name, 'after_selfref', 'procedural_knowledge')


def prompt_all_models_multi_stepback(prompters: [Prompter], num_runs: int):
    def get_answer_before(prompter, recipe_title, step_question, other_steps):
        system_msg = (
            "Imagine you are a robot tasked with determining the temporal order of steps in a recipe. "
            "Based on the recipe title and the provided steps, identify which step occurred before another. "
        )
        system_msg_principle = 'You are given a multiple-choice question. Your task is to extract the underlying concepts and principles involved in choosing the right answer.'
        user_msg_principle = 'Only answer with the 5 most important concepts and principles.'

        # Get higher level principles
        p_question = (
                f"In the recipe '{recipe_title}', which step occurs before '{step_question}'?\n"
                f"\nOptions:\n" + "\n".join(f"- {step}" for step in other_steps) + '\nPrinciples:')
        principles = prompter.prompt_model(system_msg_principle, user_msg_principle, p_question)

        # Get answer based on principles
        user_msg_stepback = f'Answer the question step by step using the following principles:\n{principles}\nEnd the answer with the step of your choosing.'
        question = (
                f"In the recipe '{recipe_title}', which step occurs before '{step_question}'?\n"
                f"\nOptions:\n" + "\n".join(f"- {step}" for step in other_steps))
        answer = prompter.prompt_model(system_msg, user_msg_stepback, question)

        return transform_prediction(answer, [step for step in other_steps]), question, answer, p_question, principles

    def get_answer_after(prompter, recipe_title, step_question, other_steps):
        system_msg = (
            "Imagine you are a robot tasked with determining the temporal order of steps in a recipe. "
            "Based on the recipe title and the provided steps, identify which step occurred after another. "
        )
        system_msg_principle = 'You are given a multiple-choice question. Your task is to extract the underlying concepts and principles involved in choosing the right answer.'
        user_msg_principle = 'Only answer with the 5 most important concepts and principles.'

        # Get higher level principles
        p_question = (
                f"In the recipe '{recipe_title}', which steps occurs after '{step_question}'?"
                f"\nOptions:\n" + "\n".join(f"- {step}" for step in other_steps) + '\nPrinciples:')
        principles = prompter.prompt_model(system_msg_principle, user_msg_principle, p_question)

        # Get answer based on principles
        user_msg_stepback = f'Answer the question step by step using the following principles:\n{principles}\nEnd the answer with the step of your choosing.'
        question = (
                f"In the recipe '{recipe_title}', which steps occurs after '{step_question}'?"
                f"\nOptions:\n" + "\n".join(f"- {step}" for step in other_steps))
        answer = prompter.prompt_model(system_msg, user_msg_stepback, question)

        return transform_prediction(answer, [step for step in other_steps]), question, answer, p_question, principles

    for prompter in prompters:
        logs_before = []
        logs_after = []
        for recipe_number in tqdm(range(1, 5), f'Prompting {prompter.model_name} for the multiple choice Procedural Knowledge task with Stepback Prompting'):
            json_file = f'procedural_knowledge/data_generation/question_components_multi/questions_recipe_' + str(
                recipe_number) + '.json'
            recipe_components = extract_json_multi_limited(json_file, num_runs)
            before_answers, after_answers = [], []

            for recipe in tqdm(recipe_components, f'Prompting recipes {recipe_number}'):
                title = recipe['goal']
                step_1 = recipe['step_1'].rstrip(". ").strip()
                step_2 = recipe['step_2'].rstrip(". ").strip()
                step_3 = recipe['step_3'].rstrip(". ").strip()

                before_steps = [step_1, step_3]
                available_indices = [i for i in range(len(recipe_components)) if i != recipe_number]
                unique_step_3s = set(before_steps)  # track what's already added
                random.shuffle(available_indices)  # randomize the order
                for idx in available_indices:
                    if len(unique_step_3s) >= 5:  # step_1 + step_3 + 3 unique additions
                        break
                    other_recipe = recipe_components[idx]
                    candidate_step = other_recipe['step_3'].rstrip(". ").strip()
                    if candidate_step not in unique_step_3s:
                        before_steps.append(candidate_step)
                        unique_step_3s.add(candidate_step)
                random.shuffle(before_steps)
                response, question, full_response, p_question, principles = get_answer_before(prompter, title, step_2, before_steps)
                before_answers.append({
                    'title': title,
                    'question': question,
                    'response': response,
                    'correct_response': step_1
                })
                save_to_json(f'procedural_knowledge/results_multi/before/{prompter.model_name}_stepback/{recipe_number}.json', before_answers)
                log_before = StepbackLogEntry(p_question, principles, question, full_response, response, step_1)
                logs_before.append(log_before)

                after_steps = [step_1, step_3]
                available_indices = [i for i in range(len(recipe_components)) if i != recipe_number]
                unique_step_1s = set(after_steps)  # track what's already added
                random.shuffle(available_indices)  # randomize the order
                for idx in available_indices:
                    if len(unique_step_1s) >= 5:  # step_1 + step_3 + 3 unique additions
                        break
                    other_recipe = recipe_components[idx]
                    candidate_step = other_recipe['step_1'].rstrip(". ").strip()
                    if candidate_step not in unique_step_1s:
                        after_steps.append(candidate_step)
                        unique_step_1s.add(candidate_step)

                random.shuffle(after_steps)
                response, question, full_response, p_question, principles = get_answer_after(prompter, title, step_2, after_steps)
                after_answers.append({
                    'title': title,
                    'question': question,
                    'response': response,
                    'correct_response': step_3
                })
                save_to_json(f'procedural_knowledge/results_multi/after/{prompter.model_name}_stepback/{recipe_number}.json', after_answers)
                log_after = StepbackLogEntry(p_question, principles, question, full_response, response, step_3)
                logs_after.append(log_after)
    evaluate_prompters.evaluate_multi(prompters, '_stepback')
    write_log_to_file(logs_before, prompter.model_name, 'before_stepback', 'procedural_knowledge')
    write_log_to_file(logs_after, prompter.model_name, 'after_stepback', 'procedural_knowledge')


def prompt_all_models_multi_sgicl(prompters: [Prompter], num_runs: int, n_ex: int):
    def get_example_before(prompter, recipe_title, step, step_before):
        system_msg_example = 'You are helping to create questions regarding household environments.'
        user_msg_example = 'For the given two steps, generate the title of a recipe that would involve the two steps. Answer only with the recipe title.'
        question = f'Step: {step}\nStep before: {step_before}\nGenerate a recipe title: {recipe_title}\nGenerate a recipe title:'
        return prompter.prompt_model(system_msg_example, user_msg_example, question)

    def get_example_after(prompter, recipe_title, step, step_before):
        system_msg_example = 'You are helping to create questions regarding household environments.'
        user_msg_example = 'For the given two steps, generate the title of a recipe that would involve the two steps. Answer only with the recipe title.'
        question = f'Step: {step}\nStep after: {step_before}\nGenerate a recipe title: {recipe_title}\nGenerate a recipe title:'
        return prompter.prompt_model(system_msg_example, user_msg_example, question)

    def get_answer_before(prompter, recipe_title, step_question, other_steps, examples_str):
        system_msg = (
            "Imagine you are a robot tasked with determining the temporal order of steps in a recipe. "
            "Based on the recipe title and the provided steps, identify which step occurred before another. "
        )
        user_msg = "Answer only with your chosen step."
        opt_str = f"\nOptions:\n" + "\n".join(f"- {step}" for step in other_steps)
        question = (
                f"Here are a few examples:\n{examples_str}"
                f"Recipe title: {recipe_title}\n"
                f"Step: {step_question}\n"
                f"{opt_str}\n"
                f"Step before:")
        answer = prompter.prompt_model(system_msg, user_msg, question)
        return transform_prediction(answer, [step for step in other_steps]), question, answer

    def get_answer_after(prompter, recipe_title, step_question, other_steps, examples_str):
        system_msg = (
            "Imagine you are a robot tasked with determining the temporal order of steps in a recipe. "
            "Based on the recipe title and the provided steps, identify  which step occurred after another. "
        )
        user_msg = "Answer only with your chosen step."
        opt_str = f"\nOptions:\n" + "\n".join(f"- {step}" for step in other_steps)
        question = (
                f'Here are a few examples:\n{examples_str}\n'
                f"Recipe title: {recipe_title}\n"
                f'Step: {step_question}\n'
                f"{opt_str}\n"
                f"Step after:")
        answer = prompter.prompt_model(system_msg, user_msg, question)
        return transform_prediction(answer, [step for step in other_steps]), question, answer

    for prompter in prompters:
        # Generate examples if needed
        ex_file = f'procedural_knowledge/examples/cooking_procedures_multi_examples_{prompter.model_name}.csv'
        if not os.path.isfile(ex_file):
            results = []
            json_file = f'procedural_knowledge/data_generation/question_components_multi/questions_recipe_' + str(
            1) + '.json'
            recipe_components = extract_json_multi_limited(json_file, 15)
                
            for recipe in tqdm(recipe_components, f'Prompting {prompter.model_name} to generate Procedural Knowledge task examples'):
                title = recipe['goal']
                step_1 = recipe['step_1'].rstrip(". ").strip()  # correct response before
                step_2 = recipe['step_2'].rstrip(". ").strip()  # question step
                step_3 = recipe['step_3'].rstrip(". ").strip()  # correct response after

                ex_before_title = get_example_before(prompter, title, step_2, step_1)
                ex_after_title = get_example_after(prompter, title, step_2, step_3)

                results.append({
                    'step_q': step_2,
                    'before_title': ex_before_title,
                    'before_step': step_1,
                    'after_title': ex_after_title,
                    'after_step': step_3,
                })
            df = pd.DataFrame(results)
            df.to_csv(ex_file, index=False)
            print('Finished generating examples')

        # Load examples
        examples = pd.read_csv(ex_file, delimiter=',', on_bad_lines='skip', nrows=n_ex)
        ex_before_str = ''
        ex_after_str = ''
        for index, row in examples.iterrows():
            step_q = row['step_q']
            before_title = row['before_title']
            before_step = row['before_step']
            after_title = row['after_title']
            after_step = row['after_step']

            ex_before_str = ex_before_str + f'Recipe Title: {before_title}\nStep: {step_q}\nStep before: {before_step}\n'
            ex_after_str = ex_after_str + f'Recipe Title: {after_title}\nStep: {step_q}\nStep after: {after_step}\n'

        logs_before = []
        logs_after = []
        # Few shot prompting
        for recipe_number in tqdm(range(1, 5), f'Prompting {prompter.model_name} for the multiple choice Procedural Knowledge task with SG-ICL Prompting'):
            json_file = f'procedural_knowledge/data_generation/question_components_multi/questions_recipe_' + str(
                recipe_number) + '.json'
            recipe_components = extract_json_multi_limited(json_file, num_runs)
            before_answers, after_answers = [], []

            for recipe in tqdm(recipe_components, f'Prompting recipes {recipe_number}'):
                title = recipe['goal']
                step_1 = recipe['step_1'].rstrip(". ").strip()
                step_2 = recipe['step_2'].rstrip(". ").strip()
                step_3 = recipe['step_3'].rstrip(". ").strip()

                before_steps = [step_1, step_3]
                available_indices = [i for i in range(len(recipe_components)) if i != recipe_number]
                unique_step_3s = set(before_steps)  # track what's already added
                random.shuffle(available_indices)  # randomize the order
                for idx in available_indices:
                    if len(unique_step_3s) >= 5:  # step_1 + step_3 + 3 unique additions
                        break
                    other_recipe = recipe_components[idx]
                    candidate_step = other_recipe['step_3'].rstrip(". ").strip()
                    if candidate_step not in unique_step_3s:
                        before_steps.append(candidate_step)
                        unique_step_3s.add(candidate_step)
                random.shuffle(before_steps)
                response, question, full_response = get_answer_before(prompter, title, step_2, before_steps, ex_before_str)
                before_answers.append({
                    'title': title,
                    'question': question,
                    'response': response,
                    'correct_response': step_1
                })
                save_to_json(f'procedural_knowledge/results_multi/before/{prompter.model_name}_sgicl/{recipe_number}.json', before_answers)
                log_before = BasicLogEntry(question, full_response, response, step_1)
                logs_before.append(log_before)

                after_steps = [step_1, step_3]
                available_indices = [i for i in range(len(recipe_components)) if i != recipe_number]
                unique_step_1s = set(after_steps)  # track what's already added
                random.shuffle(available_indices)  # randomize the order
                for idx in available_indices:
                    if len(unique_step_1s) >= 5:  # step_1 + step_3 + 3 unique additions
                        break
                    other_recipe = recipe_components[idx]
                    candidate_step = other_recipe['step_1'].rstrip(". ").strip()
                    if candidate_step not in unique_step_1s:
                        after_steps.append(candidate_step)
                        unique_step_1s.add(candidate_step)

                random.shuffle(after_steps)
                response, question, full_response = get_answer_after(prompter, title, step_2, after_steps, ex_after_str)
                after_answers.append({
                    'title': title,
                    'question': question,
                    'response': response,
                    'correct_response': step_3
                })
                save_to_json(f'procedural_knowledge/results_multi/after/{prompter.model_name}_sgicl/{recipe_number}.json', after_answers)
                log_after = BasicLogEntry(question, full_response, response, step_3)
                logs_after.append(log_after)
    evaluate_prompters.evaluate_multi(prompters, '_sgicl')
    write_log_to_file(logs_before, prompter.model_name, 'before_sgicl', 'procedural_knowledge')
    write_log_to_file(logs_after, prompter.model_name, 'after_sgicl', 'procedural_knowledge')


def prompt_all_models_multi_contr(prompters: [Prompter], num_runs: int, n_ex: int, n_cot: int):
    def get_cot_before(prompter, cot_right):
        system_msg_rewrite = 'You are helping in rewriting answers to questions regarding household environments.'
        user_msg_rewrite = 'Rewrite the given answer by swapping key points with wrong facts leading to a wrong final answer. Keep the overall structure the same.'
        question = f'Right answer:\n{cot_right}\nWrong answer:'
        return prompter.prompt_model(system_msg_rewrite, user_msg_rewrite, question)
    
    def get_cot_after(prompter, cot_right):
        system_msg_rewrite = 'You are helping in rewriting answers to questions regarding household environments.'
        user_msg_rewrite = 'Rewrite the given answer by swapping key points with wrong facts leading to a wrong final answer. Keep the overall structure the same.'
        question = f'Right answer:\n{cot_right}\nWrong answer:'
        return prompter.prompt_model(system_msg_rewrite, user_msg_rewrite, question)

    def get_answer_before(prompter, recipe_title, step_question, other_steps, examples_str):
        system_msg = (
            "Imagine you are a robot tasked with determining the temporal order of steps in a recipe. "
            "Based on the recipe title and the provided steps, identify which step occurred before another. "
        )
        user_msg = "Answer only with your chosen step."
        opt_str = f"\nOptions:\n" + "\n".join(f"- {step}" for step in other_steps)
        question = (
                f"Here are a few examples:\n{examples_str}"
                f"Recipe title: {recipe_title}\n"
                f"Step: {step_question}\n"
                f"{opt_str}\n"
                f"Step before:")
        answer = prompter.prompt_model(system_msg, user_msg, question)
        return transform_prediction(answer, [step for step in other_steps]), question, answer

    def get_answer_after(prompter, recipe_title, step_question, other_steps, examples_str):
        system_msg = (
            "Imagine you are a robot tasked with determining the temporal order of steps in a recipe. "
            "Based on the recipe title and the provided steps, identify  which step occurred after another. "
        )
        user_msg = "Answer only with your chosen step."
        opt_str = f"\nOptions:\n" + "\n".join(f"- {step}" for step in other_steps)
        question = (
                f'Here are a few examples:\n{examples_str}\n'
                f"Recipe title: {recipe_title}\n"
                f'Step: {step_question}\n'
                f"{opt_str}\n"
                f"Step after:")
        answer = prompter.prompt_model(system_msg, user_msg, question)
        return transform_prediction(answer, [step for step in other_steps]), question, answer


    for prompter in prompters:
        # Generate before examples if needed
        ex_before_file = f'procedural_knowledge/examples/cooking_procedures_multi_cot_before_examples_{prompter.model_name}.csv'
        if not os.path.isfile(ex_before_file):
            results = []
            log_before = pd.read_csv(f'procedural_knowledge/logs/{prompter.model_name}/{prompter.model_name}_before_selfcon.csv', delimiter=',', on_bad_lines='skip', nrows=10)
            for index, row in tqdm(log_before.iterrows(),
                                f'Prompting {prompter.model_name} to generate Procedural Knowledge task examples (before)'):
                corr_conf = row['correct_answer']
                # get correct cot
                cot_right = ''
                for i in range(n_cot):
                    answ = row[f'answer_{i}']
                    if answ == corr_conf:
                        cot_right = row[f'cot_{i}']
                        break
                if cot_right == '': continue

                cot_wrong = get_cot_before(prompter, cot_right)
                entry = {
                    'question': row['question'],
                    'cot_right': cot_right,
                    'cot_wrong': cot_wrong
                }
                results.append(entry)
            df = pd.DataFrame(results)
            df.to_csv(ex_before_file, index=False)
            print('Finished generating before examples')

        # Generate after examples if needed
        ex_after_file = f'procedural_knowledge/examples/cooking_procedures_multi_cot_after_examples_{prompter.model_name}.csv'
        if not os.path.isfile(ex_after_file):
            results = []
            log_after = pd.read_csv(f'procedural_knowledge/logs/{prompter.model_name}/{prompter.model_name}_after_selfcon.csv', delimiter=',', on_bad_lines='skip', nrows=10)
            for index, row in tqdm(log_after.iterrows(),
                                f'Prompting {prompter.model_name} to generate Procedural Knowledge task examples (after)'):
                corr_conf = row['correct_answer']
                # get correct cot
                cot_right = ''
                for i in range(n_cot):
                    answ = row[f'answer_{i}']
                    if answ == corr_conf:
                        cot_right = row[f'cot_{i}']
                        break
                if cot_right == '': continue

                cot_wrong = get_cot_after(prompter, cot_right)
                entry = {
                    'question': row['question'],
                    'cot_right': cot_right,
                    'cot_wrong': cot_wrong
                }
                results.append(entry)
            df = pd.DataFrame(results)
            df.to_csv(ex_after_file, index=False)
            print('Finished generating after examples')

        # Load examples (before)
        examples = pd.read_csv(ex_before_file, delimiter=',', on_bad_lines='skip', nrows=n_ex)
        ex_before_str = ''
        for index, row in examples.iterrows():
            question = row['question']
            cot_right = row['cot_right']
            cot_wrong = row['cot_wrong']
            ex_before_str = ex_before_str + f'Question: {question}\n\nRight Explanation: {cot_right}\n\nWrong Explanation: {cot_wrong}\n\n'
        
        # Load examples (after)
        examples = pd.read_csv(ex_after_file, delimiter=',', on_bad_lines='skip', nrows=n_ex)
        ex_after_str = ''
        for index, row in examples.iterrows():
            question = row['question']
            cot_right = row['cot_right']
            cot_wrong = row['cot_wrong']
            ex_after_str = ex_after_str + f'Question: {question}\n\nRight Explanation: {cot_right}\n\nWrong Explanation: {cot_wrong}\n\n'

        logs_before = []
        logs_after = []
        # Few shot prompting
        for recipe_number in tqdm(range(1, 5), f'Prompting {prompter.model_name} for the multiple choice Procedural Knowledge task with Contrastive CoT Prompting'):
            json_file = f'procedural_knowledge/data_generation/question_components_multi/questions_recipe_' + str(
                recipe_number) + '.json'
            recipe_components = extract_json_multi_limited(json_file, num_runs)
            before_answers, after_answers = [], []

            for recipe in tqdm(recipe_components, f'Prompting recipes {recipe_number}'):
                title = recipe['goal']
                step_1 = recipe['step_1'].rstrip(". ").strip()
                step_2 = recipe['step_2'].rstrip(". ").strip()
                step_3 = recipe['step_3'].rstrip(". ").strip()

                before_steps = [step_1, step_3]
                available_indices = [i for i in range(len(recipe_components)) if i != recipe_number]
                unique_step_3s = set(before_steps)  # track what's already added
                random.shuffle(available_indices)  # randomize the order
                for idx in available_indices:
                    if len(unique_step_3s) >= 5:  # step_1 + step_3 + 3 unique additions
                        break
                    other_recipe = recipe_components[idx]
                    candidate_step = other_recipe['step_3'].rstrip(". ").strip()
                    if candidate_step not in unique_step_3s:
                        before_steps.append(candidate_step)
                        unique_step_3s.add(candidate_step)
                random.shuffle(before_steps)
                response, question, full_response = get_answer_before(prompter, title, step_2, before_steps, ex_before_str)
                before_answers.append({
                    'title': title,
                    'question': question,
                    'response': response,
                    'correct_response': step_1
                })
                save_to_json(f'procedural_knowledge/results_multi/before/{prompter.model_name}_contr/{recipe_number}.json', before_answers)
                log_before = BasicLogEntry(question, response, step_1)
                logs_before.append(log_before)

                after_steps = [step_1, step_3]
                available_indices = [i for i in range(len(recipe_components)) if i != recipe_number]
                unique_step_1s = set(after_steps)  # track what's already added
                random.shuffle(available_indices)  # randomize the order
                for idx in available_indices:
                    if len(unique_step_1s) >= 5:  # step_1 + step_3 + 3 unique additions
                        break
                    other_recipe = recipe_components[idx]
                    candidate_step = other_recipe['step_1'].rstrip(". ").strip()
                    if candidate_step not in unique_step_1s:
                        after_steps.append(candidate_step)
                        unique_step_1s.add(candidate_step)

                random.shuffle(after_steps)
                response, question, full_response = get_answer_after(prompter, title, step_2, after_steps, ex_after_str)
                after_answers.append({
                    'title': title,
                    'question': question,
                    'response': response,
                    'correct_response': step_3
                })
                save_to_json(f'procedural_knowledge/results_multi/after/{prompter.model_name}_contr/{recipe_number}.json', after_answers)
                log_after = BasicLogEntry(question, full_response, response, step_3)
                logs_after.append(log_after)
    evaluate_prompters.evaluate_multi(prompters, '_contr')
    write_log_to_file(logs_before, prompter.model_name, 'before_contr', 'procedural_knowledge')
    write_log_to_file(logs_after, prompter.model_name, 'after_contr', 'procedural_knowledge')