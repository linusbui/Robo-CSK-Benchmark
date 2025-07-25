import pandas as pd
import random

from procedural_knowledge.json_utils import extract_json, extract_json_multi, save_to_json
from procedural_knowledge.prompting import evaluate_prompters
from tidy_up.prompting.tidy_up_prompter_open import user_msg
from utils.prompter import Prompter

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


def prompt_all_models_multi(prompters: [Prompter]):
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
            recipe_components = extract_json_multi(json_file)
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
    evaluate_prompters.evaluate_multi(prompters)
