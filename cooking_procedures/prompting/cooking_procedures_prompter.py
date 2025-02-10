import pandas as pd

from cooking_procedures.json_utils import extract_json, save_to_json
from cooking_procedures.prompting import evaluate_prompters
from utils.prompter import Prompter

system_msg = "Imagine you are a robot tasked with determining the temporal order of two steps from one recipe. "
system_msg_before = "Based on the recipe title and the two steps provided, identify whether one action occurred before another. "
system_msg_after = "Based on the recipe title and the two steps provided, identify whether one action occurred after another. "
user_msg = "Answer only with 'Yes' or 'No'."

def prompt_all_models(prompters: [Prompter]):
    # for prompter in prompters:
    #     for recipe_number in range(1, 1):
    #         json_file = f'cooking_procedures/data_generation/question_components/Order/questions_recipe_' + str(recipe_number) + '.json'
    #         recipe_components = extract_json(json_file)
    #
    #         all_yes_before_answers, all_no_before_answers = [], []
    #         all_yes_after_answers, all_no_after_answers = [], []
    #         for recipe in recipe_components:
    #             title = recipe['goal']
    #             step_1 = recipe['step_1'].rstrip(". ").strip()
    #             step_2 = recipe['step_2'].rstrip(". ").strip()
    #
    #             for steps, correct_response, answers in [
    #                 ((step_1, step_2), "Yes", all_yes_before_answers),
    #                 ((step_2, step_1), "No", all_no_before_answers),
    #             ]:
    #                 question = f"In the recipe '{title}', did '{steps[0]}' occur before '{steps[1]}'?"
    #                 response = prompter.prompt_model(system_msg + system_msg_before, user_msg, question)
    #                 answers.append({
    #                     'title': title,
    #                     'question': system_msg + system_msg_before + user_msg + question,
    #                     'response': response,
    #                     'correct_response': correct_response
    #                 })
    #                 # Save the result to json after every step if an error happens
    #                 # save_to_json(f'results/Yes/{model}/before_{recipe_number}.json', all_yes_before_answers)
    #                 # save_to_json(f'results/No/{model}/before_{recipe_number}.json', all_no_before_answers)
    #
    #             for steps, correct_response, answers in [
    #                 ((step_1, step_2), "Yes", all_yes_after_answers),
    #                 ((step_2, step_1), "No", all_no_after_answers),
    #             ]:
    #                 question = f"In the recipe '{title}', did '{steps[1]}' occur after '{steps[0]}'?"
    #                 response = prompter.prompt_model(system_msg + system_msg_after, user_msg, question)
    #                 answers.append({
    #                     'title': title,
    #                     'question': system_msg + system_msg_after + user_msg + question,
    #                     'response': response,
    #                     'correct_response': correct_response
    #                 })
    #             # save_to_json(f'results/Yes/{model}/after_{recipe_number}.json', all_yes_after_answers)
    #             # save_to_json(f'results/No/{model}/after_{recipe_number}.json', all_no_after_answers)
    evaluate_prompters.evaluate(prompters)