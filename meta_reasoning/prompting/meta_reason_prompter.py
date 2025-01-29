import pandas as pd

from meta_reasoning.data_extraction.gripper_configs import GripperConfig
from meta_reasoning.prompting.meta_reason_model_result import MetaReasoningModelResult
from utils.prompter import Prompter

system_msg = 'Imagine you are a robot with a given hardware configuration and you should decide whether you are capable of executing a task.'
user_msg = 'Please only answer with Yes or No.'


def prompt_all_models(prompters: [Prompter]):
    comb_result = pd.DataFrame(columns=['model', 'tn', 'tp', 'fn', 'fp', 'ratio', 'acc', 'prec', 'rec', 'spec', 'f1'])
    for prompter in prompters:
        data = pd.read_csv('meta_reasoning/meta_reasoning_data.csv', delimiter=',', on_bad_lines='skip')
        results = []
        for index, row in data.iterrows():
            # get data from csv
            task = row['Task']
            is_mobile = row['Mobile?']
            no_arms = row['Arms']
            dofs = row['DoFs']
            gripper = row['Gripper Config']
            has_rigid_gripper = row['Rigid Gripper?']

            # create the two hardware descriptions
            pos_hardware = create_hardware_description(is_mobile, no_arms, dofs, gripper, has_rigid_gripper)
            neg_hardware = make_hardware_negative(is_mobile, no_arms, dofs, gripper, has_rigid_gripper)
            # only include both of them if a negative configuration can be created
            if neg_hardware != "minimal":
                pos_question = f'Task: {task}\nHardware: {pos_hardware}'
                pos_res = prompter.prompt_model(system_msg, user_msg, pos_question)
                pos_tup = MetaReasoningModelResult(task, pos_hardware, get_binary_answer(pos_res))
                results.append(pos_tup)
                neg_question = f'Task: {task}\nHardware: {neg_hardware}'
                neg_res = prompter.prompt_model(system_msg, user_msg, neg_question)
                neg_tup = MetaReasoningModelResult(task, neg_hardware, get_binary_answer(neg_res), False)
                results.append(neg_tup)
        write_results_to_file(results, prompter.model_name)
        comb_result = pd.concat([comb_result, calculate_metrics(results, prompter.model_name)], ignore_index=True)
    comb_result.to_csv('meta_reasoning/results/model_overview.csv', index=False)


def make_hardware_negative(is_mobile: bool, arms: int, dofs: int, gripper: str, is_rigid: bool) -> str:
    original = create_hardware_description(is_mobile, arms, dofs, gripper, is_rigid)
    arms = 1
    dofs = 1
    is_mobile = False
    gripper = GripperConfig.NO
    changed = create_hardware_description(is_mobile, arms, dofs, gripper, is_rigid)
    if changed == original:
        return "minimal"
    else:
        return changed


def create_hardware_description(is_mobile: bool, arms: int, dofs: int, gripper: str, is_rigid: bool) -> str:
    if is_mobile:
        walk = 'can walk'
    else:
        walk = 'can not walk'
    if is_rigid:
        rigidity = 'rigid'
    else:
        rigidity = 'soft'
    return f'The robot has {arms} arm(s) with {dofs} DoFs and {rigidity} {gripper} and it {walk}.'


def write_results_to_file(results: [MetaReasoningModelResult], model: str):
    dict_list = [re.to_dict() for re in results]
    df = pd.DataFrame(dict_list)
    df.to_csv(f'meta_reasoning/results/{model.lower()}.csv', index=False)


def get_binary_answer(answer: str) -> bool:
    return 'yes' in answer.lower()


def calculate_metrics(results: [MetaReasoningModelResult], model: str):
    assert len(results) > 0
    counter = {met: 0 for met in ['tn', 'tp', 'fn', 'fp', 'ratio']}
    for res in results:
        ct = res.get_classification_type()
        counter[ct] += 1
    assert (counter['tn'] + counter['tp'] + counter['fn'] + counter['fp']) == len(results)

    if (counter['tp'] + counter['fp']) == 0:
        precision = 0.0
    else:
        precision = counter['tp'] / (counter['tp'] + counter['fp'])
    if (counter['tp'] + counter['fn']) == 0:
        recall = 0.0
    else:
        recall = counter['tp'] / (counter['tp'] + counter['fn'])
    if (counter['tn'] + counter['fp']) == 0:
        specificity = 0.0
    else:
        specificity = counter['tn'] / (counter['tn'] + counter['fp'])

    new_row = pd.Series(
        {'model': model, 'tn': counter['tn'], 'tp': counter['tp'], 'fn': counter['fn'], 'fp': counter['fp'],
         'ratio': (counter['tp'] + counter['fp']) / (counter['tn'] + counter['fn']),
         'acc': (counter['tp'] + counter['tn']) / len(results), 'prec': precision, 'rec': recall, 'spec': specificity,
         'f1': 2 * precision * recall / (precision + recall)})
    return new_row.to_frame().T
