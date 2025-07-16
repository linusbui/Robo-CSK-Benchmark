import random

import pandas as pd
from tqdm import tqdm

from meta_reasoning.data_extraction.gripper_configs import GripperConfig


def create_multi_choice_questions():
    question_data = pd.DataFrame(columns=['Task', 'Correct_Configuration', 'Wrong_Configurations'])
    meta_reason = pd.read_csv('../meta_reasoning_data.csv', delimiter=',', on_bad_lines='skip')
    for idx, row in tqdm(meta_reason.iterrows(), f'Creating multiple choice questions'):
        # turn hardware data into configuration description
        is_mobile = row['Mobile?']
        no_arms = row['Arms']
        dofs = row['DoFs']
        gripper = row['Gripper Config']
        has_rigid_gripper = row['Rigid Gripper?']
        corr_hardware = create_hardware_description(is_mobile, no_arms, dofs, gripper, has_rigid_gripper)

        task = row['Task']
        choices = get_wrong_configurations(is_mobile, no_arms, dofs, gripper, has_rigid_gripper)
        random.shuffle(choices)
        new_row = pd.DataFrame([{
            'Task': task,
            'Correct_Configuration': corr_hardware,
            'Wrong_Configurations': choices
        }])
        question_data = pd.concat([question_data, new_row], ignore_index=True)
    question_data.to_csv('../meta_reasoning_multi_questions.csv', index=False)


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


def get_wrong_configurations(is_mobile: bool, arms: int, dofs: int, gripper: str, is_rigid: bool, amount=4) -> [str]:
    choices = [get_minimal_config()]
    corr_conf = create_hardware_description(is_mobile, arms, dofs, gripper, is_rigid)
    while len(choices) < amount:
        is_mobile = is_mobile
        arms = arms
        dofs = dofs
        gripper = GripperConfig(gripper)
        is_rigid = is_rigid

        w_dim = random.randint(1, 5)
        if w_dim == 1:
            is_mobile = False
        elif w_dim == 2:
            arms = arms - 1 if arms > 0 else 1
        elif w_dim == 3:
            dofs = dofs - 1 if dofs > 0 else 1
        elif w_dim == 4:
            gripper = gripper.add(-1) if gripper != GripperConfig.NO else GripperConfig.NO
        elif w_dim == 5:
            is_rigid = True

        wrong_conf = create_hardware_description(is_mobile, arms, dofs, gripper, is_rigid)
        if wrong_conf not in choices and wrong_conf != corr_conf:
            choices.append(wrong_conf)
    return choices


def get_minimal_config() -> str:
    return create_hardware_description(False, 1, 1, GripperConfig.NO, True)


if __name__ == '__main__':
    create_multi_choice_questions()
