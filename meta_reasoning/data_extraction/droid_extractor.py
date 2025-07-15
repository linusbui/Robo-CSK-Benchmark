import pandas as pd
from tqdm import tqdm

from gripper_configs import GripperConfig
from task_capability import TaskCapability


def get_task_capabilities_from_droid() -> [TaskCapability]:
    data = pd.read_csv("../data/droid_task_descriptions.csv", delimiter=',', on_bad_lines='skip')
    res = []

    for index, row in tqdm(data.iterrows(), 'Collecting data from DROID'):
        goal = row['Task']
        task = TaskCapability(goal,False, 1, 7, GripperConfig.TWO_FINGERS, True)
        res.append(task)
    return res


if __name__ == '__main__':
    res = get_task_capabilities_from_droid()
    for r in res:
        print(r)
    print(f'Found {len(res)} tasks')
