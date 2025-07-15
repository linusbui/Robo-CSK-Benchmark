import json

from tqdm import tqdm

from gripper_configs import GripperConfig
from task_capability import TaskCapability


def get_task_capabilities_from_rh20t() -> [TaskCapability]:
    with open("../data/rh20t_task_description.json") as f:
        data = json.load(f)

    res = []
    for key, info in tqdm(data.items(), 'Collecting data from RH20T'):
        goal = info.get('task_description_english')
        if '?' in goal:
            continue
        task = TaskCapability(goal, False, 1, 6, GripperConfig.TWO_FINGERS, True)
        res.append(task)
    return res


if __name__ == '__main__':
    res = get_task_capabilities_from_rh20t()
    for r in res:
        print(r)
    print(f'Found {len(res)} tasks')
