import json

from task_capability import TaskCapability


def get_task_capabilities_from_rh20t() -> [TaskCapability]:
    with open("../data/rh20t_task_description.json") as f:
        data = json.load(f)

    res = []
    for key, info in data.items():
        goal = info.get('task_description_english')
        if '?' in goal:
            continue
        task = TaskCapability(goal, 1, 6, False)
        res.append(task)
    return res


if __name__ == '__main__':
    res = get_task_capabilities_from_rh20t()
    for r in res:
        print(r)
    print(f'Found {len(res)} tasks')
