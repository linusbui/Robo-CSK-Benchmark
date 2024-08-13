import json
from pathlib import Path


def get_activity_body_combinations_from_alfred() -> [(str, int, bool, str)]:
    folder_path = Path('../data/alfred_data/')
    res = []

    for json_file in folder_path.glob('*.json'):
        with json_file.open('r') as f:
            data = json.load(f)
            tasks = data['turk_annotations']
            for task in tasks['anns']:
                goal = task['task_desc']
                desc = task['high_descs']
                count_arms, walk = analyse_task_steps(desc)
                res.append((goal, count_arms, walk, json_file.name))
    return res


def analyse_task_steps(desc: str) -> (int, bool):
    count_arms = 0
    walk = False
    pick_ups = 0
    place_downs = 0
    for step in desc:
        if is_walk_step(step.lower()):
            walk = True
        if is_pick_up_step(step.lower()):
            pick_ups += 1
        if is_place_down_step(step.lower()):
            place_downs += 1
        count_arms = max(count_arms, pick_ups - place_downs)
    return count_arms, walk


def is_pick_up_step(step: str) -> bool:
    if ("pick" in step and "up" in step) or "grab" in step:
        return True


def is_place_down_step(step: str) -> bool:
    if "place" in step:
        return True


def is_walk_step(step: str) -> bool:
    if "walk" in step:
        return True


if __name__ == '__main__':
    res = get_activity_body_combinations_from_alfred()
    for r in res:
        if r[2]:
            walk = 'does need to walk'
        else:
            walk = 'doesn\'t need to walk'
        print(f'To {r[0].lower().strip()} the robot needs {r[1]} arms and {walk} ({r[3]}).')
