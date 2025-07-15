import os

from tqdm import tqdm

from gripper_configs import GripperConfig
from task_capability import TaskCapability


def get_task_capabilities_from_virtualhome() -> [TaskCapability]:
    res = []
    folder_path = "../data/virtualhome_data/"
    for file_name in tqdm(os.listdir(folder_path), 'Collecting data from VirtualHome'):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, 'r') as f:
                    goal = f.readline().strip()
                    task = TaskCapability(goal, True, 2, 6, GripperConfig.NO, False)
                    res.append(task)
            except Exception as e:
                print(f"Error reading file {file_name}: {e}")
    return res


if __name__ == '__main__':
    res = get_task_capabilities_from_virtualhome()
    for r in res:
        print(r)
    print(f'Found {len(res)} tasks')
