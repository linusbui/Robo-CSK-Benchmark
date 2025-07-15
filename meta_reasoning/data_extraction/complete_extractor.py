import pandas as pd
from tqdm import tqdm

from alfred_extractor import get_task_capabilities_from_alfred
from capability_combiner import combine_capabilities
from droid_extractor import get_task_capabilities_from_droid
from gripper_configs import GripperConfig
from rh20t_extractor import get_task_capabilities_from_rh20t
from task_capability import TaskCapability
from virtual_home_extractor import get_task_capabilities_from_virtualhome


def extract_from_all_sources() -> [TaskCapability]:
    res_alfred = get_task_capabilities_from_alfred()
    res_droid = get_task_capabilities_from_droid()
    res_rh20t = get_task_capabilities_from_rh20t()
    res_virtual_home = get_task_capabilities_from_virtualhome()
    return res_alfred + res_droid + res_rh20t + res_virtual_home


def create_negative_samples(positive_samples: [TaskCapability]) -> [TaskCapability]:
    samples = []
    for p_s in tqdm(positive_samples, 'Creating negative samples'):
        n_s = TaskCapability(p_s.get_task(), False, 1, 1, GripperConfig.NO, True)
        if str(n_s) == str(p_s):
            continue
        samples.append(p_s)
        samples.append(n_s)
    return samples


def write_results_to_file(results: [TaskCapability]):
    dict_list = [re.to_dict() for re in results]
    df = pd.DataFrame(dict_list)
    df.to_csv('../meta_reasoning_data.csv', index=False)


if __name__ == '__main__':
    res = extract_from_all_sources()
    res = [r for r in res if r.verify()]
    print(f'Finished the initial extraction. {len(res)} tasks found.')
    res = combine_capabilities(res)
    print(f'Finished the combination. {len(res)} tasks remain.')
    res = create_negative_samples(res)
    print(f'Added the negative samples. {len(res)} overall samples.')
    write_results_to_file(sorted(res, key=lambda r: r.get_task()))
