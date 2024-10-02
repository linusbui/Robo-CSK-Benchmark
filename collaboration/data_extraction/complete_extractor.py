import pandas as pd

from alfred_extractor import get_task_capabilities_from_alfred
from capability_combiner import combine_capabilities
from droid_extractor import get_task_capabilities_from_droid
from rh20t_extractor import get_task_capabilities_from_rh20t
from task_capability import TaskCapability
from virtual_home_extractor import get_task_capabilities_from_virtualhome


def extract_from_all_sources() -> [TaskCapability]:
    res_alfred = get_task_capabilities_from_alfred()
    res_droid = get_task_capabilities_from_droid()
    res_rh20t = get_task_capabilities_from_rh20t()
    res_virtual_home = get_task_capabilities_from_virtualhome()
    return res_alfred + res_droid + res_rh20t + res_virtual_home


def write_results_to_file(results: [TaskCapability]):
    dict_list = [re.to_dict() for re in results]
    df = pd.DataFrame(dict_list)
    df.to_csv('../collaboration_data.csv', index=False)


if __name__ == '__main__':
    res = extract_from_all_sources()
    res = [r for r in res if r.verify()]
    res = combine_capabilities(res)
    write_results_to_file(sorted(res, key=lambda r: r.get_task()))
    print(f'Finished the combination. {len(res)} tasks remain.')
