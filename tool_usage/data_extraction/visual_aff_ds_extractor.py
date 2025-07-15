import pandas as pd
from tqdm import tqdm

from object_affordance_tuple import ObjectAffordanceTuple, combine_all_tuples


def get_affordances_from_visual_dataset() -> [ObjectAffordanceTuple]:
    data = pd.read_csv("../data/Object_Categories-Affordance_Labels.xlsx - CORS0050.csv", delimiter=',', on_bad_lines='skip')
    data = data.dropna(axis=1, how='all')
    tuples = []

    for index, row in tqdm(data.iterrows(), 'Collecting data from the Visual Affordance dataset'):
        obj = row['Object Class']
        for col in data.columns[2:]:
            aff = row[col]
            if not pd.isna(aff):
                obj_aff = ObjectAffordanceTuple(obj, aff.lower().strip(), "Visual Dataset", False)
                tuples.append(obj_aff)

    return combine_all_tuples(tuples)


if __name__ == '__main__':
    res = get_affordances_from_visual_dataset()
    count_aff = 0
    unique_aff = set()
    for r in res:
        r.process_affordances()
        count_aff += len(r.get_affordances())
        for aff in r.get_affordances():
            unique_aff.add(aff.get_affordance())
        print(r)
    print(f'Objects: {len(res)}\nAffordances: {count_aff}\nUnique Affordances: {len(unique_aff)}')
