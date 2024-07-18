import pandas as pd

from object_affordance_tuple import ObjectAffordanceTuple, combine_all_tuples, get_trust_for_source


def get_affordances_from_visual_dataset() -> [ObjectAffordanceTuple]:
    data = pd.read_csv("../data/Object_Categories-Affordance_Labels.xlsx - CORS0050.csv", delimiter=',', on_bad_lines='skip')
    data = data.dropna(axis=1, how='all')
    tuples = []

    for index, row in data.iterrows():
        obj = row['Object Class']
        for col in data.columns[2:]:
            aff = row[col]
            if not pd.isna(aff):
                obj_aff = ObjectAffordanceTuple(obj)
                obj_aff.add_affordance(aff, source="Visual Dataset")
                tuples.append(obj_aff)

    return combine_all_tuples(tuples)


if __name__ == '__main__':
    res = get_affordances_from_visual_dataset()
    for r in res:
        print(r)
