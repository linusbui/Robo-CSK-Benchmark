import pandas as pd

from object_affordance_tuple import ObjectAffordanceTuple, combine_all_tuples


def get_affordances_from_cskg() -> [ObjectAffordanceTuple]:
    data = pd.read_csv("../data/cskg.tsv", delimiter='\t', on_bad_lines='skip')
    filtered = data[data['relation;label'] == 'used for']
    tuples = []

    for index, row in filtered.iterrows():
        obj = row['node1;label']
        aff = row['node2;label']
        obj_aff = ObjectAffordanceTuple(obj, aff, "CSKG")
        tuples.append(obj_aff)

    return combine_all_tuples(tuples)


if __name__ == '__main__':
    res = get_affordances_from_cskg()
    for r in res:
        print(r)
