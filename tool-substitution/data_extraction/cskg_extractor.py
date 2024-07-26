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

    res = combine_all_tuples(tuples)
    for r in res:
        r.process_affordances()
    return res


if __name__ == '__main__':
    res = get_affordances_from_cskg()
    count_aff = 0
    count_fil = 0
    unique_aff = set()
    for r in res:
        if r.verify():
            count_fil += 1
            count_aff += len(r.get_affordances())
            for aff in r.get_affordances():
                unique_aff.add(aff.get_affordance())
        print(r)
    print(
        f'Objects: {len(res)}\nFiltered Objects: {count_fil}\nAffordances: {count_aff}\nUnique Affordances: {len(unique_aff)}')
