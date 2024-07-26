import json

from object_affordance_tuple import ObjectAffordanceTuple, combine_all_tuples


def get_affordances_from_rocs() -> [ObjectAffordanceTuple]:
    rocs_factor = 0.75
    with open("../data/object_concept_knowledge.json") as f:
        data = json.load(f)

    aff_tuples = []
    for key_obj, value_dict in data.items():
        for key_aff, value_float in value_dict.items():
            if value_float >= 0.5:
                aff = key_aff[:-2]
                rocs_trust = round(value_float*rocs_factor, 2)
                obj_loc = ObjectAffordanceTuple(key_obj, aff, 'RoCS', False, rocs_trust)
                aff_tuples.append(obj_loc)
    return combine_all_tuples(aff_tuples)


if __name__ == '__main__':
    res = get_affordances_from_rocs()
    count_aff = 0
    unique_aff = set()
    for r in res:
        r.process_affordances()
        count_aff += len(r.get_affordances())
        for aff in r.get_affordances():
            unique_aff.add(aff.get_affordance())
        print(r)
    print(f'Objects: {len(res)}\nAffordances: {count_aff}\nUnique Affordances: {len(unique_aff)}')
