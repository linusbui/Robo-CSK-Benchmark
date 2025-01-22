import json
import re

from object_affordance_tuple import ObjectAffordanceTuple, combine_all_tuples


def get_affordances_from_coat() -> [ObjectAffordanceTuple]:
    with open("../data/coat_objects.json") as f:
        data = json.load(f)

    tuples = []
    for aff, obj_list in data.items():
        proc_aff = aff.lower()
        for obj in obj_list:
            proc_obj = camel_case_to_spaces(obj)
            obj_loc = ObjectAffordanceTuple(proc_obj, proc_aff, 'COAT', False)
            tuples.append(obj_loc)

    return combine_all_tuples(tuples)


def camel_case_to_spaces(camel_case_string):
    spaced_string = re.sub(r'(?<!^)(?=[A-Z])', ' ', camel_case_string)
    return spaced_string


if __name__ == '__main__':
    res = get_affordances_from_coat()
    count_aff = 0
    unique_aff = set()
    for r in res:
        r.process_affordances()
        count_aff += len(r.get_affordances())
        for aff in r.get_affordances():
            unique_aff.add(aff.get_affordance())
        print(r)
    print(f'Objects: {len(res)}\nAffordances: {count_aff}\nUnique Affordances: {len(unique_aff)}')
