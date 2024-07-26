import json

from object_affordance_tuple import ObjectAffordanceTuple, combine_all_tuples


def get_affordances_from_narrative_objects_export() -> [ObjectAffordanceTuple]:
    with open("../data/narrative_tools.json") as f:
        data = json.load(f)

    tuples = []
    for tool, disp_list in data.items():
        for disp in disp_list:
            proc_tool = tool.replace("soma:", "")
            proc_disp = disp.replace("soma:", "").lower()
            obj_loc = ObjectAffordanceTuple(proc_tool, proc_disp, 'Narrative Objects', False)
            tuples.append(obj_loc)

    return combine_all_tuples(tuples)


if __name__ == '__main__':
    res = get_affordances_from_narrative_objects_export()
    count_aff = 0
    unique_aff = set()
    for r in res:
        r.process_affordances()
        count_aff += len(r.get_affordances())
        for aff in r.get_affordances():
            unique_aff.add(aff.get_affordance())
        print(r)
    print(f'Objects: {len(res)}\nAffordances: {count_aff}\nUnique Affordances: {len(unique_aff)}')
