import json

import pandas as pd

from coat_extractor import get_affordances_from_coat
from cskg_extractor import get_affordances_from_cskg
from narrative_objects_extractor import get_affordances_from_narrative_objects_export
from object_affordance_tuple import ObjectAffordanceTuple, combine_all_tuples
from rocs_extractor import get_affordances_from_rocs
from visual_aff_ds_extractor import get_affordances_from_visual_dataset
from ai2thor_extractor import get_object_affordances_from_ai2thor


def extract_from_all_sources() -> [ObjectAffordanceTuple]:
    res_cskg = get_affordances_from_cskg()
    res_narrative_objects = get_affordances_from_narrative_objects_export()
    res_rocs = get_affordances_from_rocs()
    res_visual_dataset = get_affordances_from_visual_dataset()
    res_coat = get_affordances_from_coat()
    res_ai2thor = get_object_affordances_from_ai2thor()
    return res_cskg + res_narrative_objects + res_rocs + res_visual_dataset + res_coat + res_ai2thor


def filter_combined_results(results: [ObjectAffordanceTuple]):
    thresh = 0.5
    for r in results:
        to_rem = []
        for aff in r.get_affordances():
            if aff.get_trust() <= thresh:
                to_rem.append(aff)
        r.remove_affordances(to_rem)


def write_affordance_list(res: [ObjectAffordanceTuple], before_map=True):
    unique_aff = set()
    for r in res:
        for aff in r.get_affordances():
            unique_aff.add(aff.get_affordance())
    if before_map:
        with open("../affordances.json", "w") as f:
            json.dump(sorted(list(unique_aff)), f)
    else:
        with open("../affordances_after_mapping.json", "w") as f:
            json.dump(sorted(list(unique_aff)), f)


def map_affordances(res: [ObjectAffordanceTuple]) -> [ObjectAffordanceTuple]:
    with open("../affordance_map.json") as f:
        aff_map = json.load(f)

    for r in res:
        to_rem = []
        for aff in r.get_affordances():
            mapped = aff_map.get(aff.get_affordance())
            if mapped == "None":
                to_rem.append(aff)
            else:
                aff.rename_affordance(mapped)
        r.remove_affordances(to_rem)
        r.combine_affordances()
    return res


def write_results_to_file(results: [ObjectAffordanceTuple]):
    dict_list = [re.to_dict() for re in sorted(results, key=lambda r: r.get_object())]
    df = pd.DataFrame(dict_list)
    df.to_csv('../affordance_data.csv', index=False)


if __name__ == '__main__':
    # extract and pre-process affordance data
    res = extract_from_all_sources()
    for r in res:
        r.process_affordances()
    res = combine_all_tuples(res)
    filter_combined_results(res)
    res = [r for r in res if r.verify()]

    # handle the affordance mapping
    write_affordance_list(res)
    res = map_affordances(res)

    # count the final results
    aff_count = 0
    for r in res:
        aff_count += len(r.get_affordances())
    print(f"Affordance count: {aff_count}")

    # write final results
    write_affordance_list(res, False)
    write_results_to_file(res)
