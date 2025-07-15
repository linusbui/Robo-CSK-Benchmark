import pandas as pd
from tqdm import tqdm

from ai2thor_extractor import get_object_locations_from_ai2thor
from ascent_extractor import get_object_locations_from_ascent
from coco_extractor import get_object_locations_from_coco
from cskg_extractor import get_object_locations_from_cskg
from housekeep_extractor import get_object_locations_from_housekeep
from object_location_tuple import ObjectLocationTuple, combine_all_tuples


def extract_from_all_sources() -> [ObjectLocationTuple]:
    res_coco = get_object_locations_from_coco()
    res_ascent = get_object_locations_from_ascent()
    res_ai2thor = get_object_locations_from_ai2thor()
    res_cskg = get_object_locations_from_cskg()
    res_housekeep = get_object_locations_from_housekeep()
    return res_coco + res_ascent + res_ai2thor + res_cskg + res_housekeep


def write_results_to_file(results: [ObjectLocationTuple]):
    dict_list = [re.to_dict() for re in results]
    df = pd.DataFrame(dict_list)
    df.to_csv('../tidy_up_data.csv', index=False)


def create_and_write_reverse_dataset(results: [ObjectLocationTuple]):
    loc_dict = {}
    for res in tqdm(results, 'Creating the reverse version of the dataset'):
        for loc in res.get_locations():
            key = loc.get_location()
            if key not in loc_dict:
                loc_dict[key] = [res.get_object()]
            else:
                loc_dict[key].append(res.get_object())
    loc_dict_list = [{'Location': key, 'Objects': sorted(val)} for key, val in sorted(loc_dict.items())]
    df = pd.DataFrame(loc_dict_list)
    df.to_csv('../tidy_up_data_reversed.csv', index=False)


if __name__ == '__main__':
    res = extract_from_all_sources()
    res = [r for r in res if r.verify()]
    res = combine_all_tuples(res)
    write_results_to_file(sorted(res, key=lambda r: r.get_object()))
    create_and_write_reverse_dataset(res)
