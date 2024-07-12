import json

from object_location_tuple import ObjectLocationTuple, combine_all_tuples


def get_object_locations_from_coco() -> [ObjectLocationTuple]:
    with open("../data/coco_instances_val2017.json") as f:
        data = json.load(f)

    obj_loc_tuples = []
    for e in data['categories']:
        loc = e['supercategory']
        if loc == 'kitchen':
            obj = e['name']
            obj_loc = ObjectLocationTuple(obj, loc, 'Microsoft COCO')
            obj_loc_tuples.append(obj_loc)
    return combine_all_tuples(obj_loc_tuples)


if __name__ == '__main__':
    res = get_object_locations_from_coco()
    for r in res:
        print(r)
