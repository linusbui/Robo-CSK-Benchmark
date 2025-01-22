import pandas as pd

from object_location_tuple import ObjectLocationTuple, combine_all_tuples


def get_object_locations_from_cskg() -> [ObjectLocationTuple]:
    data = pd.read_csv("../data/cskg.tsv", delimiter='\t', on_bad_lines='skip')
    filtered = data[data['relation;label'] == 'at location']
    obj_loc_tuples = []

    for index, row in filtered.iterrows():
        obj = row['node1;label']
        loc = row['node2;label']
        obj_loc = ObjectLocationTuple(obj, loc, 'CSKG')
        obj_loc_tuples.append(obj_loc)

    return combine_all_tuples(obj_loc_tuples)


if __name__ == '__main__':
    res = get_object_locations_from_cskg()
    for r in res:
        print(r)
