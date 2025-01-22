import numpy as np

from object_location_tuple import ObjectLocationTuple, combine_all_tuples


def get_object_locations_from_housekeep() -> [ObjectLocationTuple]:
    dict = np.load('../data/housekeep.npy', allow_pickle=True).item()
    objects = dict['objects']
    rooms = dict['rooms']
    #room_receps = dict['room_receptacles']
    data = dict['data']
    obj_loc_tuples = []

    for index, row in data.iterrows():
        obj = objects[row['object_idx']]
        room = rooms[row['room_idx']]
        #correct_receps = [room_receps[r].split('|')[1] for r in row['correct']]
        obj_loc = ObjectLocationTuple(obj, room, 'Housekeep', False)
        obj_loc_tuples.append(obj_loc)

    return combine_all_tuples(obj_loc_tuples)


if __name__ == '__main__':
    res = get_object_locations_from_housekeep()
    for r in res:
        print(r)
