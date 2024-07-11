import ai2thor.controller

from object_location_tuple import ObjectLocationTuple, combine_all_tuples


def generate_object_data(scenes: [str], room_name: str, batch_size=30) -> [ObjectLocationTuple]:
    # Initialize an AI2-THOR controller
    controller = ai2thor.controller.Controller()
    obj_loc_tuples = []

    # Iterate over scenes (batch_size = scenes per iteration)
    for i in range(0, len(scenes), batch_size):
        current_scene_batch = scenes[i:i + batch_size]

        # Iterate over scenes for each room in the current batch
        for scene in current_scene_batch:
            controller.reset(scene)
            event = controller.step(action='Initialize')
            objects = event.metadata['objects']

            for obj in objects:
                obj_loc = ObjectLocationTuple(obj['objectType'].lower(), 'AI2Thor')
                obj_loc.add_location(room_name)
                obj_loc_tuples.append(obj_loc)

    # End the AI2-THOR environment
    controller.stop()
    return obj_loc_tuples


def get_object_locations_from_ai2thor() -> [ObjectLocationTuple]:
    kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
    living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
    bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
    bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]

    kitchen_objs = generate_object_data(kitchens, 'kitchen')
    living_room_objs = generate_object_data(living_rooms, 'living_room')
    bedrooms_objs = generate_object_data(bedrooms, 'bedroom')
    bathrooms_objs = generate_object_data(bathrooms, 'bathroom')

    return combine_all_tuples(kitchen_objs + living_room_objs + bedrooms_objs + bathrooms_objs)


if __name__ == '__main__':
    res = get_object_locations_from_ai2thor()
    for r in res:
        print(r)
