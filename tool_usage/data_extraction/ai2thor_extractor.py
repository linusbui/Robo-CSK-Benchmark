from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

from object_affordance_tuple import ObjectAffordanceTuple, combine_all_tuples


def extract_object_data(scenes: [str], batch_size=30) -> [ObjectAffordanceTuple]:
    # Initialize an AI2-THOR controller
    controller = Controller(platform=CloudRendering)
    tuples = []

    # Iterate over scenes (batch_size = scenes per iteration)
    for i in range(0, len(scenes), batch_size):
        current_scene_batch = scenes[i:i + batch_size]

        # Iterate over scenes for each room in the current batch
        for scene in current_scene_batch:
            controller.reset(scene)
            event = controller.step(action='Initialize')
            objects = event.metadata['objects']

            for obj in objects:
                obj_name = obj['objectType']
                if obj['receptacle']:
                    tuples.append(ObjectAffordanceTuple(obj_name, 'receptacle', 'AI2Thor', False))
                if obj['toggleable']:
                    tuples.append(ObjectAffordanceTuple(obj_name, 'toggleable', 'AI2Thor', False))
                if obj['breakable']:
                    tuples.append(ObjectAffordanceTuple(obj_name, 'breakable', 'AI2Thor', False))
                if obj['canFillWithLiquid']:
                    tuples.append(ObjectAffordanceTuple(obj_name, 'canFillWithLiquid', 'AI2Thor', False))
                if obj['cookable']:
                    tuples.append(ObjectAffordanceTuple(obj_name, 'cookable', 'AI2Thor', False))
                if obj['isHeatSource']:
                    tuples.append(ObjectAffordanceTuple(obj_name, 'isHeatSource', 'AI2Thor', False))
                if obj['isColdSource']:
                    tuples.append(ObjectAffordanceTuple(obj_name, 'isColdSource', 'AI2Thor', False))
                if obj['sliceable']:
                    tuples.append(ObjectAffordanceTuple(obj_name, 'sliceable', 'AI2Thor', False))
                if obj['openable']:
                    tuples.append(ObjectAffordanceTuple(obj_name, 'openable', 'AI2Thor', False))
                if obj['pickupable']:
                    tuples.append(ObjectAffordanceTuple(obj_name, 'pickupable', 'AI2Thor', False))
                if obj['moveable']:
                    tuples.append(ObjectAffordanceTuple(obj_name, 'moveable', 'AI2Thor', False))

    # End the AI2-THOR environment
    controller.stop()
    return tuples


def get_object_affordances_from_ai2thor() -> [ObjectAffordanceTuple]:
    kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
    living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
    bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
    bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]

    kitchen_objs = extract_object_data(kitchens)
    living_room_objs = extract_object_data(living_rooms)
    bedrooms_objs = extract_object_data(bedrooms)
    bathrooms_objs = extract_object_data(bathrooms)

    return combine_all_tuples(kitchen_objs + living_room_objs + bedrooms_objs + bathrooms_objs)


if __name__ == '__main__':
    res = get_object_affordances_from_ai2thor()
    count_aff = 0
    unique_aff = set()
    for r in res:
        r.process_affordances()
        count_aff += len(r.get_affordances())
        for aff in r.get_affordances():
            unique_aff.add(aff.get_affordance())
        print(r)
    print(f'Objects: {len(res)}\nAffordances: {count_aff}\nUnique Affordances: {len(unique_aff)}')
