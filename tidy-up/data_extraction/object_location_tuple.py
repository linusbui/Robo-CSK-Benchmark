from wordnet_filter import check_if_household_location, check_if_household_object
from ranked_location import RankedLocation


class ObjectLocationTuple:
    def __init__(self, obj: str, initial_loc: str, source: str, should_check=True):
        if should_check:
            self._is_correct = check_if_household_object(obj) & check_if_household_location(initial_loc)
        else:
            self._is_correct = True
        self._object = obj.lower().strip().replace('the', '').replace('_', ' ')
        processed_loc = initial_loc.lower().strip().replace('the', '').replace('_', ' ')
        loc = RankedLocation(processed_loc, [source])
        self._locations = [loc]

    def __str__(self):
        return f'[{str(self.verify()).upper()}] {self._object} located at/in: {self._locations}'

    def verify(self) -> bool:
        return self._is_correct

    def get_object(self) -> str:
        return self._object

    def get_locations(self) -> [RankedLocation]:
        return self._locations

    def add_location(self, location: RankedLocation):
        for loc in self._locations:
            if loc.get_location() == location.get_location():
                for source in location.get_sources():
                    loc.add_source(source)
                return

        self._locations.append(location)

    def combine_tuples(self, tup: 'ObjectLocationTuple'):
        if not tup.verify():
            return

        if tup.get_object() != self.get_object():
            return

        for loc in tup.get_locations():
            self.add_location(loc)

        tup._is_correct = False

    def to_dict(self):
        locs = []
        count_max = 5
        sorted_locs = sorted(self._locations)
        for c in range(count_max):
            if c < len(sorted_locs):
                locs.append(sorted_locs[c].get_location())
            else:
                locs.append('None')

        return {
            'Object': self.get_object(),
            'Location1': locs[0],
            'Location2': locs[1],
            'Location3': locs[2],
            'Location4': locs[3],
            'Location5': locs[4],
        }


def combine_all_tuples(tuples: [ObjectLocationTuple]) -> [ObjectLocationTuple]:
    combined = {}
    for tup in [t for t in tuples if t.verify]:
        obj = tup.get_object()
        if obj in combined:
            combined[obj].combine_tuples(tup)
        else:
            combined[obj] = tup
    return list(combined.values())
