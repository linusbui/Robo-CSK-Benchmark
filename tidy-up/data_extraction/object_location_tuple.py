from wordnet_filter import check_if_household_location, check_if_household_object


class ObjectLocationTuple:
    def __init__(self, obj: str, initial_loc: str, source: str):
        self._is_correct = check_if_household_object(obj) & check_if_household_location(initial_loc)
        self._object = obj
        self._source = source
        self._locations = [initial_loc]

    def __str__(self):
        return f'[{str(self.verify()).upper()}] {self._object} located at/in: {self._locations} (Source: {self._source})'

    def verify(self) -> bool:
        return self._is_correct

    def get_object(self) -> str:
        return self._object

    def get_locations(self) -> [str]:
        return self._locations

    def get_source(self) -> str:
        return self._source

    def add_location(self, location: str):
        if location not in self._locations and check_if_household_location(location):
            self._locations.append(location)

    def combine_tuples(self, tup: 'ObjectLocationTuple'):
        if not tup.verify():
            return

        if tup.get_object() != self.get_object():
            return

        for loc in tup.get_locations():
            self.add_location(loc)

        tup._is_correct = False


def combine_all_tuples(tuples: [ObjectLocationTuple]) -> [ObjectLocationTuple]:
    combined = {}
    for tup in [t for t in tuples if t.verify]:
        obj = tup.get_object()
        if obj in combined:
            combined[obj].combine_tuples(tup)
        else:
            combined[obj] = tup
    return list(combined.values())
