class ObjectLocationTuple:
    def __init__(self, obj: str, source: str):
        self._object = obj
        self._source = source
        self._locations = []
        self._is_active = True

    def __str__(self):
        return f'{self._object} located at/in: {self._locations} (Source: {self._source})'

    def get_object(self) -> str:
        return self._object

    def get_locations(self) -> [str]:
        return self._locations

    def get_source(self) -> str:
        return self._source

    def add_location(self, location):
        if location not in self._locations:
            self._locations.append(location)

    def combine_tuples(self, tup: 'ObjectLocationTuple'):
        if tup.get_object() != self.get_object():
            return

        for loc in tup.get_locations():
            self.add_location(loc)

        tup._is_active = False
