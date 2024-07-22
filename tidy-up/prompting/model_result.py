class ModelResultTuple:
    def __init__(self, obj: str, locations: [str]):
        self._object = obj
        self._locations = []
        self._locations.extend(locations)

    def __str__(self):
        return f'{self._object} located at/in: {self._locations}'

    def get_object(self) -> str:
        return self._object

    def get_locations(self) -> [str]:
        return self._locations
