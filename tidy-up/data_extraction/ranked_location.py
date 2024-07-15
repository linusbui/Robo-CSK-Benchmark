class RankedLocation:
    trust_mapping = {'AI2Thor': 1.0, 'Ascent++': 0.5, 'Microsoft COCO': 0.75, 'CSKG': 0.5, 'Housekeep': 1.0}

    def __init__(self, location: str, sources: [str]):
        self._location = location
        self._sources = sources
        self._trust = self.calculate_trust()
        self._rank = -1

    def get_location(self) -> str:
        return self._location

    def get_trust(self) -> float:
        return self._trust

    def get_rank(self) -> int:
        return self._rank

    def get_sources(self) -> [str]:
        return self._sources

    def add_source(self, source: str):
        if source not in self._sources:
            self._sources.append(source)
            self._trust = self.calculate_trust()

    def calculate_trust(self) -> float:
        trust = 0
        for s in self._sources:
            trust += self.trust_mapping[s]
        return trust

    def change_rank(self, rank: int):
        self._rank = rank

    def _cmp(self, other: 'RankedLocation') -> float:
        return self._trust - other.get_trust()

    def __lt__(self, other: 'RankedLocation') -> bool:
        return self._cmp(other) > 0

    def __le__(self, other: 'RankedLocation') -> bool:
        return self._cmp(other) >= 0

    def __eq__(self, other: 'RankedLocation') -> bool:
        return self._cmp(other) == 0

    def __ne__(self, other: 'RankedLocation') -> bool:
        return self._cmp(other) != 0

    def __ge__(self, other: 'RankedLocation') -> bool:
        return self._cmp(other) <= 0

    def __gt__(self, other: 'RankedLocation') -> bool:
        return self._cmp(other) < 0
