class ExtractedAffordance:
    trust_mapping = {'Visual Dataset': 1.0, 'CSKG': 0.5}

    def __init__(self, affordance: str, sources: [str]):
        self._affordance = affordance
        self._sources = sources
        self._trust = self.calculate_trust()

    def __str__(self):
        return f'[Trust: {self._trust}] {self._affordance} from: {self._sources}'

    def get_affordance(self) -> str:
        return self._affordance

    def get_trust(self) -> float:
        return self._trust

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

    def _cmp(self, other: 'ExtractedAffordance') -> float:
        return self._trust - other.get_trust()

    def __lt__(self, other: 'ExtractedAffordance') -> bool:
        return self._cmp(other) > 0

    def __le__(self, other: 'ExtractedAffordance') -> bool:
        return self._cmp(other) >= 0

    def __eq__(self, other: 'ExtractedAffordance') -> bool:
        return self._cmp(other) == 0

    def __ne__(self, other: 'ExtractedAffordance') -> bool:
        return self._cmp(other) != 0

    def __ge__(self, other: 'ExtractedAffordance') -> bool:
        return self._cmp(other) <= 0

    def __gt__(self, other: 'ExtractedAffordance') -> bool:
        return self._cmp(other) < 0
