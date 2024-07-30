import spacy


class ExtractedAffordance:
    trust_mapping = {'Visual Dataset': 1.0, 'CSKG': 0.5, 'Narrative Objects': 0.75, 'COAT': 1.0}
    nlp = spacy.load("en_core_web_trf")

    def __init__(self, unprocessed: str, sources: [str], rocs_trust=-1.0):
        self._unprocessed = unprocessed
        self._affordance = unprocessed
        self._sources = sources
        self._rocs_trust = rocs_trust
        self._trust = self.calculate_trust()

    def __str__(self):
        return f'[Trust: {self._trust}] {self._affordance} (Orig: {self._unprocessed}) from: {self._sources}'

    def get_affordance(self) -> str:
        return self._affordance

    def get_trust(self) -> float:
        return self._trust

    def get_rocs_trust(self) -> float:
        return self._rocs_trust

    def get_sources(self) -> [str]:
        return self._sources

    def rename_affordance(self, new_name: str):
        self._affordance = new_name

    def add_source(self, source: str, rocs_trust=-1.0):
        if source not in self._sources:
            self._sources.append(source)
            self._rocs_trust = rocs_trust
            self._trust = self.calculate_trust()

    def calculate_trust(self) -> float:
        trust = 0
        for s in self._sources:
            if s == 'RoCS':
                trust += self._rocs_trust
            else:
                trust += self.trust_mapping[s]
        return trust

    def process_affordance(self, use_none=False):
        doc = ExtractedAffordance.nlp(self._unprocessed)
        for token in doc:
            if token.pos_ == "VERB":
                base_form = token.lemma_
                self._affordance = base_form
                return
        if use_none:
            self._affordance = "None"

    def _compare(self, other: 'ExtractedAffordance') -> float:
        return self._trust - other.get_trust()

    def __lt__(self, other: 'ExtractedAffordance') -> bool:
        return self._compare(other) > 0

    def __le__(self, other: 'ExtractedAffordance') -> bool:
        return self._compare(other) >= 0

    def __eq__(self, other: 'ExtractedAffordance') -> bool:
        return (self.get_affordance() == other.get_affordance()) & (self.get_sources() == other.get_sources())

    def __ne__(self, other: 'ExtractedAffordance') -> bool:
        return (self.get_affordance() != other.get_affordance()) | (self.get_sources() != other.get_sources())

    def __ge__(self, other: 'ExtractedAffordance') -> bool:
        return self._compare(other) <= 0

    def __gt__(self, other: 'ExtractedAffordance') -> bool:
        return self._compare(other) < 0
