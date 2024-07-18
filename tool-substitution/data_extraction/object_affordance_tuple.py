from pattern.text.en import singularize


class ObjectAffordanceTuple:
    def __init__(self, obj: str):
        self._object = preprocess_string(obj)
        self._affordances = []
        self._is_correct = True

    def __str__(self):
        return f'[{str(self.verify()).upper()}] {self._object} affords: {self._affordances}'

    def verify(self) -> bool:
        return self._is_correct

    def get_object(self) -> str:
        return self._object

    def get_affordances(self) -> [(str, float)]:
        return self._affordances

    def add_affordance(self, affordance: str, trust=-1.0, source='None'):
        if source == 'None':
            to_add = trust
        else:
            to_add = get_trust_for_source(source)

        for aff in self._affordances:
            if aff[0] == affordance:
                aff[1] += to_add
                return

        self._affordances.append((affordance, to_add))

    def combine_tuples(self, tup: 'ObjectAffordanceTuple'):
        if not tup.verify():
            return

        if tup.get_object() != self.get_object():
            return

        for aff in tup.get_affordances():
            self.add_affordance(aff[0], aff[1])

        tup._is_correct = False

    def to_dict(self):
        affs = {}
        rank = 1
        sorted_affs = sorted(self._affordances, key=lambda a: a[1])
        for aff in sorted_affs:
            affs[rank] = aff
            rank += 1

        return {
            'Object': self.get_object(),
            'Affordances': affs
        }


def preprocess_string(word: str) -> str:
    processed = word.lower().replace('_', ' ').strip()
    return singularize(processed)


def combine_all_tuples(tuples: [ObjectAffordanceTuple]) -> [ObjectAffordanceTuple]:
    combined = {}
    for tup in [t for t in tuples if t.verify]:
        obj = tup.get_object()
        if obj in combined:
            combined[obj].combine_tuples(tup)
        else:
            combined[obj] = tup
    return list(combined.values())


trust_mapping = {'Visual Dataset': 1.0}


def get_trust_for_source(source: str) -> float:
    return trust_mapping[source]
