from pattern.text.en import singularize

from extracted_affordance import ExtractedAffordance
from utils import check_if_household_object


class ObjectAffordanceTuple:
    def __init__(self, obj: str, initial_aff: str, source: str, should_check=True, rocs_trust=-1.0):
        proc_obj = preprocess_string(obj)
        if should_check:
            self._is_correct = check_if_household_object(proc_obj)
        else:
            self._is_correct = True
        self._object = proc_obj
        proc_aff = preprocess_string(initial_aff)
        self._affordances = [ExtractedAffordance(proc_aff, [source], rocs_trust)]

    def __str__(self):
        return f'[{str(self.verify()).upper()}] {self._object} affords: {[str(aff) for aff in self._affordances]}'

    def verify(self) -> bool:
        return self._is_correct

    def get_object(self) -> str:
        return self._object

    def get_affordances(self) -> [ExtractedAffordance]:
        return self._affordances

    def add_affordance(self, affordance: ExtractedAffordance):
        for aff in self._affordances:
            if aff.get_affordance() == affordance.get_affordance():
                for source in affordance.get_sources():
                    if source == 'RoCS':
                        aff.add_source(source, aff.get_rocs_trust())
                    else:
                        aff.add_source(source)
                return

        self._affordances.append(affordance)

    def combine_tuples(self, tup: 'ObjectAffordanceTuple'):
        if not tup.verify():
            return

        if tup.get_object() != self.get_object():
            return

        for aff in tup.get_affordances():
            self.add_affordance(aff)

        tup._is_correct = False

    def to_dict(self):
        affs = {}
        rank = 1
        sorted_affs = sorted(self._affordances)
        for aff in sorted_affs:
            affs[rank] = (aff.get_affordance(), aff.get_trust())
            rank += 1

        return {
            'Object': self.get_object(),
            'Affordances': affs
        }

    def process_affordances(self):
        valid_affordances = False
        for aff in self._affordances:
            aff.process_affordance()
            valid_affordances = aff.get_affordance() != 'None'
        self._is_correct = valid_affordances
        for a in self._affordances :
            if a.get_affordance() == "None":
                self._affordances.remove(a)


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
