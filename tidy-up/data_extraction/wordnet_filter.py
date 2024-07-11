#import nltk
#nltk.download('wordnet')
import wordninja
from nltk.corpus import wordnet
from nltk.corpus.reader import Synset


def get_hypernyms(synset: Synset) -> [str]:
    hypernyms_set = []
    hypernym_path = synset.hypernym_paths()
    for hyper in hypernym_path:
        for h in hyper:
            hypernyms_set.append(h.name().split('.')[0])
    return hypernyms_set


def has_overlap(list1: [str], list2: [str]) -> bool:
    set1 = set(list1)
    return any(item in set1 for item in list2)


def gather_synsets(word: str) -> [Synset]:
    synsets = wordnet.synsets(word, pos=wordnet.NOUN)
    if len(synsets) == 0:
        components = wordninja.split(word)
        return wordnet.synsets(' '.join(components), pos=wordnet.NOUN)

    return synsets


def check_if_household_object(hh_object: str) -> bool:
    household_object_synsets = ['furniture', 'food', 'appliance', 'artifact']
    synsets = gather_synsets(hh_object)
    for synset in synsets:
        res = get_hypernyms(synset)
        if has_overlap(res, household_object_synsets):
            return True
    return False


def check_if_household_location(location: str) -> bool:
    household_location_synsets = ['home', 'household', 'residence', 'home_appliance', 'housing', 'room', 'basement',
                                  'garden', 'floor', 'backyard', 'container', 'shelf', 'furniture']
    if 'room' in location:
        return True

    synsets = gather_synsets(location)
    for synset in synsets:
        res = get_hypernyms(synset)
        if has_overlap(res, household_location_synsets):
            return True
    return False
