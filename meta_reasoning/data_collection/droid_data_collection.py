# Prints all high-level tasks contained in the DROID dataset to the console.
# Expects the extracted dataset to be found in the "../data/DROID/" folder.

import random as rndm

import tensorflow_datasets as tfds
from tqdm import tqdm

if __name__ == '__main__':
    builder = tfds.builder_from_directory('../data/DROID/')
    ds = builder.as_dataset(split='train')
    counter = 0

    print('\nEpisodeNum,Task')
    for episode in tqdm(ds, 'Collecting data from DROID'):
        try:
            # choose one step at random since all steps in one episode have the same high-level goal
            rndm_step = rndm.choice(list(episode["steps"]))
            task = rndm_step["language_instruction"].numpy().decode('utf-8')
            task_proc = task.lower().replace(".", "").replace("\n", " ").strip()
            if task_proc != "":
                print(f'{counter},\"{task_proc}\"')
                counter += 1

        except Exception as ex:
            continue

