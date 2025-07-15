from typing import Dict, List

from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from task_capability import TaskCapability


def combine_capabilities(caps: List[TaskCapability]) -> List[TaskCapability]:
    # combine all capabilities with the exact same text
    unique_caps = combine_same_tasks(caps)
    unique_sentences = list(unique_caps.keys())

    # calculate their embeddings & clustering
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedded_caps = model.encode(unique_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
    clusters = util.community_detection(embedded_caps, min_community_size=1, threshold=0.75)
    print(f"Clustering finished. Found {len(clusters)} clusters:")
    for i, c in enumerate(clusters):
        if len(c) > 1:
            print(f"\nCluster {i + 1}, #{len(c)} Elements ")
            for sentence_id in c:
                print("\t", unique_sentences[sentence_id])

    # combine all capabilities in one cluster
    cluster_reps = []
    for cluster in tqdm(clusters, 'Combining clusters'):
        if len(cluster) == 1:
            continue
        cluster_rep = unique_caps[unique_sentences[cluster[0]]]
        for cap in cluster[1:]:
            cluster_rep.combine_task(unique_caps[unique_sentences[cap]])
        cluster_reps.append(cluster_rep)
    return cluster_reps


def combine_same_tasks(caps: List[TaskCapability]) -> Dict[str, TaskCapability]:
    unique = {}
    for cap in caps:
        if cap.get_task() == "no action":
            continue

        if cap.get_task() not in unique:
            unique[cap.get_task()] = cap
        else:
            unique[cap.get_task()].combine_task(cap)
    return unique
