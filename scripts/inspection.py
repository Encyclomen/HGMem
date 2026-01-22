import asyncio
import copy
import json
import math
import os
PROJ_DIR = os.path.dirname(os.path.dirname(__file__))

import numpy as np

from transformers import AutoModel, AutoTokenizer
from scripts.utils import storage_class, load_text_chunks, load_graph, combine_consecutive_overlapping_chunks


def inspect_raised_questions(raised_questions_path, filtered_raised_questions_path):
    with open(raised_questions_path, "r", encoding="utf-8") as rjf:#, open(filtered_raised_questions_path, "w", encoding="utf-8") as wjf:
        statistics_1 = {}
        statistics_2 = {}
        statistics_3 = {}
        all_items = []
        for line in rjf:
            item = json.loads(line.strip())
            #new_item = copy.deepcopy(item)
            all_items.append(item)
            question = item["question"]
            answer = item["reference_answer"]
            involved_entities = item["involved_entities"]

            involved_chunks = item["origin_chunks"]

            if len(involved_entities) not in statistics_1:
                statistics_1[len(involved_entities)] = 1
            else:
                statistics_1[len(involved_entities)] += 1

            if len(involved_chunks) not in statistics_3:
                statistics_3[len(involved_chunks)] = 1
            else:
                statistics_3[len(involved_chunks)] += 1

    print(statistics_1)
    print(statistics_2)
    print(statistics_3)


def inspect_multiturn_reasoning(multiturn_reasoning_path):
    num_total = 0
    num_turn_gt_1 = 0
    with open(multiturn_reasoning_path, "r", encoding="utf-8") as rjf:
        for line in rjf:
            item = json.loads(line.strip())
            question = item["question"]
            ref_answer = item["reference_answer"]
            involved_entities = item["involved_entities"]
            origin_chunks = item["origin_chunks"]

            multiturn_selected_entities = item["multiturn_selected_entities"]
            multiturn_retrieved_infos = item["multiturn_retrieved_infos"]
            multiturn_hints = item["multiturn_hints"]
            multiturn_reasonings = item["multiturn_reasonings"]
            multiturn_queries = item["multiturn_queries"]
            final_response = item["final_response"]


            num_total += 1
            if len(multiturn_queries) > 1:
                num_turn_gt_1 += 1
            print(len(multiturn_queries))
        a = 1


async def inspect_entity_distribution_across_chunks(graph, text_chunk_storage):
    history = {}

    all_entities = await graph.list_all_nodes()
    for entity_name in all_entities:
        node = await graph.get_node(entity_name)
        chunk_ids = node["source_id"].split("<SEP>")
        for c_id in chunk_ids:
            if c_id not in history:
                history[c_id] = [entity_name]
            else:
                history[c_id].append(entity_name)
    print(max([len(e_list) for e_list in history.values()]))
    print(min([len(e_list) for e_list in history.values()]))

    all_chunks = []
    all_chunks_ids = await text_chunk_storage.all_keys()
    for c_id in all_chunks_ids:
        chunk = await text_chunk_storage.get_by_id(c_id)
        # chunk_order_index = chunk["chunk_order_index"]
        all_chunks.append(chunk['content'])
    assert len(all_chunks) == len(all_chunks_ids)

    statistics = []
    step_size = 10
    for i in range(0, len(all_chunks_ids), step_size):
        consecutive_chunks_ids = all_chunks_ids[i:i+step_size]
        entities_within_interval = []
        for chunk_id in consecutive_chunks_ids:
            entities_within_interval.extend(history[chunk_id])
        unique_entities_within_interval = set(entities_within_interval)
        statistics.append(len(unique_entities_within_interval))

    return history, statistics


def inspect_entity_distribution_across_questions(created_data_dir):
    statistics = []
    raised_questions_path = os.path.join(created_data_dir, "raised_questions.jsonl")
    with open(raised_questions_path, "r", encoding="utf-8") as rjf:
        for line in rjf:
            item = json.loads(line.strip())
            question = item["question"]
            ref_answer = item["reference_answer"]
            gathered_relevant_entities = item["involved_entities"]
            origin_chunks = item["origin_chunks"]
            statistics.append(len(gathered_relevant_entities))

    print(max(statistics))
    print(min(statistics))

    return statistics


if __name__ == "__main__":
    model_name_or_path = os.path.join(PROJ_DIR, 'cache/bge-m3')

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    embed_hidden_size = 1024

    domain = "legal"
    for i in range(0, 2):
        target_dir = os.path.join(PROJ_DIR, f"data/created_data/ultradomain/{domain}/{i}")
        raised_questions_path = os.path.join(target_dir, "raised_questions.jsonl")
        filtered_raised_questions_path = os.path.join(target_dir, "raised_questions_new.jsonl")
        inspect_raised_questions(raised_questions_path, filtered_raised_questions_path)


    print("Finish")
