import asyncio
import copy
import itertools
import json
import math
import os
import random
import sys
from datetime import datetime
from queue import Queue

PROJ_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJ_DIR)

from myrag.utils import compute_args_hash, encode_string_by_tiktoken


from scripts.utils import storage_class, load_text_chunks, load_graph, load_entity_vdb, load_relationship_vdb, \
    combine_consecutive_overlapping_chunks, load_llm_response_cache


def collect_all_difficult_questions(target_dir):
    eval_path = os.path.join(target_dir, f"responses_to_raised_questions_{source_type}_{rag_mode}_{LLM_MODEL_NAME}_{graph_mode}_eval.jsonl")
    out_path = os.path.join(target_dir, f"difficult_questions_{source_type}.jsonl")
    with open(eval_path, "r", encoding="utf-8") as rjf, open(out_path, "w", encoding="utf-8") as wjf:
        difficult_questions = []
        for line in rjf:
            item = json.loads(line.strip())
            question = item["question"]
            ref_answer = item["reference_answer"]
            involved_entities = item["involved_entities"] if "involved_entities" in item else None
            origin_full_doc_id = item["origin_full_doc_id"]
            origin_chunks_ids = item["origin_chunks_ids"]
            origin_chunks_order_index = item["origin_chunks_order_index"]
            origin_chunks = item["origin_chunks"]
            paragraph = combine_consecutive_overlapping_chunks(origin_chunks)
            evaluation_results = item["evaluation_results"]
            evaluation_results_naive_rag = evaluation_results["naive_rag"]
            if "comprehensiveness" in evaluation_results_naive_rag:
                comprehensiveness = evaluation_results_naive_rag["comprehensiveness"]
                level_comprehensiveness, score_comprehensiveness = comprehensiveness[0], comprehensiveness[0]
            if "diversity" in evaluation_results_naive_rag:
                diversity = evaluation_results_naive_rag["diversity"]
                level_diversity, score_diversity = diversity[0], diversity[0]

            new_item = {"question": question, "reference_answer": ref_answer}
            if level_comprehensiveness <= 3:
                difficult_questions.append(question)
                wjf.write(json.dumps(new_item, ensure_ascii=False) + "\n")
        a=1
        all_difficult_questions.append(difficult_questions)


def sample_difficult_questions():
    all_difficult_questions_dicts = []
    for i in range(0, len(valid_subdirs)):
        target_dir = os.path.join(PROJ_DIR, f"data/created_data/{DATASET_NAME}/{domain}/{i}")
        all_difficult_questions_dict = fetch_all_difficult_questions_dict(target_dir)
        all_difficult_questions_dicts.append([qa_pair for q_idx, qa_pair in enumerate(all_difficult_questions_dict.items())])

    total_num_questions_to_sample = 100
    avg_num_questions = total_num_questions_to_sample // len(valid_subdirs)
    list_num_questions_to_sample = [avg_num_questions for _ in range(0, len(valid_subdirs))]
    for j in range(0, total_num_questions_to_sample-avg_num_questions*len(valid_subdirs)):
        list_num_questions_to_sample[j] += 1

    for i in range(0, len(valid_subdirs)):
        sampled_questions_indices = sorted(random.sample(list(range(len(all_difficult_questions_dicts[i]))), min(max(list_num_questions_to_sample[i], 1), len(all_difficult_questions_dicts[i]))))
        sampled_questions_dict = [{"question": all_difficult_questions_dicts[i][q_idx][0], "reference_answer": all_difficult_questions_dicts[i][q_idx][1]} for q_idx in sampled_questions_indices]
        target_dir = os.path.join(PROJ_DIR, f"data/created_data/{DATASET_NAME}/{domain}/{i}")
        out_path = os.path.join(target_dir, f"sampled_difficult_questions_{source_type}.jsonl")
        with open(out_path, "w", encoding="utf-8") as wjf:
            for qa_dict in sampled_questions_dict:
                wjf.write(json.dumps(qa_dict, ensure_ascii=False) + "\n")


def collect_all_simple_questions(target_dir):
    all_difficult_questions_dict = fetch_all_difficult_questions_dict(target_dir)

    eval_path = os.path.join(target_dir, f"responses_to_raised_questions_{source_type}_{rag_mode}_{LLM_MODEL_NAME}_{graph_mode}_eval.jsonl")
    out_path = os.path.join(target_dir, f"simple_questions_{source_type}.jsonl")
    with (open(eval_path, "r", encoding="utf-8") as rjf, open(out_path, "w", encoding="utf-8") as wjf):
        simple_questions_level_4 = []
        simple_questions_level_5 = []
        for line in rjf:
            item = json.loads(line.strip())
            question = item["question"]
            if question in all_difficult_questions_dict:
                continue
            ref_answer = item["reference_answer"]
            involved_entities = item["involved_entities"] if "involved_entities" in item else None
            origin_full_doc_id = item["origin_full_doc_id"]
            origin_chunks_ids = item["origin_chunks_ids"]
            origin_chunks_order_index = item["origin_chunks_order_index"]
            origin_chunks = item["origin_chunks"]
            paragraph = combine_consecutive_overlapping_chunks(origin_chunks)
            evaluation_results = item["evaluation_results"]
            evaluation_results_naive_rag = evaluation_results["naive_rag"]
            if "comprehensiveness" in evaluation_results_naive_rag:
                comprehensiveness = evaluation_results_naive_rag["comprehensiveness"]
                level_comprehensiveness, score_comprehensiveness = comprehensiveness[0], comprehensiveness[0]
            if "diversity" in evaluation_results_naive_rag:
                diversity = evaluation_results_naive_rag["diversity"]
                level_diversity, score_diversity = diversity[0], diversity[0]

            new_item = {"question": question, "reference_answer": ref_answer}
            if level_comprehensiveness == 4:
                simple_questions_level_4.append(new_item)
            if level_comprehensiveness == 5:
                simple_questions_level_5.append(new_item)
        num_difficult_questions = len(all_difficult_questions_dict)
        num_sample = min(len(simple_questions_level_4), max(num_difficult_questions, 1))
        sampled_simple_questions_level_4 = random.sample(simple_questions_level_4, num_sample)
        for item in sampled_simple_questions_level_4:
            wjf.write(json.dumps(item, ensure_ascii=False) + "\n")
        num_sample = min(len(simple_questions_level_5), max(num_difficult_questions, 1))
        sampled_simple_questions_level_5 = random.sample(simple_questions_level_5, num_sample)
        for item in sampled_simple_questions_level_5:
            wjf.write(json.dumps(item, ensure_ascii=False) + "\n")


def sample_simple_questions():
    all_simple_questions_dicts = []
    for i in range(0, len(valid_subdirs)):
        target_dir = os.path.join(PROJ_DIR, f"data/created_data/{DATASET_NAME}/{domain}/{i}")
        all_simple_questions_dict = fetch_all_simple_questions_dict(target_dir)
        all_simple_questions_dicts.append([qa_pair for q_idx, qa_pair in enumerate(all_simple_questions_dict.items())])

    total_num_questions_to_sample = 100
    avg_num_questions = total_num_questions_to_sample // len(valid_subdirs)
    list_num_questions_to_sample = [avg_num_questions for _ in range(0, len(valid_subdirs))]
    for j in range(0, total_num_questions_to_sample-avg_num_questions*len(valid_subdirs)):
        list_num_questions_to_sample[j] += 1

    for i in range(0, len(valid_subdirs)):
        sampled_questions_indices = sorted(random.sample(list(range(len(all_simple_questions_dicts[i]))), min(max(list_num_questions_to_sample[i], 1), len(all_simple_questions_dicts[i]))))
        sampled_questions_dict = [{"question": all_simple_questions_dicts[i][q_idx][0], "reference_answer": all_simple_questions_dicts[i][q_idx][1]} for q_idx in sampled_questions_indices]
        target_dir = os.path.join(PROJ_DIR, f"data/created_data/{DATASET_NAME}/{domain}/{i}")
        out_path = os.path.join(target_dir, f"sampled_simple_questions_{source_type}.jsonl")
        with open(out_path, "w", encoding="utf-8") as wjf:
            for qa_dict in sampled_questions_dict:
                wjf.write(json.dumps(qa_dict, ensure_ascii=False) + "\n")


def fetch_all_difficult_questions_dict(target_dir):
    all_difficult_questions_path = os.path.join(target_dir, f"all_difficult_questions_{source_type}.jsonl")
    print(f"All difficult questions path: {all_difficult_questions_path}")
    with open(all_difficult_questions_path, "r", encoding="utf-8") as rjf:
        difficult_questions_dict = {}
        difficult_questions = []
        for line in rjf:
            item = json.loads(line.strip())
            question = item["question"]
            ref_answer = item["reference_answer"]
            difficult_questions_dict[question] = ref_answer
            difficult_questions.append(question)
        #all_difficult_questions.append(difficult_questions)

    return difficult_questions_dict


def fetch_sampled_difficult_questions_dict(target_dir):
    sampled_difficult_questions_path = os.path.join(target_dir, f"sampled_difficult_questions_{source_type}.jsonl")
    print(f"Sampled difficult questions path: {sampled_difficult_questions_path}")
    with open(sampled_difficult_questions_path, "r", encoding="utf-8") as rjf:
        difficult_questions_dict = {}
        difficult_questions = []
        for line in rjf:
            item = json.loads(line.strip())
            question = item["question"]
            ref_answer = item["reference_answer"]
            difficult_questions_dict[question] = ref_answer
            difficult_questions.append(question)
        #all_difficult_questions.append(difficult_questions)

    return difficult_questions_dict


def fetch_all_simple_questions_dict(target_dir):
    all_simple_questions_path = os.path.join(target_dir, f"all_simple_questions_{source_type}.jsonl")
    print(f"All simple questions path: {all_simple_questions_path}")
    with open(all_simple_questions_path, "r", encoding="utf-8") as rjf:
        simple_questions_dict = {}
        simple_questions = []
        for line in rjf:
            item = json.loads(line.strip())
            question = item["question"]
            ref_answer = item["reference_answer"]
            simple_questions_dict[question] = ref_answer
            simple_questions.append(question)
        #all_difficult_questions.append(simple_questions)

    return simple_questions_dict


def fetch_sampled_simple_questions_dict(target_dir):
    sampled_simple_questions_path = os.path.join(target_dir, f"sampled_simple_questions_{source_type}.jsonl")
    print(f"Sampled simple questions path: {sampled_simple_questions_path}")
    with open(sampled_simple_questions_path, "r", encoding="utf-8") as rjf:
        simple_questions_dict = {}
        simple_questions = []
        for line in rjf:
            item = json.loads(line.strip())
            question = item["question"]
            ref_answer = item["reference_answer"]
            simple_questions_dict[question] = ref_answer
            simple_questions.append(question)
        #all_difficult_questions.append(simple_questions)

    return simple_questions_dict


def inspect_difficult_questions_from_full(target_dir):
    all_difficult_questions_dict = fetch_all_difficult_questions_dict(target_dir)

    questions = []
    ref_answers = []
    rag_responses = []
    raised_questions_eval_path = os.path.join(target_dir, f"responses_to_raised_questions_{source_type}_{rag_mode}_{LLM_MODEL_NAME}_{graph_mode}_eval.jsonl")
    print(f"Evaluation of difficult questions from {raised_questions_eval_path}")
    with open(raised_questions_eval_path, "r", encoding="utf-8") as rjf:
        for line in rjf:
            item = json.loads(line.strip())
            question = item["question"]
            if question not in all_difficult_questions_dict:
                continue
            ref_answer = item["reference_answer"]
            questions.append(question)
            ref_answers.append(ref_answer)
            involved_entities = item["involved_entities"] if "involved_entities" in item else None
            origin_full_doc_id = item["origin_full_doc_id"] if "origin_full_doc_id" in item else None
            origin_chunks_ids = item["origin_chunks_ids"] if "origin_chunks_ids" in item else None
            origin_chunks_order_index = item["origin_chunks_order_index"] if "origin_chunks_order_index" in item else None
            origin_chunks = item["origin_chunks"] if "origin_chunks" in item else None
            if origin_chunks is not None:
                paragraph = combine_consecutive_overlapping_chunks(origin_chunks)

            if rag_mode == "naive":
                response = item["naive_rag_response"]
                rag_responses.append(response)
                evaluation_results = item["evaluation_results"]["naive_rag"]
            elif rag_mode == "light-direct":
                response = item["light-direct_rag_response"]
                rag_responses.append(response)
                evaluation_results = item["evaluation_results"]["light-direct_rag"]
            elif rag_mode == "memory-v1":
                response = item["memory-v1_rag_response"]
                #response = item["debug_rag_response"]
                rag_responses.append(response)
                evaluation_results = item["evaluation_results"]["memory-v1_rag"]
                #evaluation_results = item["evaluation_results"]["debug_rag"]
            elif rag_mode == "debug":
                response = item["debug_rag_response"]
                rag_responses.append(response)
                evaluation_results = item["evaluation_results"]["debug_rag"]
            else:
                raise ValueError(f"Unknown rag_mode: {rag_mode}")

            if "prompt" in evaluation_results:
                prompt = evaluation_results["prompt"]
                eval_statistics["prompt"].append(prompt)

            if "comprehensiveness" in evaluation_results:
                comprehensiveness = evaluation_results["comprehensiveness"]
                if "comprehensiveness" in eval_statistics:
                    eval_statistics["comprehensiveness"]["level"].append(comprehensiveness[0])
                    eval_statistics["comprehensiveness"]["score"].append(comprehensiveness[1])
                    eval_statistics["comprehensiveness"]["explanation"].append(comprehensiveness[2])
            if "diversity" in evaluation_results:
                diversity = evaluation_results["diversity"]
                if "diversity" in eval_statistics:
                    eval_statistics["diversity"]["level"].append(diversity[0])
                    eval_statistics["diversity"]["score"].append(diversity[1])
                    eval_statistics["diversity"]["explanation"].append(diversity[2])
            if "relevance" in evaluation_results:
                relevance = evaluation_results["relevance"]
                if "relevance" in eval_statistics:
                    eval_statistics["relevance"]["level"].append(relevance[0])
                    eval_statistics["relevance"]["score"].append(relevance[1])
                    eval_statistics["relevance"]["explanation"].append(relevance[2])

        all_eval_statistics.append(eval_statistics)
        a=1


def inspect_sampled_difficult_questions_from_all(target_dir):
    #all_difficult_questions_dict = fetch_all_difficult_questions_dict(target_dir)
    sampled_difficult_questions_dict = fetch_sampled_difficult_questions_dict(target_dir)

    questions = []
    ref_answers = []
    rag_responses = []
    #all_difficult_questions_eval_path = os.path.join(target_dir, f"responses_to_raised_questions_{source_type}_{rag_mode}_{LLM_MODEL_NAME}_{graph_mode}_eval.jsonl")
    all_difficult_questions_eval_path = os.path.join(target_dir, f"responses_to_all_difficult_questions_{source_type}_{rag_mode}_{LLM_MODEL_NAME}_{graph_mode}_eval.jsonl")
    print(f"Evaluation of all difficult questions from {all_difficult_questions_eval_path}")
    with open(all_difficult_questions_eval_path, "r", encoding="utf-8") as rjf:
        for line in rjf:
            item = json.loads(line.strip())
            question = item["question"]
            if question not in sampled_difficult_questions_dict:
                continue
            ref_answer = item["reference_answer"]
            questions.append(question)
            ref_answers.append(ref_answer)
            involved_entities = item["involved_entities"] if "involved_entities" in item else None
            origin_full_doc_id = item["origin_full_doc_id"] if "origin_full_doc_id" in item else None
            origin_chunks_ids = item["origin_chunks_ids"] if "origin_chunks_ids" in item else None
            origin_chunks_order_index = item["origin_chunks_order_index"] if "origin_chunks_order_index" in item else None
            origin_chunks = item["origin_chunks"] if "origin_chunks" in item else None
            if origin_chunks is not None:
                paragraph = combine_consecutive_overlapping_chunks(origin_chunks)

            if rag_mode == "naive":
                response = item["naive_rag_response"]
                rag_responses.append(response)
                evaluation_results = item["evaluation_results"]["naive_rag"]
            elif rag_mode == "light-direct":
                response = item["light-direct_rag_response"]
                rag_responses.append(response)
                evaluation_results = item["evaluation_results"]["light-direct_rag"]
            elif rag_mode == "memory-v1":
                response = item["memory-v1_rag_response"]
                #response = item["debug_rag_response"]
                rag_responses.append(response)
                evaluation_results = item["evaluation_results"]["memory-v1_rag"]
                #evaluation_results = item["evaluation_results"]["debug_rag"]
            elif rag_mode == "debug":
                response = item["debug_rag_response"]
                rag_responses.append(response)
                evaluation_results = item["evaluation_results"]["debug_rag"]
            else:
                raise ValueError(f"Unknown rag_mode: {rag_mode}")

            if "prompt" in evaluation_results:
                prompt = evaluation_results["prompt"]
                eval_statistics["prompt"].append(prompt)

            if "comprehensiveness" in evaluation_results:
                comprehensiveness = evaluation_results["comprehensiveness"]
                if "comprehensiveness" in eval_statistics:
                    eval_statistics["comprehensiveness"]["level"].append(comprehensiveness[0])
                    eval_statistics["comprehensiveness"]["score"].append(comprehensiveness[1])
                    eval_statistics["comprehensiveness"]["explanation"].append(comprehensiveness[2])
            if "diversity" in evaluation_results:
                diversity = evaluation_results["diversity"]
                if "diversity" in eval_statistics:
                    eval_statistics["diversity"]["level"].append(diversity[0])
                    eval_statistics["diversity"]["score"].append(diversity[1])
                    eval_statistics["diversity"]["explanation"].append(diversity[2])
            if "relevance" in evaluation_results:
                relevance = evaluation_results["relevance"]
                if "relevance" in eval_statistics:
                    eval_statistics["relevance"]["level"].append(relevance[0])
                    eval_statistics["relevance"]["score"].append(relevance[1])
                    eval_statistics["relevance"]["explanation"].append(relevance[2])

        all_eval_statistics.append(eval_statistics)
        a = 1


def inspect_simple_questions_from_full(target_dir):
    simple_questions_dict = fetch_all_simple_questions_dict(target_dir)

    questions = []
    ref_answers = []
    rag_responses = []
    raised_questions_eval_path = os.path.join(target_dir, f"responses_to_raised_questions_{source_type}_{rag_mode}_{LLM_MODEL_NAME}_{graph_mode}_eval.jsonl")
    print(f"Evaluation of simple questions from {raised_questions_eval_path}")
    with open(raised_questions_eval_path, "r", encoding="utf-8") as rjf:
        for line in rjf:
            item = json.loads(line.strip())
            question = item["question"]
            if question not in simple_questions_dict:
                continue
            ref_answer = item["reference_answer"]
            questions.append(question)
            ref_answers.append(ref_answer)
            involved_entities = item["involved_entities"] if "involved_entities" in item else None
            origin_full_doc_id = item["origin_full_doc_id"] if "origin_full_doc_id" in item else None
            origin_chunks_ids = item["origin_chunks_ids"] if "origin_chunks_ids" in item else None
            origin_chunks_order_index = item["origin_chunks_order_index"] if "origin_chunks_order_index" in item else None
            origin_chunks = item["origin_chunks"] if "origin_chunks" in item else None
            if origin_chunks is not None:
                paragraph = combine_consecutive_overlapping_chunks(origin_chunks)

            if rag_mode == "naive":
                response = item["naive_rag_response"]
                rag_responses.append(response)
                evaluation_results = item["evaluation_results"]["naive_rag"]

            elif rag_mode == "light-direct":
                response = item["light-direct_rag_response"]
                rag_responses.append(response)
                evaluation_results = item["evaluation_results"]["light-direct_rag"]
            elif rag_mode == "memory-v1":
                response = item["memory-v1_rag_response"]
                #response = item["debug_rag_response"]
                rag_responses.append(response)
                evaluation_results = item["evaluation_results"]["memory-v1_rag"]
                #evaluation_results = item["evaluation_results"]["debug_rag"]
            elif rag_mode == "debug":
                response = item["debug_rag_response"]
                rag_responses.append(response)
                evaluation_results = item["evaluation_results"]["debug_rag"]
            else:
                raise ValueError(f"Unknown rag_mode: {rag_mode}")

            if "prompt" in evaluation_results:
                prompt = evaluation_results["prompt"]
                eval_statistics["prompt"].append(prompt)

            if "comprehensiveness" in evaluation_results:
                comprehensiveness = evaluation_results["comprehensiveness"]
                if "comprehensiveness" in eval_statistics:
                    eval_statistics["comprehensiveness"]["level"].append(comprehensiveness[0])
                    eval_statistics["comprehensiveness"]["score"].append(comprehensiveness[1])
                    eval_statistics["comprehensiveness"]["explanation"].append(comprehensiveness[2])
            if "diversity" in evaluation_results:
                diversity = evaluation_results["diversity"]
                if "diversity" in eval_statistics:
                    eval_statistics["diversity"]["level"].append(diversity[0])
                    eval_statistics["diversity"]["score"].append(diversity[1])
                    eval_statistics["diversity"]["explanation"].append(diversity[2])
            if "relevance" in evaluation_results:
                relevance = evaluation_results["relevance"]
                if "relevance" in eval_statistics:
                    eval_statistics["relevance"]["level"].append(relevance[0])
                    eval_statistics["relevance"]["score"].append(relevance[1])
                    eval_statistics["relevance"]["explanation"].append(relevance[2])

        all_eval_statistics.append(eval_statistics)
        a=1


def inspect_sampled_simple_questions_from_full(target_dir):
    #all_simple_questions_dict = fetch_all_simple_questions_dict(target_dir)
    sampled_simple_questions_dict = fetch_sampled_simple_questions_dict(target_dir)

    questions = []
    ref_answers = []
    rag_responses = []
    all_simple_questions_eval_path = os.path.join(target_dir, f"responses_to_all_simple_questions_{source_type}_{rag_mode}_{LLM_MODEL_NAME}_{graph_mode}_eval.jsonl")
    print(f"Evaluation of all simple questions from {all_simple_questions_eval_path}")
    with open(all_simple_questions_eval_path, "r", encoding="utf-8") as rjf:
        for line in rjf:
            item = json.loads(line.strip())
            question = item["question"]
            if question not in sampled_simple_questions_dict:
                continue
            ref_answer = item["reference_answer"]
            questions.append(question)
            ref_answers.append(ref_answer)
            involved_entities = item["involved_entities"] if "involved_entities" in item else None
            origin_full_doc_id = item["origin_full_doc_id"] if "origin_full_doc_id" in item else None
            origin_chunks_ids = item["origin_chunks_ids"] if "origin_chunks_ids" in item else None
            origin_chunks_order_index = item["origin_chunks_order_index"] if "origin_chunks_order_index" in item else None
            origin_chunks = item["origin_chunks"] if "origin_chunks" in item else None
            if origin_chunks is not None:
                paragraph = combine_consecutive_overlapping_chunks(origin_chunks)

            if rag_mode == "naive":
                response = item["naive_rag_response"]
                rag_responses.append(response)
                evaluation_results = item["evaluation_results"]["naive_rag"]

            elif rag_mode == "light-direct":
                response = item["light-direct_rag_response"]
                rag_responses.append(response)
                evaluation_results = item["evaluation_results"]["light-direct_rag"]
            elif rag_mode == "memory-v1":
                response = item["memory-v1_rag_response"]
                #response = item["debug_rag_response"]
                rag_responses.append(response)
                evaluation_results = item["evaluation_results"]["memory-v1_rag"]
                #evaluation_results = item["evaluation_results"]["debug_rag"]
            elif rag_mode == "debug":
                response = item["debug_rag_response"]
                rag_responses.append(response)
                evaluation_results = item["evaluation_results"]["debug_rag"]
            else:
                raise ValueError(f"Unknown rag_mode: {rag_mode}")

            if "prompt" in evaluation_results:
                prompt = evaluation_results["prompt"]
                eval_statistics["prompt"].append(prompt)

            if "comprehensiveness" in evaluation_results:
                comprehensiveness = evaluation_results["comprehensiveness"]
                if "comprehensiveness" in eval_statistics:
                    eval_statistics["comprehensiveness"]["level"].append(comprehensiveness[0])
                    eval_statistics["comprehensiveness"]["score"].append(comprehensiveness[1])
                    eval_statistics["comprehensiveness"]["explanation"].append(comprehensiveness[2])
            if "diversity" in evaluation_results:
                diversity = evaluation_results["diversity"]
                if "diversity" in eval_statistics:
                    eval_statistics["diversity"]["level"].append(diversity[0])
                    eval_statistics["diversity"]["score"].append(diversity[1])
                    eval_statistics["diversity"]["explanation"].append(diversity[2])
            if "relevance" in evaluation_results:
                relevance = evaluation_results["relevance"]
                if "relevance" in eval_statistics:
                    eval_statistics["relevance"]["level"].append(relevance[0])
                    eval_statistics["relevance"]["score"].append(relevance[1])
                    eval_statistics["relevance"]["explanation"].append(relevance[2])

        all_eval_statistics.append(eval_statistics)
        a=1


max_prompt_len = 0
async def inspect_prompt(llm_response_cache):
    global max_prompt_len
    all_items = await llm_response_cache.get_by_id(rag_mode)
    for k, v in all_items.items():
        prompt = v["prompt"]
        #encoded = tokenizer.encode(prompt)
        encoded = encode_string_by_tiktoken(prompt)
        max_prompt_len = max(max_prompt_len, len(encoded))


def inspect_cached_content(target_dir):
    sampled_difficult_questions_dict = fetch_sampled_difficult_questions_dict(target_dir)

    #eval_path = os.path.join(target_dir,f"responses_to_all_difficult_questions_{source_type}_{rag_mode}_{LLM_MODEL_NAME}_{graph_mode}_eval.jsonl")
    eval_path = os.path.join(target_dir, f"responses_to_{question_type}_questions_{source_type}_{rag_mode}_{LLM_MODEL_NAME}_{graph_mode}.jsonl")
    #eval_path = os.path.join(target_dir, f"responses_to_{question_type}_questions_{source_type}_{rag_mode}_{LLM_MODEL_NAME}_{graph_mode}_eval.jsonl")
    print(f"Path of evaluation: {eval_path}")
    with open(eval_path, "r", encoding="utf-8") as rjf:
        for idx, line in enumerate(rjf, start=1):
            item = json.loads(line.strip())
            question = item["question"]
            ref_answer = item["reference_answer"]
            if question not in sampled_difficult_questions_dict:
                continue
            args_hash = compute_args_hash(rag_mode, question, LLM_MODEL_NAME)
            mode_cache = asyncio.run(llm_response_cache.get_by_id(rag_mode))
            cached_content = mode_cache[args_hash]
            print(idx)

    return cached_content


def inspect_target_questions(target_dir):
    questions = []
    ref_answers = []
    rag_responses = []
    eval_path = os.path.join(target_dir, f"responses_to_{question_type}_questions_{source_type}_{rag_mode}_{LLM_MODEL_NAME}_{graph_mode}_eval.jsonl")
    print(f"Path of evaluation: {eval_path}")
    with open(eval_path, "r", encoding="utf-8") as rjf:
        for line in rjf:
            item = json.loads(line.strip())
            question = item["question"]
            ref_answer = item["reference_answer"]
            questions.append(question)
            ref_answers.append(ref_answer)
            involved_entities = item["involved_entities"] if "involved_entities" in item else None
            origin_full_doc_id = item["origin_full_doc_id"] if "origin_full_doc_id" in item else None
            origin_chunks_ids = item["origin_chunks_ids"] if "origin_chunks_ids" in item else None
            origin_chunks_order_index = item["origin_chunks_order_index"] if "origin_chunks_order_index" in item else None
            origin_chunks = item["origin_chunks"] if "origin_chunks" in item else None
            if origin_chunks is not None:
                paragraph = combine_consecutive_overlapping_chunks(origin_chunks)
            if rag_mode == "naive":
                response = item["naive_rag_response"]
                rag_responses.append(response)
                evaluation_results = item["evaluation_results"]["naive_rag"]
            elif rag_mode == "light-direct":
                response = item["light-direct_rag_response"]
                rag_responses.append(response)
                evaluation_results = item["evaluation_results"]["light-direct_rag"]
            elif rag_mode == "memory-v1":
                response = item["memory-v1_rag_response"]
                rag_responses.append(response)
                evaluation_results = item["evaluation_results"]["memory-v1_rag"]
            elif rag_mode == "memory-v1_hint":
                response = item["memory-v1_hint_rag_response"]
                rag_responses.append(response)
                evaluation_results = item["evaluation_results"]["memory-v1_hint_rag"]
            elif rag_mode == "memory-v2":
                response = item["memory-v2_rag_response"]
                rag_responses.append(response)
                evaluation_results = item["evaluation_results"]["memory-v2_rag"]
            elif rag_mode == "memory-v3":
                response = item["memory-v3_rag_response"]
                rag_responses.append(response)
                evaluation_results = item["evaluation_results"]["memory-v3_rag"]
            elif rag_mode == "debug":
                response = item["debug_rag_response"]
                rag_responses.append(response)
                evaluation_results = item["evaluation_results"]["debug_rag"]
            else:
                raise ValueError(f"Unknown rag_mode: {rag_mode}")

            args_hash = compute_args_hash(rag_mode, question, LLM_MODEL_NAME)
            mode_cache = asyncio.run(llm_response_cache.get_by_id(rag_mode))
            cached_content = mode_cache[args_hash]

            if "prompt" in evaluation_results:
                prompt = evaluation_results["prompt"]
                eval_statistics["prompt"].append(prompt)

            if "comprehensiveness" in evaluation_results:
                comprehensiveness = evaluation_results["comprehensiveness"]
                if "comprehensiveness" in eval_statistics:
                    eval_statistics["comprehensiveness"]["level"].append(comprehensiveness[0])
                    eval_statistics["comprehensiveness"]["score"].append(comprehensiveness[1])
                    eval_statistics["comprehensiveness"]["explanation"].append(comprehensiveness[2])
            if "diversity" in evaluation_results:
                diversity = evaluation_results["diversity"]
                if "diversity" in eval_statistics:
                    eval_statistics["diversity"]["level"].append(diversity[0])
                    eval_statistics["diversity"]["score"].append(diversity[1])
                    eval_statistics["diversity"]["explanation"].append(diversity[2])
            if "relevance" in evaluation_results:
                relevance = evaluation_results["relevance"]
                if "relevance" in eval_statistics:
                    eval_statistics["relevance"]["level"].append(relevance[0])
                    eval_statistics["relevance"]["score"].append(relevance[1])
                    eval_statistics["relevance"]["explanation"].append(relevance[2])

        all_eval_statistics.append(eval_statistics)
        a=1


def calculate_metrics(eval_statistics):
    levels_comprehensiveness, scores_comprehensiveness = [], []
    levels_diversity, scores_diversity = [], []
    levels_relevance, scores_relevance = [], []
    for stat_dict in eval_statistics:
        prompt = stat_dict["prompt"]
        comprehensiveness = stat_dict["comprehensiveness"]
        level_comprehensiveness, score_comprehensiveness = comprehensiveness["level"], comprehensiveness["score"]
        levels_comprehensiveness.extend(level_comprehensiveness)
        scores_comprehensiveness.extend(score_comprehensiveness)
        diversity = stat_dict["diversity"]
        level_diversity, score_diversity = diversity["level"], diversity["score"]
        levels_diversity.extend(level_diversity)
        scores_diversity.extend(score_diversity)
        relevance = stat_dict["relevance"]
        level_relevance, score_relevance = relevance["level"], relevance["score"]
        levels_relevance.extend(level_relevance)
        scores_relevance.extend(score_relevance)
    print("Average Levels Comprehensiveness:", sum(levels_comprehensiveness)/len(levels_comprehensiveness))
    print("Average Scores Comprehensiveness:", sum(scores_comprehensiveness)/len(scores_comprehensiveness))
    print("Average Levels Diversity:", sum(levels_diversity)/len(levels_diversity))
    print("Average Scores Diversity:", sum(scores_diversity)/len(scores_diversity))
    a=1


if __name__ == "__main__":
    """
    from transformers import AutoModel, AutoTokenizer
    #model_name_or_path = os.path.join(PROJ_DIR, 'cache/xlm-roberta-base')
    model_name_or_path = os.path.join(PROJ_DIR, 'cache/Qwen2.5-7B-Instruct')
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    embed_model = AutoModel.from_pretrained(model_name_or_path)
    embed_hidden_size = embed_model.config.hidden_size
    """
    embed_hidden_size = 1024

    graph_mode = "single"
    #graph_mode = "merge"
    #question_type = "raised"
    #question_type = "simple"
    #question_type = "difficult"
    #question_type = "all_simple"
    #question_type = "all_difficult"
    #question_type = "sampled_simple"
    question_type = "sampled_difficult"
    #source_type = "chunks50"
    source_type = "chunks100"
    #rag_mode = "naive"
    #rag_mode = "light-direct"
    #rag_mode = "memory-v1"
    #rag_mode = "memory-v1_hint"
    #rag_mode = "memory-v2"
    rag_mode = "memory-v3"
    #rag_mode = "debug"
    #LLM_MODEL_NAME = "gpt-4o"
    #LLM_MODEL_NAME = "Qwen2.5-7B-Instruct"
    LLM_MODEL_NAME = "Qwen2.5-32B-Instruct"
    #DATASET_NAME = "ultradomain"
    #domains = ["agriculture", "art", "fin", "legal", "mathematics", "neurology", "pathology", "physics"]
    #domains = ["agriculture"]
    DATASET_NAME = "longbench_v2_qa"
    #domains = ["Academic", "Detective", "Event ordering", "Financial", "Governmental", "Legal", "Literary", "Multi-news"]
    #domains = ["Financial", "Governmental", "Legal", "Literary"]
    domains = ["Legal"]
    print(f"RAG mode: {rag_mode}, Question type: {source_type}")
    for domain in domains:
        DOMAIN_DIR = os.path.join(PROJ_DIR, f"data/{DATASET_NAME}/domains/{domain}")
        valid_subdirs = sorted([int(subdir) for subdir in os.listdir(DOMAIN_DIR) if subdir.isdigit()])
        all_difficult_questions = []
        all_eval_statistics = []
        #sample_difficult_questions()
        #sample_simple_questions()
        #"""
        for i in range(0, len(valid_subdirs)):
            #if i >= 1:
            #    break
            if graph_mode == "merge":
                work_dir = os.path.join(DOMAIN_DIR, f"merged_{i}_top5_similar")
            else:
                work_dir = os.path.join(DOMAIN_DIR, f"{i}")
            target_dir = os.path.join(PROJ_DIR, f"data/created_data/{DATASET_NAME}/{domain}/{i}")
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)

            llm_response_cache = load_llm_response_cache(storage_class["JsonKVStorage"], work_dir, embed_hidden_size=embed_hidden_size)

            eval_statistics = {
                "prompt": [],
                "comprehensiveness": {"level": [], "score": [], "explanation": []},
                "diversity": {"level": [], "score": [], "explanation": []},
                "relevance": {"level": [], "score": [], "explanation": []}
            }
            #fetch_difficult_questions_dict(target_dir)
            #inspect_target_questions(target_dir)
            inspect_cached_content(target_dir)
            #collect_all_difficult_questions(target_dir)
            #collect_all_simple_questions(target_dir)
            #inspect_difficult_questions_from_full(target_dir)
            #inspect_sampled_difficult_questions_from_all(target_dir)
            #inspect_simple_questions_from_full(target_dir)
            #llm_response_cache = load_llm_response_cache(storage_class["JsonKVStorage"], work_dir, embed_hidden_size=embed_hidden_size)
            #loop = asyncio.get_event_loop()
            #loop.run_until_complete(inspect_prompt(llm_response_cache))
        #calculate_metrics(all_eval_statistics)
        #num_difficult_questions_of_this_domain = sum([len(difficult_questions) for difficult_questions in all_difficult_questions])
        #print(f"{num_difficult_questions_of_this_domain} difficult questions")
        #"""
    print("Finish")