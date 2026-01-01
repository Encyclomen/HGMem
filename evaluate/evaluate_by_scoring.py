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

from myrag.utils import safe_chat_call, compute_args_hash
from myrag.wechatgpt_interface import OpenaiWeChat
from scripts.utils import storage_class, load_text_chunks, load_llm_response_cache, load_graph, load_entity_vdb, load_relationship_vdb, combine_consecutive_overlapping_chunks

APP_ID = "xxx"
APP_KEY = "xxx"
model = "gpt-4o"
chat = OpenaiWeChat(APP_ID, APP_KEY)

PROMPTS = {
################################################################################
"evaluate_response_comprehensiveness": "Given a [Paragraph], a [Response] to the [Question], you will conduct evaluation by using the relevant [Paragraph] in terms of Comprehensiveness.\n\n"
"Comprehensiveness measures whether the [Response] comprehensively covers all key aspects of the question and whether there are non-negligible missing content.\n"
"Level   | score range | description\n"
"Level 1 | 0-20   | The response is extremely one-sided, leaving out key parts or important aspects of the question.\n"
"Level 2 | 20-40  | The response has some content, but it misses many important aspects of the question and is not comprehensive enough.\n"
"Level 3 | 40-60  | The response is moderately comprehensive, covering the main aspects of the question, but still missing some important aspects.\n"
"Level 4 | 60-80  | The response is comprehensive, covering most aspects of the question, with few missing details.\n"
"Level 5 | 80-100 | The response is extremely comprehensive, covering almost all aspects of the question with sufficient details, enabling the reader to gain a complete and thorough understanding.\n\n"
"Evaluate the [Response] using the criteria listed above, give a level of comprehensiveness in [Level] based on the description of the indicator, then give a score in [Score] based on the corresponding value range, and finally explain in [Explanation].\n\n"
"######################-Anticipated Output Format-######################\n"
"[Level]: A level ranging from 1 to 5  # This should be a single number, not a range\n"
"[Score]: A value ranging from 0 to 100  # This should be a single number satisfying the ranging constraint of the corresponding [Level], not a range\n"
"[Explanation]: xxx \n"
"\n\n"
"######################-Real Case-######################\n"
"[Paragraph]:\n{paragraph}\n\n"
"[Question]:\n{question}\n\n"
"[Response]:\n{response}\n\n",
################################################################################
"evaluate_response_comprehensiveness_2": "Given a [Question] and a [Response], you will evaluate the quality of the [Response] by using the [Reference Answer] in terms of Comprehensiveness.\n\n"
"Comprehensiveness measures whether the [Response] comprehensively covers all key aspects and non-negligible content in the [Reference Answer] with respect to the [Question].\n"
"Level   | score range | description\n"
"Level 1 | 0-20   | The response is extremely one-sided, leaving out key parts or important aspects of the question.\n"
"Level 2 | 20-40  | The response has some content, but it misses many important aspects of the question and is not comprehensive enough.\n"
"Level 3 | 40-60  | The response is moderately comprehensive, covering the main aspects of the question, but still missing some important aspects.\n"
"Level 4 | 60-80  | The response is comprehensive, covering most aspects of the question, with few missing details.\n"
"Level 5 | 80-100 | The response is extremely comprehensive, covering almost all aspects of the question with sufficient details, enabling the reader to gain a complete and thorough understanding.\n\n"
"Evaluate the [Response] using the criteria listed above, give a level of comprehensiveness in [Level] based on the description of the indicator, then give a score in [Score] based on the corresponding value range, and finally explain in [Explanation].\n"
"Note that only assess the [Response] by referencing to the [Reference Answer] and avoid misinterpreting any content of [Reference Answer] as part of the [Response].\n\n"
"######################-Anticipated Output Format-######################\n"
"[Level]: A level ranging from 1 to 5  # This should be a single number, not a range\n"
"[Score]: A value ranging from 0 to 100  # This should be a single number satisfying the ranging constraint of the corresponding [Level], not a range\n"
"[Explanation]: xxx \n"
"\n\n"
"######################-Real Case-######################\n"
"[Question]:\n{question}\n\n"
"[Reference Answer]:\n{ref_answer}\n\n"
"[Response]:\n{response}\n\n",
################################################################################
"evaluate_response_diversity": "Given a [Paragraph], a [Response] to the [Question], you will conduct evaluation by using the relevant [Paragraph] in terms of Diversity.\n\n"
"Diversity measures how varied and rich is the response in offering different perspectives and insights related to the question.\n"
"Level   | score range | description\n"
"Level 1 | 0-20   | The response is extremely narrow and repetitive, providing only a single perspective or insight without exploring alternative viewpoints or additional information.\n"
"Level 2 | 20-40  | The response offers a few different perspectives but remains largely superficial. It may touch on alternative viewpoints but does not elaborate or provide substantial insights.\n"
"Level 3 | 40-60  | The response moderately presents several perspectives with moderate depth. It begins to integrate different viewpoints and insights but may still miss some important angles or lack thorough exploration.\n"
"Level 4 | 60-80  | The response is rich in perspectives and insights. It basically explores multiple viewpoints and provides substantial evidence and examples to support each angle.\n"
"Level 5 | 80-100 | The response is exceptionally varied and rich in perspectives and insights. It offers a comprehensive exploration of the question, addressing multiple angles with depth and originality.\n\n"
"Evaluate the [Response] using the criteria listed above, give a level of diversity in [Level] based on the description of the indicator, then give a score in [Score] based on the corresponding value range, and finally explain in [Explanation].\n\n"
"######################-Anticipated Output Format-######################\n"
"[Level]: A level ranging from 1 to 5  # This should be a single number, not a range\n"
"[Score]: A value ranging from 0 to 100  # This should be a single number satisfying the ranging constraint of the corresponding [Level], not a range\n"
"[Explanation]: xxx \n"
"\n\n"
"######################-Real Case-######################\n"
"[Paragraph]:\n{paragraph}\n\n"
"[Question]:\n{question}\n\n"
"[Response]:\n{response}\n\n",
################################################################################
"evaluate_response_diversity_2": "Given a [Question] and a [Response], you will evaluate the quality of the [Response] by using the [Reference Answer] in terms of Diversity.\n\n"
"Diversity measures how varied and rich is the response in offering different perspectives and insights related to the question.\n"
"Level   | score range | description\n"
"Level 1 | 0-20   | The response is extremely narrow and repetitive, providing only a single perspective or insight without exploring alternative viewpoints or additional information.\n"
"Level 2 | 20-40  | The response offers a few different perspectives but remains largely superficial. It may touch on alternative viewpoints but does not elaborate or provide substantial insights.\n"
"Level 3 | 40-60  | The response moderately presents several perspectives with moderate depth. It begins to integrate different viewpoints and insights but may still miss some important angles or lack thorough exploration.\n"
"Level 4 | 60-80  | The response is rich in perspectives and insights. It basically explores multiple viewpoints and provides substantial evidence and examples to support each angle.\n"
"Level 5 | 80-100 | The response is exceptionally varied and rich in perspectives and insights. It offers a comprehensive exploration of the question, addressing multiple angles with depth and originality.\n\n"
"Evaluate the [Response] using the criteria listed above, give a level of diversity in [Level] based on the description of the indicator, then give a score in [Score] based on the corresponding value range, and finally explain in [Explanation].\n"
"Note that only assess the [Response] by referencing to the [Reference Answer] and avoid misinterpreting any content of [Reference Answer] as part of the [Response].\n\n"
"######################-Anticipated Output Format-######################\n"
"[Level]: A level ranging from 1 to 5  # This should be a single number, not a range\n"
"[Score]: A value ranging from 0 to 100  # This should be a single number satisfying the ranging constraint of the corresponding [Level], not a range\n"
"[Explanation]: xxx \n"
"\n\n"
"######################-Real Case-######################\n"
"[Question]:\n{question}\n\n"
"[Reference Answer]:\n{ref_answer}\n\n"
"[Response]:\n{response}\n\n",
################################################################################
"evaluate_response_relevance": "Given a [Paragraph], a [Response] to the [Question], you will conduct evaluation by using the relevant [Paragraph] in terms of Diversity.\n\n"
"Relevance reflects how accurately and directly the response addresses the target question without containing extraneous irrelevant content\n"
"Level   | score range | description\n"
"Level 1 | 0-20   | The response does not address the question at all. It lacks any pertinent details, concepts, or facts from the document. The content may be completely off-topic or irrelevant.\n"
"Level 2 | 20-40  | The response attempts to address the question but does so inadequately. It includes minimal relevant information from the document and may contain obviously irrelevant or redundant content.\n"
"Level 3 | 40-60  | The response addresses the question with some relevant information from the document. However, it may lack comprehensiveness or clarity and could include some irrelevant content.\n"
"Level 4 | 60-80  | The response effectively addresses the question with pertinent details, concepts, or facts from the document. It is mostly comprehensive and clear, with minimal irrelevant or redundant content.\n"
"Level 5 | 80-100 | The response fully addresses the question with all necessary and pertinent information from the document. It is comprehensive, clear, and concise, with no irrelevant or redundant content.\n\n"
"Evaluate the [Response] using the criteria listed above, give a level of relevance in [Level] based on the description of the indicator, then give a score in [Score] based on the corresponding value range, and finally explain in [Explanation].\n\n"
"######################-Anticipated Output Format-######################\n"
"[Level]: A level ranging from 1 to 5  # This should be a single number, not a range\n"
"[Score]: A value ranging from 0 to 100  # This should be a single number satisfying the ranging constraint of the corresponding [Level], not a range\n"
"[Explanation]: xxx \n"
"\n\n"
"######################-Real Case-######################\n"
"[Paragraph]:\n{paragraph}\n\n"
"[Question]:\n{question}\n\n"
"[Response]:\n{response}\n\n",
}


def post_process_evaluate(response: str):
    response = response.strip()
    loc_level = response.find("[Level]:")
    loc_score = response.find("[Score]:")
    loc_explanation = response.find("[Explanation]:")
    assert loc_level != -1 and loc_score != -1 and loc_explanation != -1
    level_str = response[loc_level:loc_score].strip()
    score_str = response[loc_score:loc_explanation].strip()
    explanation_str = response[loc_explanation:].strip()

    level = int(level_str.replace("[Level]:", "").strip())
    score = float(score_str.replace("[Score]:", "").strip())
    explanation = explanation_str.replace("[Explanation]:", "").strip()

    return level, score, explanation


async def main(llm_response_cache, text_chunk_storage, target_dir):
    qa_filename = f"responses_to_{question_type}_questions_{source_type}_{rag_mode}_{LLM_MODEL_NAME}_{graph_mode}"
    qa_path = os.path.join(target_dir, f"{qa_filename}.jsonl")
    eval_path = os.path.join(target_dir, f"{qa_filename}_eval.jsonl")
    #qa_path = os.path.join(target_dir, f"{qa_filename}_eval.jsonl")
    #eval_path = os.path.join(target_dir, f"{qa_filename}_eval2.jsonl")

    with open(qa_path, "r", encoding="utf-8") as rjf, open(eval_path, "a", encoding="utf-8") as wjf:
        for line_idx, line in enumerate(rjf, start=1):
            #if i < 10 or (i == 10 and line_idx <= 7):
            #    continue
            item = json.loads(line.strip())
            question = item["question"]
            ref_answer = item["reference_answer"]

            involved_entities = item["involved_entities"] if "involved_entities" in item else None
            origin_full_doc_id = item["origin_full_doc_id"] if "origin_full_doc_id" in item else None
            origin_chunks_ids = item["origin_chunks_ids"] if "origin_chunks_ids" in item else None
            origin_chunks_order_index = item["origin_chunks_order_index"] if "origin_chunks_order_index" in item else None
            origin_chunks = item["origin_chunks"] if "origin_chunks" in item else None
            if origin_chunks is not None:
                paragraph = combine_consecutive_overlapping_chunks(origin_chunks)
            new_item = copy.deepcopy(item)
            if "evaluation_results" not in new_item:
                new_item["evaluation_results"] = {}

            args_hash = compute_args_hash(rag_mode, question, LLM_MODEL_NAME)
            mode_cache = await llm_response_cache.get_by_id(rag_mode)
            #prompt = ""
            prompt = mode_cache[args_hash]["prompt"]

            if rag_mode == "naive":
                response = item["naive_rag_response"]
                #naive_rag_retrieved_objects = mode_cache[args_hash]["misc"]["retrieved_objects"]
                #naive_rag_retrieved_chunks_ids = naive_rag_retrieved_objects["text_chunks_ids"]
                if "naive_rag" not in new_item["evaluation_results"]:
                    new_item["evaluation_results"]["naive_rag"] = {}
            elif rag_mode == "light-direct":
                response = item["light-direct_rag_response"]
                #light_direct_rag_retrieved_objects = mode_cache[args_hash]["misc"]["retrieved_objects"]
                #light_direct_rag_retrieved_entities_ids = light_direct_rag_retrieved_objects["entities_ids"]
                #light_direct_rag_retrieved_relationships_ids = light_direct_rag_retrieved_objects["relationships_ids"]
                #light_direct_rag_retrieved_chunks_ids = light_direct_rag_retrieved_objects["text_chunks_ids"]
                if "light-direct_rag" not in new_item["evaluation_results"]:
                    new_item["evaluation_results"]["light-direct_rag"] = {}
            elif rag_mode == "memory-v1":
                response = item["memory-v1_rag_response"]
                if "memory-v1_rag" not in new_item["evaluation_results"]:
                    new_item["evaluation_results"]["memory-v1_rag"] = {}
            elif rag_mode == "memory-v1_hint":
                response = item["memory-v1_hint_rag_response"]
                if "memory-v1_hint_rag" not in new_item["evaluation_results"]:
                    new_item["evaluation_results"]["memory-v1_hint_rag"] = {}
            elif rag_mode == "memory-v2":
                response = item["memory-v2_rag_response"]
                if "memory-v2_rag" not in new_item["evaluation_results"]:
                    new_item["evaluation_results"]["memory-v2_rag"] = {}
            elif rag_mode == "memory-v3":
                response = item["memory-v3_rag_response"]
                if "memory-v3_rag" not in new_item["evaluation_results"]:
                    new_item["evaluation_results"]["memory-v3_rag"] = {}
            elif rag_mode == "debug":
                response = item["debug_rag_response"]
                if "debug_rag" not in new_item["evaluation_results"]:
                    new_item["evaluation_results"]["debug_rag"] = {}
            else:
                raise ValueError(f"Unknown rag_mode: {rag_mode}")

            new_item["evaluation_results"][f"{rag_mode}_rag"]["prompt"] = prompt
            context_base = dict(
                ref_answer=ref_answer,
                question=question,
                response=response,
            )
            ############
            # """
            use_prompt_evaluate_response = PROMPTS["evaluate_response_comprehensiveness_2"].format(**context_base)
            system_message = (f"---Role---\nYou are an expert tasked with evaluating a response to the question from a specific aspect by using the relevant paragraph.\n")
            response_evaluate_response, api_state = safe_chat_call(chat, model, system_message, use_prompt_evaluate_response, temperature=1.0)
            if api_state != "Success":
                raise ConnectionError("API error.")
            level_comprehensiveness, score_comprehensiveness, explanation_comprehensiveness = post_process_evaluate(response_evaluate_response)
            new_item["evaluation_results"][f"{rag_mode}_rag"]["comprehensiveness"] = (level_comprehensiveness, score_comprehensiveness, explanation_comprehensiveness)
            # """
            ############
            # """
            use_prompt_evaluate_response = PROMPTS["evaluate_response_diversity_2"].format(**context_base)
            system_message = (f"---Role---\nYou are an expert tasked with evaluating a response to the question from a specific aspect by using the relevant paragraph.\n")
            response_evaluate_response, api_state = safe_chat_call(chat, model, system_message, use_prompt_evaluate_response, temperature=1.0)
            if api_state != "Success":
                raise ConnectionError("API error.")
            level_diversity, score_diversity, explanation_diversity = post_process_evaluate(response_evaluate_response)
            new_item["evaluation_results"][f"{rag_mode}_rag"]["diversity"] = (level_diversity, score_diversity, explanation_diversity)
            # """
            ############
            """
            use_prompt_evaluate_response = PROMPTS["evaluate_response_relevance"].format(**context_base)
            system_message = (f"---Role---\nYou are an expert tasked with evaluating a response to the question from a specific aspect by using the relevant paragraph.\n")
            response_evaluate_response, api_state = safe_chat_call(chat, model, system_message, use_prompt_evaluate_response, temperature=1.0)
            if api_state != "Success":
                raise ConnectionError("API error.")
            level_relevance, score_relevance, explanation_relevance = post_process_evaluate(response_evaluate_response)
            new_item["evaluation_results"][f"{rag_mode}_rag"]["relevance"] = (level_relevance, score_relevance, explanation_relevance)
            """
            ############
            print(eval_path)
            print(line_idx)
            wjf.write(json.dumps(new_item, ensure_ascii=False) + "\n")
            a=1


if __name__ == "__main__":
    """
    from transformers import AutoModel, AutoTokenizer
    model_name_or_path = os.path.join(PROJ_DIR, 'cache/xlm-roberta-base')
    #model_name_or_path = "jinaai/jina-embeddings-v3"
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
    question_type = "sampled_difficult"
    #source_type = "chunks50"
    source_type = "chunks100"
    #rag_mode = "naive"
    #rag_mode = "light-direct"
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
    domains = ["Legal"]
    for domain in domains:
        DOMAIN_DIR = os.path.join(PROJ_DIR, f"data/{DATASET_NAME}/domains/{domain}")
        valid_subdirs = sorted([int(subdir) for subdir in os.listdir(DOMAIN_DIR) if subdir.isdigit()])
        #for i in range(1, len(valid_subdirs)):
        for i in range(0, 1):
            if graph_mode == "merge":
                work_dir = os.path.join(DOMAIN_DIR, f"merged_{i}_top5_similar")
            else:
                work_dir = os.path.join(DOMAIN_DIR, f"{i}")

            target_dir = os.path.join(PROJ_DIR, f"data/created_data/{DATASET_NAME}/{domain}/{i}")
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)

            text_chunk_storage = load_text_chunks(storage_class["JsonKVStorage"], work_dir, embed_hidden_size=embed_hidden_size)
            graph = load_graph(storage_class["NetworkXStorage"], work_dir, embed_hidden_size=embed_hidden_size)
            llm_response_cache = load_llm_response_cache(storage_class["JsonKVStorage"], work_dir, embed_hidden_size=embed_hidden_size)

            loop = asyncio.get_event_loop()
            loop.run_until_complete(main(llm_response_cache, text_chunk_storage, target_dir))

    print("Finish")