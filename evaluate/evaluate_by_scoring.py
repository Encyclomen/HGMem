import asyncio
import copy
import json
import os
import sys
from datetime import datetime

from myrag.llm import openai_complete_if_cache


PROJ_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJ_DIR)

from scripts.utils import storage_class, load_text_chunks, load_llm_response_cache, load_graph, \
    combine_consecutive_overlapping_chunks
from myrag.utils import compute_args_hash

APP_KEY = "xxx"
BASE_URL = None
EVAL_MODEL = "gpt-4o-mini"

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
    # 增加一些鲁棒性处理，防止 LLM 输出 Markdown 代码块包裹
    response = response.replace("```json", "").replace("```", "")

    loc_level = response.find("[Level]:")
    loc_score = response.find("[Score]:")
    loc_explanation = response.find("[Explanation]:")

    if loc_level == -1 or loc_score == -1 or loc_explanation == -1:
        # 如果格式严重错误，可以返回默认值或抛出异常
        print(f"Warning: Output format mismatch. Response: {response}")
        return 0, 0.0, "Format Error"

    level_str = response[loc_level:loc_score].strip()
    score_str = response[loc_score:loc_explanation].strip()
    explanation_str = response[loc_explanation:].strip()

    try:
        level = int(level_str.replace("[Level]:", "").strip())
        score = float(score_str.replace("[Score]:", "").strip())
        explanation = explanation_str.replace("[Explanation]:", "").strip()
    except ValueError:
        print(f"Warning: Value parsing error. Response: {response}")
        return 0, 0.0, "Parsing Error"

    return level, score, explanation


async def main(llm_response_cache, text_chunk_storage, target_dir):
    qa_filename = f"responses_to_{question_type}_questions_{source_type}_{rag_mode}_{LLM_MODEL_NAME}_{graph_mode}"
    qa_path = os.path.join(target_dir, f"{qa_filename}.jsonl")
    eval_path = os.path.join(target_dir, f"{qa_filename}_eval.jsonl")

    # 检查输入文件是否存在
    if not os.path.exists(qa_path):
        print(f"Warning: Input file not found: {qa_path}")
        return

    with open(qa_path, "r", encoding="utf-8") as rjf, open(eval_path, "a", encoding="utf-8") as wjf:
        for line_idx, line in enumerate(rjf, start=1):
            item = json.loads(line.strip())
            question = item["question"]
            ref_answer = item["reference_answer"]

            # 处理 Origin Chunks (如果存在)
            origin_chunks = item.get("origin_chunks")
            if origin_chunks is not None:
                paragraph = combine_consecutive_overlapping_chunks(origin_chunks)

            new_item = copy.deepcopy(item)
            if "evaluation_results" not in new_item:
                new_item["evaluation_results"] = {}

            args_hash = compute_args_hash(rag_mode, question, LLM_MODEL_NAME)
            mode_cache = await llm_response_cache.get_by_id(rag_mode)

            # 安全获取 prompt，防止 KeyError
            prompt_in_cache = mode_cache.get(args_hash, {}).get("prompt", "")

            # 根据 rag_mode 获取 response
            rag_key_map = {
                "naive": "naive_rag_response",
                "light-direct": "light-direct_rag_response",
                "hgmem": "hgmem_rag_response"
            }

            if rag_mode not in rag_key_map:
                raise ValueError(f"Unknown rag_mode: {rag_mode}")

            response_key = rag_key_map[rag_mode]
            if response_key not in item:
                print(f"Skipping line {line_idx}: {response_key} not found.")
                continue

            response = item[response_key]

            # 初始化 eval result 字典
            eval_key = f"{rag_mode}_rag"
            if eval_key not in new_item["evaluation_results"]:
                new_item["evaluation_results"][eval_key] = {}

            new_item["evaluation_results"][eval_key]["prompt"] = prompt_in_cache

            context_base = dict(
                ref_answer=ref_answer,
                question=question,
                response=response,
            )

            # System message for evaluation
            system_message = "---Role---\nYou are an expert tasked with evaluating a response to the question from a specific aspect by using the relevant paragraph.\n"

            # ---------------------------------------------------------
            # 1. Evaluate Comprehensiveness
            # ---------------------------------------------------------
            use_prompt_comprehensiveness = PROMPTS["evaluate_response_comprehensiveness_2"].format(**context_base)

            try:
                response_evaluate_comp = await openai_complete_if_cache(
                    model=EVAL_MODEL,
                    prompt=use_prompt_comprehensiveness,
                    system_prompt=system_message,
                    api_key=APP_KEY,
                    base_url=BASE_URL,
                    temperature=1.0,
                    max_tokens=1024
                )

                level_comp, score_comp, explanation_comp = post_process_evaluate(response_evaluate_comp)
                new_item["evaluation_results"][eval_key]["comprehensiveness"] = (level_comp, score_comp,
                                                                                 explanation_comp)

            except Exception as e:
                print(f"Error evaluating comprehensiveness at line {line_idx}: {e}")
                new_item["evaluation_results"][eval_key]["comprehensiveness"] = (-1, -1, f"Error: {str(e)}")

            # ---------------------------------------------------------
            # 2. Evaluate Diversity
            # ---------------------------------------------------------
            use_prompt_diversity = PROMPTS["evaluate_response_diversity_2"].format(**context_base)

            try:
                response_evaluate_div = await openai_complete_if_cache(
                    model=EVAL_MODEL,
                    prompt=use_prompt_diversity,
                    system_prompt=system_message,
                    api_key=APP_KEY,
                    base_url=BASE_URL,
                    temperature=1.0,
                    max_tokens=1024
                )

                level_div, score_div, explanation_div = post_process_evaluate(response_evaluate_div)
                new_item["evaluation_results"][eval_key]["diversity"] = (level_div, score_div, explanation_div)

            except Exception as e:
                print(f"Error evaluating diversity at line {line_idx}: {e}")
                new_item["evaluation_results"][eval_key]["diversity"] = (-1, -1, f"Error: {str(e)}")

            # Write to file
            print(f"Processed line {line_idx} -> {eval_path}")
            wjf.write(json.dumps(new_item, ensure_ascii=False) + "\n")
            wjf.flush()  # 确保实时写入


if __name__ == "__main__":
    embed_hidden_size = 1024
    graph_mode = "single"
    question_type = "sampled_difficult"
    source_type = "chunks100"
    rag_mode = "HGMem"
    LLM_MODEL_NAME = "Qwen2.5-32B-Instruct"  # 这是生成 Response 的模型
    DATASET_NAME = "longbench_v2_qa"

    domains = ["Legal"]
    for domain in domains:
        DOMAIN_DIR = os.path.join(PROJ_DIR, f"data/{DATASET_NAME}/domains/{domain}")
        # 增加容错，防止目录不存在报错
        if not os.path.exists(DOMAIN_DIR):
            print(f"Directory not found: {DOMAIN_DIR}")
            continue

        try:
            valid_subdirs = sorted([int(subdir) for subdir in os.listdir(DOMAIN_DIR) if subdir.isdigit()])
        except Exception as e:
            print(f"Error reading subdirectories in {DOMAIN_DIR}: {e}")
            continue

        for i in range(0, 1):  # Processing first subdir only as per original code
            if graph_mode == "merge":
                work_dir = os.path.join(DOMAIN_DIR, f"merged_{i}_top5_similar")
            else:
                work_dir = os.path.join(DOMAIN_DIR, f"{i}")

            target_dir = os.path.join(PROJ_DIR, f"data/created_data/{DATASET_NAME}/{domain}/{i}")
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)  # 使用 makedirs 并开启 exist_ok

            print(f"Working on: {work_dir}")

            try:
                # 假设这些 loader 函数在你本地环境中是可用的
                text_chunk_storage = load_text_chunks(storage_class["JsonKVStorage"], work_dir,
                                                      embed_hidden_size=embed_hidden_size)
                # graph = load_graph(storage_class["NetworkXStorage"], work_dir, embed_hidden_size=embed_hidden_size) # main 中似乎没用到 graph，注释掉节省内存
                llm_response_cache = load_llm_response_cache(storage_class["JsonKVStorage"], work_dir,
                                                             embed_hidden_size=embed_hidden_size)

                loop = asyncio.get_event_loop()
                loop.run_until_complete(main(llm_response_cache, text_chunk_storage, target_dir))
            except Exception as e:
                print(f"Failed to process {work_dir}: {e}")

    print("Finish")
