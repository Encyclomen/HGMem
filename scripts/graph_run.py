import argparse
import copy
import json
import hashlib
import os
import sys

sys_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(sys_dir)
from myrag import MyRAG, QueryParam
from myrag.llm import gpt_4o_mini_complete, hf_model_complete, hf_embedding, ollama_model_complete, vllm_complete
from myrag.utils import EmbeddingFunc, encode_string_by_tiktoken


import torch.cuda
from transformers import AutoModel, AutoTokenizer

PROJ_DIR = sys_dir
DATA_DIR = os.path.join(PROJ_DIR, "data")


os.environ['HF_HOME'] = os.path.join(PROJ_DIR, "cache/")
os.environ['TRANSFORMERS_CACHE'] = os.path.join(PROJ_DIR, "cache/")
os.environ['TIKTOKEN_CACHE_DIR'] = os.path.join(PROJ_DIR, ".tiktoken/")
#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

embed_model_name_or_path = os.path.join(PROJ_DIR, 'cache/bge-m3')
tokenizer = AutoTokenizer.from_pretrained(embed_model_name_or_path)
embed_model = AutoModel.from_pretrained(embed_model_name_or_path)
if torch.cuda.is_available():
    embed_model.cuda()

default_entity_types_of_interest = ["organization", "person", "geo", "event", "role", "concept"]



parser = argparse.ArgumentParser(prog='graphrag_run.py', description='')
#parser.add_argument('domains')
parser.add_argument('--domains', default='Financial,Governmental')
parser.add_argument('--dataset', default='longbench_v2_qa')
parser.add_argument('--graph_mode', default='single')
parser.add_argument('--rag_mode', default='debug')
parser.add_argument('--source_type', default='chunks100')
parser.add_argument('--question_type', default='sampled_difficult')
parser.add_argument('--llm_model_name', default='gpt-4o-mini')
args = parser.parse_args()

graph_mode = args.graph_mode
rag_mode = args.rag_mode
LLM_MODEL_NAME = args.llm_model_name
source_type = args.source_type
question_type = args.question_type
DATASET_NAME = args.dataset
domains = args.domains.split(',')

for domain in domains:
    DOMAIN_DIR = os.path.join(PROJ_DIR, f"data/{DATASET_NAME}/domains/{domain}")
    CREATED_DATA_DIR = os.path.join(PROJ_DIR, f"data/created_data/{DATASET_NAME}/{domain}")
    valid_subdirs = sorted([int(subdir) for subdir in os.listdir(DOMAIN_DIR) if subdir.isdigit()])

    indexed_chunk_statistics = []
    written_failed_questions = set()
    for i in range(0, len(valid_subdirs)):
        if graph_mode == "merge":
            work_dir = os.path.join(DOMAIN_DIR, f"merged_{i}_top5_similar")
        else:
            work_dir = os.path.join(DOMAIN_DIR, f"{i}")
        target_dir = os.path.join(CREATED_DATA_DIR, f"{i}")

        rag = MyRAG(
            working_dir=work_dir,
            embedding_func=EmbeddingFunc(
                embedding_dim=embed_model.config.hidden_size,
                max_token_size=5000,
                func=lambda texts: hf_embedding(
                    texts,
                    tokenizer=tokenizer,
                    embed_model=embed_model
                )
            ),
            llm_model_func=vllm_complete,
            llm_model_name=os.path.join(sys_dir, f"cache/{LLM_MODEL_NAME}"),
            llm_model_kwargs={"max_tokens": 512, "temperature": 0.8, "top_p": 0.8, "extra_body": {"repetition_penalty": 1.05}},
            #llm_model_func=gpt_4o_complete,
            addon_params={"language": "English",
                          "entity_types": default_entity_types_of_interest,
            }
        )

        response_type = "First provide provide your final [Answer], and then provide an [Explaination] of your decision-making process in at most one paragraph. Use the following format:\n [Answer]: YOUR ANSWER \n[Explaination]: YOUR EXPLANATION.\n" \

        if question_type:
            raised_questions_path = os.path.join(target_dir, f"{question_type}_questions_{source_type}.jsonl")
            responses_path = os.path.join(target_dir, f"responses_to_{question_type}_questions_{source_type}_{rag_mode}_{LLM_MODEL_NAME}_{graph_mode}.jsonl")
        else:
            raised_questions_path = os.path.join(target_dir, f"raised_questions.jsonl")
            responses_path = os.path.join(target_dir, f"responses_to_raised_questions_{LLM_MODEL_NAME}_{graph_mode}.jsonl")
        with open(raised_questions_path, "r", encoding="utf-8") as rjf, open(responses_path, "a", encoding="utf-8") as wjf:
            #每个文件只取前两行内容
            for idx, line in enumerate(rjf, start=1):
                # if idx > 2:
                #    continue
                item = json.loads(line.strip())
                new_item = None
                question = item["question"]
                ref_answer = item["reference_answer"]
                involved_entities = item["involved_entities"] if "involved_entities" in item else None
                origin_full_doc_id = item["origin_full_doc_id"] if "origin_full_doc_id" in item else None
                origin_chunks_ids = item["origin_chunks_ids"] if "origin_chunks_ids" in item else None
                origin_chunks_order_index = item["origin_chunks_order_index"] if "origin_chunks_order_index" in item else None
                origin_chunks = item["origin_chunks"] if "origin_chunks" in item else None

                if rag_mode == "naive":
                    """
                    retrieved_chunks_data = rag.indexing(question, query_param=QueryParam(mode="chunk_indexing", top_k=10))
                    retrieved_full_doc_id = []
                    retrieved_chunks_ids = []
                    retrieved_chunks_order_index = []
                    num_hit = 0
                    for chunk_id, chunk_data in retrieved_chunks_data:
                        if "full_doc_id" in chunk_data:
                            retrieved_full_doc_id.append(chunk_data["full_doc_id"])
                        retrieved_chunks_ids.append(chunk_id)
                        retrieved_chunks_order_index.append(chunk_data["chunk_order_index"])
                        if chunk_id in origin_chunks_ids:
                            num_hit += 1
                    indexed_chunk_statistics.append(num_hit / len(origin_chunks_order_index))
                    """
                    naive_rag_response = rag.query(question, query_param=QueryParam(mode="naive", top_k=10,
                                                                                    ignore_cache=True,
                                                                                    save_to_cache=True,
                                                                                    response_type=response_type))
                    new_item = copy.deepcopy(item)
                    new_item["naive_rag_response"] = naive_rag_response
                    #

                elif rag_mode == "light-direct":
                    #retrieved_nodes_data = rag.indexing(question, query_param=QueryParam(mode="node_indexing", top_k=10))
                    light_direct_rag_direct_response = rag.query(question,
                                                                 query_param=QueryParam(mode="light-direct", top_k=10,
                                                                                        max_token_for_global_context=1500,
                                                                                        max_token_for_text_chunks=2000,
                                                                                        ignore_cache=True,
                                                                                        save_to_cache=True,
                                                                                        return_text_chunks_context=True,
                                                                                        response_type=response_type))
                    new_item = copy.deepcopy(item)
                    new_item["light-direct_rag_response"] = light_direct_rag_direct_response
                elif rag_mode == "HGMem":
                    hgmem_rag_response = rag.query(question,
                                                   query_param=QueryParam(mode="HGMem", max_num_turns=3, top_k=10,
                                                                          top_k_entities=20, first_k_entities=5,
                                                                          llm_selection=False, llm_select_k_entities=5,
                                                                          top_k_relationships=5, top_k_chunks=5,
                                                                          max_inner_chunks_per_memory_point=5,
                                                                          max_outer_chunks_per_memory_point=5,
                                                                          max_token_for_global_context=1500,
                                                                          max_token_for_text_chunks=4000,
                                                                          max_token_for_final_text_chunks=10000,
                                                                          ignore_cache=True,
                                                                          save_to_cache=True,
                                                                          return_text_chunks_context=True,
                                                                          response_type=response_type))
                    if hgmem_rag_response == "Failed":
                        print(hgmem_rag_response)
                        # 创建失败问题目录
                        failed_dir = os.path.join(CREATED_DATA_DIR, f"failed_{i}")
                        if not os.path.exists(failed_dir):
                            os.makedirs(failed_dir)
                        failed_file = os.path.join(failed_dir, f"failed_questions_{source_type}.jsonl")

                        question_text = item.get("question", "")  # 根据实际数据结构调整
                        question_hash = hashlib.md5(question_text.encode('utf-8')).hexdigest()

                        if question_hash not in written_failed_questions:
                            with open(failed_file, "a", encoding="utf-8") as fjf:
                                fjf.write(json.dumps(item, ensure_ascii=False) + "\n")
                            written_failed_questions.add(question_hash)  # 添加到已写入集合
                            print(f"Failed question recorded in {failed_file}, skipped.")
                        else:
                            print(f"Question already exists in failed file, skipped: {question_text}")
                    else:
                        new_item = copy.deepcopy(item)
                        new_item["hgmem_rag_response"] = hgmem_rag_response

                print(f"Current target directory: {target_dir}")
                print(f"Current line number: {idx}")
                #如果new_item已经被定义过

                if new_item==None:
                    # print(f"No response generated for line {idx} in {raised_questions_path}. likely skipped because of failure")
                    print("failed")
                else:
                    wjf.write(json.dumps(new_item, ensure_ascii=False) + "\n")

                #
            a = 1