import json
import os
import sys
import argparse
import asyncio
import numpy as np

PROJ_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJ_DIR)

from myrag.storage import JsonKVStorage, NanoVectorDBStorage, NetworkXStorage
from myrag import MyRAG, QueryParam
from myrag.llm import wechat_gpt_complete, hf_embedding, hf_model_complete
from myrag.utils import EmbeddingFunc, truncate_list_by_token_size
from transformers import AutoModel, AutoTokenizer

storage_class = {
    "JsonKVStorage": JsonKVStorage,
    "NanoVectorDBStorage": NanoVectorDBStorage,
    "NetworkXStorage": NetworkXStorage
}

APP_ID = "xxx"
APP_KEY = "xxx"
LLM_MODEL_NAME = "Qwen2.5-7B-Instruct"

DATA_DIR = os.path.join(PROJ_DIR, "data")
os.environ['HF_HOME'] = os.path.join(PROJ_DIR, "cache/")
os.environ['TRANSFORMERS_CACHE'] = os.path.join(PROJ_DIR, "cache/")


def normalize_question_str(q_str, dataset_name):
    if dataset_name == "financial":
        q_str = q_str.replace(
            "You are asked to act as a member of the Financial Results Conference Call and answer the question:", "")
    return q_str


def rag_insert(working_dir, context):
    dummy_embedding_func = EmbeddingFunc(
        embedding_dim=embed_hidden_size,
        max_token_size=5000,
        func=lambda texts: np.array([0])
    )

    default_entity_types_of_interest = ["organization", "person", "geo", "event", "role", "concept"]

    rag = MyRAG(
        working_dir=working_dir,
        embedding_func=dummy_embedding_func,
        llm_model_func=hf_model_complete,
        llm_model_name=os.path.join(PROJ_DIR, f"cache/{LLM_MODEL_NAME}"),
        entity_vector_storage=None,
        relationship_vector_storage=None,
        chunk_vector_storage=None,
        addon_params={"language": "English", "entity_types": default_entity_types_of_interest}
    )
    rag.insert(context)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--source_texts",
        type=str,
        default="prelude",
        help="The source texts to build the graph from.",
    )
    argparser.add_argument(
        "--domains",
        type=str,
        default="general",
        help="The domain of the source texts.",
    )
    args = argparser.parse_args()
    model_name_or_path = os.path.join(PROJ_DIR, 'cache/bge-m3')

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    embed_model = AutoModel.from_pretrained(model_name_or_path)
    embed_hidden_size = embed_model.config.hidden_size

    source_texts = args.source_texts
    domains = args.domains
    BASE_WORKING_DIR = os.path.join(PROJ_DIR, f"data/domains/{domains}/{source_texts}")

    # 创建基础目录
    if not os.path.exists(BASE_WORKING_DIR):
        os.makedirs(BASE_WORKING_DIR)

    # 读取 JSON 文件
    json_file_path = os.path.join(DATA_DIR, f"/{source_texts}.json")

    with open(json_file_path, "r", encoding="utf-8") as rf:
        data = json.load(rf)

    for idx, item in enumerate(data):
        if isinstance(item, list) and len(item) > 0:
            context = item[0] if isinstance(item[0], str) else str(item[0])
        elif isinstance(item, str):
            context = item
        else:
            context = str(item)

        working_dir = os.path.join(BASE_WORKING_DIR, str(idx))
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        print(f"Processing item {idx}...")
        print(f"Context length: {len(context)} characters")

        # 插入到 RAG
        rag_insert(working_dir, context)

        print(f"Finished processing item {idx}, saved to {working_dir}\n")

    print(f"All {len(data)} items processed successfully!")