import numpy as np

from myrag.llm import hf_embedding
from myrag.utils import EmbeddingFunc
from myrag.storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from transformers import AutoModel, AutoTokenizer

storage_class = {
    # kv storage
    "JsonKVStorage": JsonKVStorage,
    # vector storage
    "NanoVectorDBStorage": NanoVectorDBStorage,
    # graph storage
    "NetworkXStorage": NetworkXStorage
}


def load_full_docs(full_docs_class, full_docs_dir, namespace="full_docs", embed_hidden_size=768):
    dummy_node2vec_params = {
        "dimensions": embed_hidden_size,
        "num_walks": 10,
        "walk_length": 40,
        "window_size": 2,
        "iterations": 3,
        "random_seed": 3,
    }
    dummy_embedding_func = EmbeddingFunc(
        embedding_dim=embed_hidden_size,
        max_token_size=5000,
        func=lambda texts: np.array([0])
    )
    global_config = {"working_dir": full_docs_dir, "node2vec_params": dummy_node2vec_params}

    text_chunks = full_docs_class(
        namespace=namespace,
        global_config=global_config,
        embedding_func=dummy_embedding_func,
    )

    return text_chunks


def load_text_chunks(text_chunks_class, text_chunks_dir, namespace="text_chunks", embed_hidden_size=768):
    dummy_node2vec_params = {
        "dimensions": embed_hidden_size,
        "num_walks": 10,
        "walk_length": 40,
        "window_size": 2,
        "iterations": 3,
        "random_seed": 3,
    }
    dummy_embedding_func = EmbeddingFunc(
        embedding_dim=embed_hidden_size,
        max_token_size=5000,
        func=lambda texts: np.array([0])
    )
    global_config = {"working_dir": text_chunks_dir, "node2vec_params": dummy_node2vec_params}

    text_chunks = text_chunks_class(
        namespace=namespace,
        global_config=global_config,
        embedding_func=dummy_embedding_func,
    )

    return text_chunks


def load_llm_response_cache(llm_response_cache_class, llm_response_cache_dir, namespace="llm_response_cache", embed_hidden_size=768):
    dummy_node2vec_params = {
        "dimensions": embed_hidden_size,
        "num_walks": 10,
        "walk_length": 40,
        "window_size": 2,
        "iterations": 3,
        "random_seed": 3,
    }
    dummy_embedding_func = EmbeddingFunc(
        embedding_dim=embed_hidden_size,
        max_token_size=5000,
        func=lambda texts: np.array([0])
    )
    global_config = {"working_dir": llm_response_cache_dir, "node2vec_params": dummy_node2vec_params}

    text_chunks = llm_response_cache_class(
        namespace=namespace,
        global_config=global_config,
        embedding_func=dummy_embedding_func,
    )

    return text_chunks


def load_graph(graph_class, graph_dir, namespace="chunk_entity_relation", embed_hidden_size=768):
    dummy_node2vec_params = {
            "dimensions": embed_hidden_size,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
    }
    dummy_embedding_func = EmbeddingFunc(
        embedding_dim=embed_hidden_size,
        max_token_size=5000,
        func=lambda texts: np.array([0])
    )
    global_config = {"working_dir": graph_dir, "node2vec_params": dummy_node2vec_params}

    graph_instance = graph_class(
        namespace=namespace,
        global_config=global_config,
        embedding_func=dummy_embedding_func,
    )

    return graph_instance


def load_entity_vdb(entity_vdb_class, entity_vdb_dir, model_name_or_path=None, tokenizer=None, embed_model=None, meta_fields=None, namespace="entities"):
    if model_name_or_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        embed_model = AutoModel.from_pretrained(model_name_or_path)
    else:
        assert tokenizer is not None and embed_model is not None
    embed_hidden_size = embed_model.config.hidden_size
    global_config = {"working_dir": entity_vdb_dir, "embedding_batch_num": 2, "cosine_better_than_threshold": 0.2}

    if meta_fields is None:
        meta_fields = {"entity_name"}
    entity_vdb = entity_vdb_class(
        namespace=namespace,
        embedding_func=EmbeddingFunc(
            embedding_dim=embed_hidden_size,
            max_token_size=5000,
            func=lambda texts: hf_embedding(
                texts,
                tokenizer=tokenizer,
                embed_model=embed_model
            )
        ),
        global_config=global_config,
        meta_fields=meta_fields
    )

    return entity_vdb


def load_relationship_vdb(relationship_vdb_class, relationship_vdb_dir, model_name_or_path=None, tokenizer=None, embed_model=None, meta_fields=None, namespace="relationships"):
    if model_name_or_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        embed_model = AutoModel.from_pretrained(model_name_or_path)
    else:
        assert tokenizer is not None and embed_model is not None
    embed_hidden_size = embed_model.config.hidden_size
    global_config = {"working_dir": relationship_vdb_dir, "embedding_batch_num": 2, "cosine_better_than_threshold": 0.2}

    if meta_fields is None:
        meta_fields = {"src_id", "tgt_id"}
    relationship_vdb = relationship_vdb_class(
        namespace=namespace,
        embedding_func=EmbeddingFunc(
            embedding_dim=embed_hidden_size,
            max_token_size=5000,
            func=lambda texts: hf_embedding(
                texts,
                tokenizer=tokenizer,
                embed_model=embed_model
            )
        ),
        global_config=global_config,
        meta_fields=meta_fields
    )

    return relationship_vdb


def load_chunk_vdb(chunk_vdb_class, chunk_vdb_dir, model_name_or_path=None, tokenizer=None, embed_model=None, namespace="chunks"):
    if model_name_or_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        embed_model = AutoModel.from_pretrained(model_name_or_path)
    else:
        assert tokenizer is not None and embed_model is not None
    embed_hidden_size = embed_model.config.hidden_size
    global_config = {"working_dir": chunk_vdb_dir, "embedding_batch_num": 2, "cosine_better_than_threshold": 0.2}

    chunk_vdb = chunk_vdb_class(
        namespace=namespace,
        embedding_func=EmbeddingFunc(
            embedding_dim=embed_hidden_size,
            max_token_size=5000,
            func=lambda texts: hf_embedding(
                texts,
                tokenizer=tokenizer,
                embed_model=embed_model
            )
        ),
        global_config=global_config
    )

    return chunk_vdb


def combine_consecutive_overlapping_chunks(consecutive_overlapping_chunks):
    if len(consecutive_overlapping_chunks) == 0:
        return ""
    elif len(consecutive_overlapping_chunks) == 1:
        return consecutive_overlapping_chunks[0]

    full = consecutive_overlapping_chunks[0]
    for i in range(1, len(consecutive_overlapping_chunks)):
        prev, cur = consecutive_overlapping_chunks[i-1], consecutive_overlapping_chunks[i]
        j = 0
        for j in range(len(cur), 0, -1):
            if cur[:j] == prev[len(prev)-j:]:
                break
        full += cur[j:]

    return full


if __name__ == "__main__":
    import os
    import asyncio

    async def test():
        pass

    PROJ_DIR = "/Users/zhouchulun/Documents/Projects/RAG"
    # model_name_or_path = os.path.join(PROJ_DIR, 'cache/Qwen2-1.5B-Instruct')
    model_name_or_path = os.path.join(PROJ_DIR, 'cache/xlm-roberta-base')
    # model_name_or_path = "jinaai/jina-embeddings-v3"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # embed_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
    embed_model = AutoModel.from_pretrained(model_name_or_path)
    embed_hidden_size = embed_model.config.hidden_size

    TAG = "from_yifan/fin/0"
    #TAG = "from_yifan/legal/0"
    work_dir = os.path.join(PROJ_DIR, f"graphs/graph_{TAG}")
    created_data_dir = os.path.join(PROJ_DIR, f"data/created_data/{TAG}")

    text_chunk_storage = load_text_chunks(storage_class["JsonKVStorage"], work_dir, embed_hidden_size=embed_hidden_size)
    #graph = load_graph(storage_class["NetworkXStorage"], work_dir, embed_hidden_size=embed_hidden_size)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(test())