import asyncio
import os
import sys
PROJ_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJ_DIR)

import torch.cuda
from transformers import AutoModel, AutoTokenizer

from myrag.utils import compute_mdhash_id
from scripts.utils import storage_class, load_text_chunks, load_graph, load_entity_vdb, load_relationship_vdb, load_chunk_vdb


async def build_new_entity_vdb_for_graph(graph, entity_vdb):
    def entity_description_func(entity_name, description):
        return f"{entity_name}: {description}"

    all_entities_data = list(await graph.list_all_nodes(data=True))
    data_for_vdb = {}
    for entity in all_entities_data:
        dp = {
            "entity_name": entity[0],
            "description": entity[1]["description"],
            "entity_type": entity[1]["entity_type"],
            "source_id": entity[1]["source_id"]
        }
        key = compute_mdhash_id(dp["entity_name"], prefix="ent-")
        value = {
            "content": entity_description_func(dp["entity_name"], dp["description"]),
            "entity_name": dp["entity_name"],
        }
        data_for_vdb[key] = value

    await entity_vdb.upsert(data_for_vdb)
    await entity_vdb.index_done_callback()


async def build_new_relationship_vdb_for_graph(graph, relationship_vdb):
    def relationship_description_func(keywords, src_id, tgt_id, description):
        return f"{src_id} --&-- {tgt_id} ##  {keywords}\n{description}"

    all_relationships_data = list(await graph.list_all_edges(data=True))
    data_for_vdb = {}
    for relationship in all_relationships_data:
        dp = {
            "src_id": relationship[0],
            "tgt_id": relationship[1],
            "description": relationship[2]["description"],
            "keywords": relationship[2]["keywords"],
            "source_id": relationship[2]["source_id"]
        }
        key = compute_mdhash_id(dp["src_id"]+dp["tgt_id"], prefix="rel-")
        value = {
            "src_id": dp["src_id"],
            "tgt_id": dp["tgt_id"],
            "content": relationship_description_func(dp["keywords"], dp["src_id"], dp["tgt_id"], dp["description"]),
        }
        data_for_vdb[key] = value

    await relationship_vdb.upsert(data_for_vdb)
    await relationship_vdb.index_done_callback()


async def build_new_chunk_vdb_for_graph(text_chunk_storage, chunk_vdb):
    all_chunks_ids = await text_chunk_storage.all_keys()
    all_chunks = [(await text_chunk_storage.get_by_id(c_id))['content'] for c_id in all_chunks_ids]
    all_chunks_kv = {c_id: (await text_chunk_storage.get_by_id(c_id)) for c_id in all_chunks_ids}

    await chunk_vdb.upsert(all_chunks_kv)
    await chunk_vdb.index_done_callback()


async def main(graph, work_dir):
    """
    has_node(self, node_id: str) -> bool
    get_node(self, node_id: str) -> Union[dict, None]
    has_edge(source_node_id: str, target_node_id: str) -> bool
    get_edge(source_node_id: str, target_node_id: str) -> Union[dict, None]
    node_degree(node_id: str) -> int
    edge_degree(src_id: str, tgt_id: str) -> int
    get_node_edges(source_node_id: str)
    get_neighbor_nodes(node_id) -> List(str)
    """

    if not os.path.exists(f"{work_dir}/vdb_entities.json"):
        entity_vdb = load_entity_vdb(storage_class["NanoVectorDBStorage"], entity_vdb_dir=work_dir, tokenizer=tokenizer, embed_model=embed_model)
        await build_new_entity_vdb_for_graph(graph, entity_vdb)
    if not os.path.exists(f"{work_dir}/vdb_relationships.json"):
        relationship_vdb = load_relationship_vdb(storage_class["NanoVectorDBStorage"], relationship_vdb_dir=work_dir, tokenizer=tokenizer, embed_model=embed_model)
        await build_new_relationship_vdb_for_graph(graph, relationship_vdb)
    if not os.path.exists(f"{work_dir}/vdb_chunks.json"):
        text_chunk_storage = load_text_chunks(storage_class["JsonKVStorage"], work_dir, embed_hidden_size=embed_hidden_size)
        chunk_vdb = load_chunk_vdb(storage_class["NanoVectorDBStorage"], chunk_vdb_dir=work_dir, tokenizer=tokenizer, embed_model=embed_model)
        await build_new_chunk_vdb_for_graph(text_chunk_storage, chunk_vdb)


if __name__ == "__main__":
    model_name_or_path = os.path.join(PROJ_DIR, 'cache/bge-m3')
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    embed_model = AutoModel.from_pretrained(model_name_or_path)
    if torch.cuda.is_available():
        embed_model.cuda()
    embed_hidden_size = embed_model.config.hidden_size

    DATASET_NAME = "ultradomain"
    #domains = ["agriculture", "art", "fin", "legal", "mathematics", "neurology", "pathology", "physics", "mix"]
    domains = ["agriculture"]
    #DATASET_NAME = "longbench_v2_qa"
    #domains = ["Academic", "Detective", "Event ordering", "Financial", "Governmental", "Legal", "Literary", "Multi-news"]
    #domains = ["Legal"]
    for domain in domains:
        DOMAIN_DIR = os.path.join(PROJ_DIR, f"data/{DATASET_NAME}/domains/{domain}")
        valid_subdirs = [subdir for subdir in os.listdir(DOMAIN_DIR) if subdir.isdigit()]

        for i in range(len(valid_subdirs)):
        #for i in range(2, 3):
            work_dir = os.path.join(DOMAIN_DIR, f"{i}")
            graph = load_graph(storage_class["NetworkXStorage"], work_dir, embed_hidden_size=embed_hidden_size)
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main(graph, work_dir))

    print("Finish")
