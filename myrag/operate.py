import asyncio
import re
from tqdm.asyncio import tqdm as tqdm_async
from typing import Union
from collections import Counter, defaultdict
import warnings

from .memory import Memory
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData, extract_llm_model_name,
    build_entities_context, build_relationships_context, build_text_chunks_context,
    truncate_list_by_token_size, truncate_attribute_by_token_size,
    convert_to_entity_vdb_ids_dict, convert_to_relationships_vdb_ids_dict,
)
from .base import (
    BaseKVStorage,
    BaseVectorStorage,
    BaseGraphStorage,
    BaseHypergraphStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from .prompt_compare import PROMPTS_COMPARE


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    entity_extraction_config = global_config["config"]["ENTITY_EXTRACTION"]
    entity_summary_max_tokens = global_config["addon_params"].get("entity_summary_max_tokens", entity_extraction_config["entity_summary_max_tokens"])
    language = global_config["addon_params"].get("language", entity_extraction_config["DEFAULT_LANGUAGE"])

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < entity_summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=entity_summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str = None,
):
    if len(record_attributes) < 4 or record_attributes[0] != 'entity':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key if chunk_key is not None else ""
    entity_dict = {"entity_name": entity_name, "entity_type": entity_type, "description": entity_description, "source_id": entity_source_id}
    return entity_dict


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str = None,
):
    if len(record_attributes) < 4 or record_attributes[0] != 'relationship':
        return None

    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    relationship_description = clean_str(record_attributes[3])
    relationship_source_id = chunk_key if chunk_key is not None else ""
    relationship_dict = {"src_id": source, "tgt_id": target, "description": relationship_description, "source_id": relationship_source_id}
    if len(record_attributes) >= 5:
        relationship_dict["keywords"] = clean_str(record_attributes[4])
    if len(record_attributes) >= 6:
        relationship_dict["weight"] = (
            float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
        )
    return relationship_dict


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entity_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([str(dp["source_id"]) for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_keywords.extend(
            split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    keywords = GRAPH_FIELD_SEP.join(
        sorted(set([dp["keywords"] for dp in edges_data] + already_keywords))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([str(dp["source_id"]) for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        f"({src_id}, {tgt_id})", description, global_config
    )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
    )

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extraction_config = global_config["config"]["ENTITY_EXTRACTION"]

    # add language and example number params to prompt
    language = global_config["addon_params"].get("language", entity_extraction_config["DEFAULT_LANGUAGE"])
    entity_types = global_config["addon_params"].get("entity_types", entity_extraction_config["DEFAULT_ENTITY_TYPES"])
    tuple_delimiter = global_config["addon_params"].get("tuple_delimiter", entity_extraction_config["DEFAULT_TUPLE_DELIMITER"])
    record_delimiter = global_config["addon_params"].get("record_delimiter", entity_extraction_config["DEFAULT_RECORD_DELIMITER"])
    completion_delimiter = global_config["addon_params"].get("completion_delimiter", entity_extraction_config["DEFAULT_COMPLETION_DELIMITER"])
    max_extra_entity_gleaning = global_config["addon_params"].get("max_extra_entity_gleaning", entity_extraction_config["max_extra_entity_gleaning"])
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        entity_extraction_examples_template = "\n".join(
            PROMPTS["entity_extraction_examples"][: int(example_number)]
        )
    else:
        entity_extraction_examples_template = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=tuple_delimiter,
        record_delimiter=record_delimiter,
        completion_delimiter=completion_delimiter,
        entity_types=",".join(entity_types),
        language=language,
    )
    # add example's format
    entity_extraction_examples = entity_extraction_examples_template.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=tuple_delimiter,
        record_delimiter=record_delimiter,
        completion_delimiter=completion_delimiter,
        entity_types=",".join(entity_types),
        examples=entity_extraction_examples,
        language=language,
    )

    continue_prompt = PROMPTS["entity_continue_extraction"]
    if_loop_prompt = PROMPTS["entity_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        # hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        hint_prompt = entity_extract_prompt.format(**context_base, input_text="{input_text}").format(**context_base, input_text=content)

        final_result = await use_llm_func(hint_prompt)
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for num_already_entity_gleaning in range(max_extra_entity_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)
            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if num_already_entity_gleaning == max_extra_entity_gleaning - 1:
                break
            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        # Here, split_string_by_multi_markers is necessary for dealing with multi-turn entity gleaning.
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entity = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entity is not None:
                maybe_nodes[if_entity["entity_name"]].append(if_entity)
                continue

            if_relationship = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relationship is not None:
                maybe_edges[(if_relationship["src_id"], if_relationship["tgt_id"])].append(
                    if_relationship
                )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = entity_extraction_config["process_tickers"][
            already_processed % len(entity_extraction_config["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    #"""
    ordered_chunks = list(chunks.items())
    results = []
    for result in tqdm_async(
        asyncio.as_completed([_process_single_content(c) for c in ordered_chunks]),
        total=len(ordered_chunks),
        desc="Extracting entities from chunks",
        unit="chunk",
    ):
        results.append(await result)
    #"""
    """
    results = []
    with open(os.path.join(global_config["working_dir"], "extract_entity_results.json"), "r") as rf:
        extract_entity_results = json.load(rf)
        for item in extract_entity_results:
            entity_dict = {k.replace("\"", "").replace("“", "").replace("”", ""): v for k, v in item[0].items()}
            relation_dict = {k.replace("\"", "").replace("“", "").replace("”", ""): v for k, v in item[1].items()}
            new_relation_dict = {tuple(k.split("----")): v for k, v in relation_dict.items()}
            results.append((entity_dict, new_relation_dict))
    """

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)
    logger.info("Inserting entities into storage...")
    all_entities_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        ),
        total=len(maybe_nodes),
        desc="Inserting entities",
        unit="entity",
    ):
        all_entities_data.append(await result)

    logger.info("Inserting relationships into storage...")
    all_relationships_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_edges_then_upsert(
                    k[0], k[1], v, knowledge_graph_inst, global_config
                )
                for k, v in maybe_edges.items()
            ]
        ),
        total=len(maybe_edges),
        desc="Inserting relationships",
        unit="relationship",
    ):
        all_relationships_data.append(await result)

    if not len(all_entities_data) and not len(all_relationships_data):
        logger.warning(
            "Didn't extract any entities and relationships, maybe your LLM is not working"
        )
        return None

    if not len(all_entities_data):
        logger.warning("Didn't extract any entities")
    if not len(all_relationships_data):
        logger.warning("Didn't extract any relationships")

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": f"{dp['entity_name']}: {dp['description']}",
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": f"{dp['keywords']} --&-- {dp['src_id']} ##  {dp['tgt_id']}\n{dp['description']}",
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst


def postprocess_select_entities(response):
    loc_selected_entities = response.find("[Selected]:")
    assert loc_selected_entities != -1, f"Format error: {response}"
    selected_entities_str = response[loc_selected_entities:].replace("[Selected]:", "").strip()
    if "<None>" in selected_entities_str:
        return []
    else:
        selected_entities_indices = [int(index_str) for index_str in selected_entities_str.replace(" ", "").split(",")]
        return selected_entities_indices


async def _build_local_query_context(
    queries,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    relationships_vdb: BaseVectorStorage = None,
    text_chunks_vdb: BaseVectorStorage = None,
    filter_lambda=None,
    verbose: bool = False,
    passage_retrieval: bool = False,
    llm_model_func=None,
    memory=None
):
    all_entities_data, all_entities_ids = [], []
    all_relationships_data, all_relationships_ids = [], []
    all_text_chunks, all_text_chunks_ids = [], []
    all_entities_context, all_relationships_context, all_text_chunks_context = [], [], []
    if type(queries) == str:
        queries = [queries]
    for q_id, query in enumerate(queries):
        if not query:
            warnings.warn(
                f"Query {q_id} is None. Return empty entity/relationship/source"
            )
            continue
        candidate_entities_data, candidate_entities_ids = await _get_entities_from_query(query, knowledge_graph_inst, entities_vdb, query_param, filter_lambda=filter_lambda, return_entities_ids=True)
        entities_data = candidate_entities_data[:query_param.first_k_entities]
        entities_ids = candidate_entities_ids[:query_param.first_k_entities]

        if query_param.llm_selection:
            assert llm_model_func is not None

            entities_details = []
            for entity_idx, entity in enumerate(candidate_entities_data[query_param.first_k_entities:], start=0):
                entity_str = f"- ({entity_idx}): {entity['entity_name']}\nDescription: {entity['description']}"
                entities_details.append(entity_str)
            entities_str = "\n".join(entities_details)

            context_base = dict(
                query=query,
                memory=await memory.get_memory_points_context(),
                entities_str=entities_str
            )
            select_entities_prompt = PROMPTS["select_entities"]
            select_entities_prompt = select_entities_prompt.format(**context_base)
            response_select_entities = await llm_model_func(select_entities_prompt)
            selected_entities_indices = postprocess_select_entities(response_select_entities)
            llm_selected_entities_data = [candidate_entities_data[query_param.first_k_entities:][index] for index in selected_entities_indices]
            llm_selected_entities_ids = [candidate_entities_ids[query_param.first_k_entities:][index] for index in selected_entities_indices]
            if query_param.llm_select_k_entities >= 0:
                llm_selected_entities_data = llm_selected_entities_data[:query_param.llm_select_k_entities]
                llm_selected_entities_ids = llm_selected_entities_ids[:query_param.llm_select_k_entities]
            entities_data.extend(llm_selected_entities_data)
            entities_ids.extend(llm_selected_entities_ids)
        all_entities_data.append(entities_data)
        all_entities_ids.append(entities_ids)

        if len(entities_data) == 0:
            logger.warning(f"Empty entities_data obtained from query {query}.")
        if not all([n is not None for n in entities_data]):
            logger.warning("Some nodes are missing, maybe the storage is damaged.")
        # build entity context
        entities_context = build_entities_context(entities_data)
        all_entities_context.append(entities_context)
        # get related edges from above entities
        relationships_data, relationships_ids = await _find_related_edges_from_entities(
            entities_data, query_param, knowledge_graph_inst, query=query,
            relationships_vdb=relationships_vdb, return_relationships_ids=True
        )
        all_relationships_data.append(relationships_data)
        all_relationships_ids.append(relationships_ids)
        # build relation context
        relationships_context = build_relationships_context(relationships_data)
        all_relationships_context.append(relationships_context)
        logger.info(f"Query {q_id} uses {len(entities_data)} entities, {len(relationships_data)} relations.")

        if query_param.return_text_chunks_context:
            # build entity-related text chunks context
            text_chunks, text_chunks_ids = await _find_related_text_chunks_from_entities(
                entities_data, query_param, text_chunks_db, knowledge_graph_inst,
                query=query, text_chunks_vdb=text_chunks_vdb, return_text_chunks_ids=True
            )
            logger.info(
                f"Nodes are associated with {len(text_chunks)} text chunks"
            )

            if passage_retrieval:
                vanilla_text_chunks = await _find_text_chunks_from_query(query, text_chunks_vdb, text_chunks_db, query_param)
                for chunk in vanilla_text_chunks:
                    if chunk["id"] not in text_chunks_ids:
                        text_chunks_ids.append(chunk["id"])
                        text_chunks.append(chunk)
            sorted_text_chunks_and_ids = sorted(list(zip(text_chunks, text_chunks_ids)), key=lambda x: (x[0].get("full_doc_id", ""), x[0]["chunk_order_index"]))
            text_chunks = [pair[0] for pair in sorted_text_chunks_and_ids]
            text_chunks_ids = [pair[1] for pair in sorted_text_chunks_and_ids]
            text_chunks_context = build_text_chunks_context(text_chunks)

            all_text_chunks.append(text_chunks)
            all_text_chunks_ids.append(text_chunks_ids)
            all_text_chunks_context.append(text_chunks_context)

    unique_entities_data = []
    unique_entities_ids = []
    for ent_datas, ent_ids in zip(all_entities_data, all_entities_ids):
        assert len(ent_datas) == len(ent_ids)
        for ent_data, ent_id in zip(ent_datas, ent_ids):
            if ent_id not in unique_entities_ids:
                unique_entities_data.append(ent_data)
                unique_entities_ids.append(ent_id)
    final_entities_context = build_entities_context(unique_entities_data)

    unique_relationships_data = []
    unique_relationships_ids = []
    for rel_datas, rel_ids in zip(all_relationships_data, all_relationships_ids):
        assert len(rel_datas) == len(rel_ids)
        for rel_data, rel_id in zip(rel_datas, rel_ids):
            if rel_id not in unique_relationships_ids:
                unique_relationships_data.append(rel_data)
                unique_relationships_ids.append(rel_id)
    final_relationships_context = build_relationships_context(unique_relationships_data)

    built_context = f"""-----Entities-----\n""" \
                    f"""```csv\n{final_entities_context}\n```\n""" \
                    f"""-----Relationships-----\n""" \
                    f"""```csv\n""" \
                    f"""{final_relationships_context}\n""" \
                    f"""```\n"""

    if query_param.return_text_chunks_context:
        unique_text_chunks = []
        unique_text_chunks_ids = []
        for chunks, chunks_ids in zip(all_text_chunks, all_text_chunks_ids):
            assert len(chunks) == len(chunks_ids)
            for chunk, chunk_id in zip(chunks, chunks_ids):
                if chunk_id not in unique_text_chunks_ids:
                    unique_text_chunks.append(chunk)
                    unique_text_chunks_ids.append(chunk_id)
        sorted_unique_text_chunks_and_ids = sorted(list(zip(unique_text_chunks, unique_text_chunks_ids)), key=lambda x: (x[0].get("full_doc_id", ""), x[0]["chunk_order_index"]))
        unique_text_chunks = [pair[0] for pair in sorted_unique_text_chunks_and_ids]
        unique_text_chunks_ids = [pair[1] for pair in sorted_unique_text_chunks_and_ids]

        final_text_chunks_context = build_text_chunks_context(unique_text_chunks)

        built_context += f"""-----Sources-----\n""" \
                         f"""```csv\n""" \
                         f"""{final_text_chunks_context}\n""" \
                         f"""```\n"""
    if verbose:
        return built_context, {"entities_ids": all_entities_ids, "relationships_ids": all_relationships_ids, "text_chunks_ids": all_text_chunks_ids}
    
    return built_context


async def _build_global_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    entities_vdb: BaseVectorStorage = None,
    text_chunks_vdb: BaseVectorStorage = None,
):
    results = await relationships_vdb.query(query, top_k=query_param.top_k)

    if not len(results):
        return None

    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    edge_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"]) for r in results]
    )
    edge_datas = [
        {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
        for k, v, d in zip(results, edge_datas, edge_degree)
        if v is not None
    ]
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x.get("weight", 0)), reverse=True
    )
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    relations_context = build_relationships_context(edge_datas)

    use_entities = await _find_related_entities_from_relationships(
        edge_datas, query_param, knowledge_graph_inst, query=query, entities_vdb=entities_vdb
    )
    entities_context = build_entities_context(use_entities)

    built_context = f"""-----Entities-----\n""" \
                    f"""```csv\n{entities_context}\n```\n""" \
                    f"""-----Relationships-----\n""" \
                    f"""```csv\n""" \
                    f"""{relations_context}\n""" \
                    f"""```\n"""
    logger.info(f"Global query uses {len(use_entities)} entities, {len(edge_datas)} relations")

    if query_param.return_text_chunks_context:
        use_text_chunks = await _find_related_text_chunks_from_relationships(
            edge_datas, query_param, text_chunks_db, knowledge_graph_inst, query=query, text_chunks_vdb=text_chunks_vdb,
        )
        logger.info(f"Global query also uses {len(use_text_chunks)} text chunks")

        text_chunks_context = build_text_chunks_context(use_text_chunks)
        built_context += f"""-----Sources-----\n""" \
                         f"""```csv\n""" \
                         f"""{text_chunks_context}\n""" \
                         f"""```\n"""

    return built_context


async def _find_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    query: str = None,
    relationships_vdb: BaseVectorStorage = None,
    return_relationships_ids: bool = False
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = []
    seen = set()

    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    if query is not None and relationships_vdb is not None:
        relationships_vdb_ids_dict = convert_to_relationships_vdb_ids_dict(all_edges)
        if relationships_vdb_ids_dict:
            filter_lambda = lambda data: data["__id__"] in relationships_vdb_ids_dict
            filtered_edges = await relationships_vdb.query(query, top_k=query_param.top_k_relationships, filter_lambda=filter_lambda)
        else:
            filtered_edges = []
        all_edges = [(edge["src_id"], edge["tgt_id"]) for edge in filtered_edges]

    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_id": k[0], "tgt_id": k[1], "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]

    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x.get("weight", 0)), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    truncate_attribute_by_token_size(all_edges_data, attribute="description", max_token_size=query_param.max_token_for_relationship_description)
    
    if return_relationships_ids:
        all_relationships_ids = [(e["src_id"], e["tgt_id"]) for e in all_edges_data]
        return all_edges_data, all_relationships_ids 
    
    return all_edges_data


async def _find_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    query: str = None,
    entities_vdb: BaseVectorStorage = None,
):
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    if query is not None and entities_vdb is not None:
        entity_vdb_ids_dict = convert_to_entity_vdb_ids_dict(entity_names)
        filter_lambda = lambda data: data["__id__"] in entity_vdb_ids_dict
        filtered_entities = await entities_vdb.query(query, top_k=query_param.top_k_relationships, filter_lambda=filter_lambda)
        entity_names = [entity["entity_name"] for entity in filtered_entities]
        a=1

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names]
    )

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )
    truncate_attribute_by_token_size(node_datas, attribute="description", max_token_size=query_param.max_token_for_entity_description)

    return node_datas


async def _find_related_text_chunks_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    query: str = None,
    text_chunks_vdb: BaseVectorStorage = None,
    record_relation_counts: bool = False,
    return_text_chunks_ids: bool = False
):
    text_chunks = [
        split_string_by_multi_markers(str(dp["source_id"]), [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]

    edges = None
    if record_relation_counts:
        edges = await asyncio.gather(
            *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
        )
        all_one_hop_nodes = set()
        for this_edges in edges:
            if not this_edges:
                continue
            all_one_hop_nodes.update([e[1] for e in this_edges])

        all_one_hop_nodes = list(all_one_hop_nodes)
        all_one_hop_nodes_data = await asyncio.gather(
            *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
        )

        # Add null check for node data
        all_one_hop_text_chunks_lookup = {
            k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
            for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
            if v is not None and "source_id" in v  # Add source_id check
        }

    all_text_chunks_lookup = {}
    for index, this_text_chunks in enumerate(text_chunks):
        for c_id in this_text_chunks:
            if c_id not in all_text_chunks_lookup:
                all_text_chunks_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                }
            if edges is not None and edges[index]:
                for e in edges[index]:
                    if (
                            e[1] in all_one_hop_text_chunks_lookup
                            and c_id in all_one_hop_text_chunks_lookup[e[1]]
                    ):
                        if "relation_counts" not in all_text_chunks_lookup[c_id]:
                            all_text_chunks_lookup[c_id]["relation_counts"] = 0
                        else:
                            all_text_chunks_lookup[c_id]["relation_counts"] += 1
    """
    for index, (this_text_chunks, this_edges) in enumerate(zip(text_chunks, edges)):
        for c_id in this_text_chunks:
            if c_id not in all_text_chunks_lookup:
                all_text_chunks_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                }

            if record_relation_counts and this_edges:
                for e in this_edges:
                    if (
                        e[1] in all_one_hop_text_chunks_lookup
                        and c_id in all_one_hop_text_chunks_lookup[e[1]]
                    ):
                        if "relation_counts" not in all_text_chunks_lookup[c_id]:
                            all_text_chunks_lookup[c_id]["relation_counts"] = 0
                        else:
                            all_text_chunks_lookup[c_id]["relation_counts"] += 1
    """

    if query is not None and text_chunks_vdb is not None:
        text_chunks_vdb_ids_dict = {e: i for i, e in enumerate(set(list(all_text_chunks_lookup.keys())))}
        filter_lambda = lambda data: data["__id__"] in text_chunks_vdb_ids_dict
        filtered_text_chunks = await text_chunks_vdb.query(query, top_k=query_param.top_k_chunks, filter_lambda=filter_lambda)
        all_text_chunks_lookup = {
            text_chunk["id"]: {"distance": text_chunk["distance"], **all_text_chunks_lookup[text_chunk["id"]]}
            for text_chunk in filtered_text_chunks
        }
        a=1

    # Filter out None values and ensure data has content
    valid_text_chunks = [
        {"id": k, **v}
        for k, v in all_text_chunks_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not valid_text_chunks:
        logger.warning("No valid text chunk found")
        return []

    valid_text_chunks = sorted(
        #valid_text_chunks, key=lambda x: (x["order"], -x["relation_counts"])
        valid_text_chunks, key=lambda x: (x["data"].get("full_doc_id", ""), x["data"]["chunk_order_index"])
        #valid_text_chunks, key=lambda x: (-x["distance"])
    )

    truncated_valid_text_chunks = truncate_list_by_token_size(
        valid_text_chunks,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_chunks,
    )

    all_text_chunks: list[TextChunkSchema] = [t["data"] for t in truncated_valid_text_chunks]
    if return_text_chunks_ids:
        all_text_chunks_ids: list[TextChunkSchema] = [t["id"] for t in truncated_valid_text_chunks]
        return all_text_chunks, all_text_chunks_ids
    else:
        return all_text_chunks


async def _find_related_text_chunks_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    query: str = None,
    text_chunks_vdb: BaseVectorStorage = None,
):
    text_chunks = [
        split_string_by_multi_markers(str(dp["source_id"]), [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]
    all_text_chunks_lookup = {}
    for index, chunk_list in enumerate(text_chunks):
        for c_id in chunk_list:
            if c_id not in all_text_chunks_lookup:
                chunk_data = await text_chunks_db.get_by_id(c_id)
                # Only store valid data
                if chunk_data is not None and "content" in chunk_data:
                    all_text_chunks_lookup[c_id] = {
                        "data": chunk_data,
                        "order": index,
                    }

    if not all_text_chunks_lookup:
        logger.warning("No valid text chunks found")
        return []

    if query is not None and text_chunks_vdb is not None:
        text_chunks_vdb_ids_dict = {e: i for i, e in enumerate(set(list(all_text_chunks_lookup.keys())))}

        filter_lambda = lambda data: data["__id__"] in text_chunks_vdb_ids_dict
        filtered_text_chunks = await text_chunks_vdb.query(query, top_k=query_param.top_k_chunks, filter_lambda=filter_lambda)
        all_text_chunks_lookup = {text_chunk["id"]: {"distance": text_chunk["distance"], **all_text_chunks_lookup[text_chunk["id"]]} for text_chunk in filtered_text_chunks}

    # Filter out None values and ensure data has content
    valid_text_chunks = [
        {"id": k, **v}
        for k, v in all_text_chunks_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not valid_text_chunks:
        logger.warning("No valid text chunks found")
        return []

    valid_text_chunks = sorted(
        # valid_text_chunks, key=lambda x: (x["order"], -x["relation_counts"])
        valid_text_chunks, key=lambda x: (x["data"].get("full_doc_id", ""), x["data"]["chunk_order_index"])
        # valid_text_chunks, key=lambda x: (-x["distance"])
    )

    truncated_valid_text_chunks = truncate_list_by_token_size(
        valid_text_chunks,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_chunks,
    )

    all_text_chunks: list[TextChunkSchema] = [t["data"] for t in truncated_valid_text_chunks]

    return all_text_chunks


async def _get_node_data(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    # get similar entities
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return "", "", ""
    # get entity information
    nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in nodes_data]):
        logger.warning("Some nodes are missing, maybe the storage is damaged.")

    # get entity degree
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    nodes_data = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, nodes_data, node_degrees)
        if n is not None
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
    truncate_attribute_by_token_size(nodes_data, attribute="description", max_token_size=query_param.max_token_for_entity_description)

    # get entitytext chunk
    use_text_chunks = await _find_related_text_chunks_from_entities(
        nodes_data, query_param, text_chunks_db, knowledge_graph_inst
    )
    entities_context = build_entities_context(nodes_data)
    # get relate edges
    use_relations = await _find_related_edges_from_entities(
        nodes_data, query_param, knowledge_graph_inst
    )
    truncate_attribute_by_token_size(use_relations, attribute="description", max_token_size=query_param.max_token_for_relationship_description)
    relations_context = build_relationships_context(use_relations)
    logger.info(
        f"Query uses {len(nodes_data)} entities, {len(use_relations)} relations, {len(use_text_chunks)} text chunks"
    )

    text_chunks_context = build_text_chunks_context(use_text_chunks)

    return entities_context, relations_context, text_chunks_context


def combine_contexts(high_level_context, low_level_context, include_sources=False):
    # Function to extract entities, relationships, and sources from context strings
    def extract_sections(context):
        entities_match = re.search(
            r"-----Entities-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )
        relationships_match = re.search(
            r"-----Relationships-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )
        sources_match = re.search(
            r"-----Sources-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )

        entities = entities_match.group(1) if entities_match else ""
        relationships = relationships_match.group(1) if relationships_match else ""
        sources = sources_match.group(1) if sources_match else ""

        return entities, relationships, sources

    # Extract sections from both contexts
    if high_level_context is None:
        warnings.warn(
            "High Level context is None. Return empty High_Level entity/relationship/source"
        )
        hl_entities, hl_relationships, hl_sources = "", "", ""
    else:
        hl_entities, hl_relationships, hl_sources = extract_sections(high_level_context)

    if low_level_context is None:
        warnings.warn(
            "Low Level context is None. Return empty Low_Level entity/relationship/source"
        )
        ll_entities, ll_relationships, ll_sources = "", "", ""
    else:
        ll_entities, ll_relationships, ll_sources = extract_sections(low_level_context)

    # Combine and deduplicate the entities
    combined_entities = process_combine_contexts(hl_entities, ll_entities)
    # Combine and deduplicate the relationships
    combined_relationships = process_combine_contexts(hl_relationships, ll_relationships)
    # Combine and deduplicate the sources
    combined_sources = process_combine_contexts(hl_sources, ll_sources) if include_sources else None

    return combined_entities, combined_relationships, combined_sources


async def _get_entities_from_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    query_param: QueryParam,
    filter_lambda=None,
    return_entities_ids=False,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k_entities, filter_lambda=filter_lambda)
    if not len(results):
        return []
    #get entity information
    entities_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in entities_data]):
        logger.warning("Some nodes are missing, maybe the storage is damaged.")

    if query_param.include_node_degree:
        # get entity degree
        node_degrees = await asyncio.gather(
            *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
        )
        entities_data = [
            {**n, "entity_name": k["entity_name"], "rank": d}
            for k, n, d in zip(results, entities_data, node_degrees)
            if n is not None
        ]
    else:
        entities_data = [
            {**n, "entity_name": k["entity_name"], "rank": "UNKNOWN"}
            for k, n in zip(results, entities_data)
            if n is not None
        ]
    truncate_attribute_by_token_size(entities_data, attribute="description", max_token_size=query_param.max_token_for_entity_description)
    
    if return_entities_ids:
        entities_names = [n["entity_name"] for n in entities_data]
        return entities_data, entities_names
    
    return entities_data


async def node_indexing(
    query,
    entities_vdb: BaseVectorStorage,
    knowledge_graph_inst: BaseGraphStorage,
    query_param: QueryParam,
):
    # Set mode
    if query_param.mode != "node_indexing":
        logger.error(f"Incorrect mode {query_param.mode} in node_indexing.")
        return "Fail to conduct node_indexing"

    indexed_nodes_data = await _get_entities_from_query(query, knowledge_graph_inst, entities_vdb, query_param)

    return indexed_nodes_data


async def chunk_indexing(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await chunks_vdb.query(query, top_k=query_param.top_k)

    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    # Filter out invalid chunks
    valid_chunks = [
        (chunk_id, chunk) for chunk_id, chunk in zip(chunks_ids, chunks) if chunk is not None and "content" in chunk
    ]

    return valid_chunks


async def _find_text_chunks_from_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await chunks_vdb.query(query, top_k=query_param.top_k_chunks)

    chunks = [{"id": r["id"], **(await text_chunks_db.get_by_id(r["id"]))} for r in results]

    # Filter out invalid chunks
    valid_chunks = [
        chunk for chunk in chunks if chunk is not None and "content" in chunk
    ]

    if not valid_chunks:
        logger.warning("No valid chunks found after filtering")

    maybe_trunc_chunks = truncate_list_by_token_size(
        valid_chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_chunks,
    )

    maybe_trunc_chunks = sorted(
        maybe_trunc_chunks, key=lambda x: (x.get("full_doc_id", ""), x["chunk_order_index"])
    )
    return maybe_trunc_chunks


async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
):
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    llm_model_name = extract_llm_model_name(global_config["llm_model_name"])
    args_hash = compute_args_hash(query_param.mode, query, llm_model_name)
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode
    )
    if (not query_param.ignore_cache) and cached_response is not None:
        return cached_response

    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]

    #chunks_ids = [r["id"] for r in results]
    #chunks = await text_chunks_db.get_by_ids(chunks_ids)
    chunks = [{"id": r["id"], **(await text_chunks_db.get_by_id(r["id"]))} for r in results]

    # Filter out invalid chunks
    valid_chunks = [
        chunk for chunk in chunks if chunk is not None and "content" in chunk
    ]

    if not valid_chunks:
        logger.warning("No valid chunks found after filtering")
        return PROMPTS["fail_response"]

    maybe_trunc_chunks = truncate_list_by_token_size(
        valid_chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_chunks,
    )

    if not maybe_trunc_chunks:
        logger.warning("No chunks left after truncation")
        return PROMPTS["fail_response"]

    logger.info(f"Truncate {len(chunks)} to {len(maybe_trunc_chunks)} chunks")
    maybe_trunc_chunks = sorted(
        maybe_trunc_chunks, key=lambda x: (x["data"].get("full_doc_id", ""), x["data"]["chunk_order_index"])
    )
    section = "\n--New Chunk--\n".join([c["content"] for c in maybe_trunc_chunks])

    if query_param.only_need_context:
        return section

    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )

    if query_param.only_need_prompt:
        return sys_prompt

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )

    if len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt):]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    if query_param.save_to_cache:
        # Save to cache
        chunks_ids = [chunk["id"] for chunk in maybe_trunc_chunks]
        misc = {
            "retrieved_objects": {"text_chunks_ids": chunks_ids}
        }
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=response,
                prompt=sys_prompt,
                query=query,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=query_param.mode,
                misc=misc
            ),
        )
    
    return response


async def direct_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
    relationships_vdb: BaseVectorStorage = None,
    text_chunks_vdb: BaseVectorStorage = None,
) -> str:
    use_model_func = global_config["llm_model_func"]

    llm_model_name = extract_llm_model_name(global_config["llm_model_name"])
    args_hash = compute_args_hash(query_param.mode, query, llm_model_name)
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode
    )
    if (not query_param.ignore_cache) and cached_response is not None:
        return cached_response

    context, retrieved_objects_dict = await _build_local_query_context(
        query,
        knowledge_graph_inst,
        entities_vdb,
        text_chunks_db,
        query_param,
        relationships_vdb=relationships_vdb,
        text_chunks_vdb=text_chunks_vdb,
        verbose=True
    )

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    if query_param.save_to_cache:
        misc = {
            "retrieved_objects": retrieved_objects_dict
        }
        # Save to cache
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=response,
                prompt=sys_prompt,
                query=query,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=query_param.mode,
                misc=misc
            ),
        )

    return response

async def hgmem_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    memory: Memory = None,
    hashing_kv: BaseKVStorage = None,
):
    # Set mode
    if query_param.mode != "hgmem":
        logger.error(f"Incorrect mode {query_param.mode} in hgmem_query.")
        return PROMPTS["fail_response"]

    use_model_func = global_config["llm_model_func"]
    query_config = global_config["config"]["QUERY"]
    entity_extraction_config = global_config["config"]["ENTITY_EXTRACTION"]
    format_dict = {
        "language": global_config["addon_params"].get("language", query_config["DEFAULT_LANGUAGE"]),
        "entity_types": global_config["addon_params"].get("entity_types", entity_extraction_config["DEFAULT_ENTITY_TYPES"]),
        "tuple_delimiter": global_config["addon_params"].get("tuple_delimiter", entity_extraction_config["DEFAULT_TUPLE_DELIMITER"]),
        "record_delimiter": global_config["addon_params"].get("record_delimiter", entity_extraction_config["DEFAULT_RECORD_DELIMITER"]),
        "object_delimiter": global_config["addon_params"].get("object_delimiter", entity_extraction_config["DEFAULT_OBJECT_DELIMITER"]),
        "completion_delimiter": global_config["addon_params"].get("completion_delimiter", entity_extraction_config["DEFAULT_COMPLETION_DELIMITER"])
    }

    # Handle cache
    llm_model_name = extract_llm_model_name(global_config["llm_model_name"])
    args_hash = compute_args_hash(query_param.mode, query, llm_model_name)
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode
    )
    if (not query_param.ignore_cache) and cached_response is not None:
        return cached_response

    def post_process_planning_results(response: str):
        response = response.strip()
        loc_concern = response.find("[Concerns]:")
        assert loc_concern != -1, f"Response does not conform to the required output. --{response}"

        def _handle_single_concern_extraction(record_attributes):
            assert len(record_attributes) == 1 or len(record_attributes) == 2, record_attributes

            if judgement == "2.1" and len(record_attributes) == 2:
                concern = clean_str(record_attributes[1])
                related_memory_points_indices = [int(mp_index_str) for mp_index_str in record_attributes[0].replace(" ", "").split(",")]
            elif judgement == "2.2" and len(record_attributes) == 1:
                concern = clean_str(record_attributes[0])
                related_memory_points_indices = None
            else:
                raise ValueError(f"Judgement: {judgement}, {record_attributes}")
            concern_dict = {"concern": concern, "related_memory_points_indices": related_memory_points_indices}
            return concern_dict
        try:
            judgement = response[:loc_concern].replace("[Judgement]:", "").strip()
            concern_str = response[loc_concern:].replace("[Concerns]:", "").strip()
            if judgement == "1":
                raised_concerns = None
            elif judgement == "2.1" or judgement == "2.2":
                assert "<None>" not in concern_str, f"{concern_str}"
                records = split_string_by_multi_markers(
                    concern_str,
                    [format_dict["record_delimiter"], format_dict["completion_delimiter"]],
                )

                raised_concerns = []
                for record in records:
                    if record is None:
                        continue
                    record_attributes = split_string_by_multi_markers(
                        record, [format_dict["tuple_delimiter"]]
                    )
                    if_point = _handle_single_concern_extraction(record_attributes)
                    if if_point is not None:
                        raised_concerns.append(if_point)
            else:
                raised_concerns = None
                print(f"Invalid planning results:\n{response}")
        except Exception as e:
            raise ValueError(e)

        return judgement, raised_concerns

    def post_process_generate_subqueries_results(response: str):
        subqueries_str = response.replace("[Subqueries]:", "").strip()
        records = split_string_by_multi_markers(
            subqueries_str,
            [format_dict["record_delimiter"], format_dict["completion_delimiter"]],
        )
        subqueries = []
        for record in records:
            if not record:
                continue
            subqueries.append(record.strip())

        return subqueries

    #memory_context = ""
    memory_points_context = ""
    history_judgements = []
    history_concerns = []
    history_subqueries = []
    history_retrieved_objects = []
    history_retrieved_info_context = []
    history_memory_states = []
    judgement = None
    print(f"Main query: {query}")
    cur_subqueries = [query]
    num_turn = 0
    while num_turn <= query_param.max_num_turns and cur_subqueries is not None:
        print(f"Judgement: {judgement}")
        print(f"Subqueries {num_turn+1}: {cur_subqueries}")
        history_subqueries.append(cur_subqueries)
        # Locate relevant entities
        filter_lambda = None
        if judgement == "2.1":
            assert len(raised_concerns) == len(cur_subqueries), f"{raised_concerns}\n{cur_subqueries}\n{len(raised_concerns)}, {len(cur_subqueries)}"
            cur_memory_points = memory.get_memory_points()
            k_hops_nodes_dict = {}
            entity_vdb_ids_dict = {}
            for concern in raised_concerns:
                concern_content = concern["concern"]
                related_memory_points_indices = concern["related_memory_points_indices"]
                related_memory_points = [cur_memory_points[mp_index] for mp_index in related_memory_points_indices]
                unique_involved_objects = []
                for involved_objects in related_memory_points:
                    unique_involved_objects.extend(involved_objects)
                unique_involved_objects = list(set(unique_involved_objects))
                for obj in unique_involved_objects:
                    nodes_within_k_hops = k_hops_nodes_dict.get(obj, await knowledge_graph_inst.get_k_hop_nodes(obj, k=2, mode="nodes_within_k_hops"))
                    k_hops_nodes_dict[obj] = nodes_within_k_hops
                    cur_entity_vdb_ids_dict = convert_to_entity_vdb_ids_dict(nodes_within_k_hops)
                    entity_vdb_ids_dict.update(cur_entity_vdb_ids_dict)
            filter_lambda = lambda data: data["__id__"] in entity_vdb_ids_dict

        retrieved_info_context, retrieved_objects_dict = await _build_local_query_context(
            cur_subqueries,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
            relationships_vdb=relationships_vdb,
            text_chunks_vdb=text_chunks_vdb,
            filter_lambda=filter_lambda,
            verbose=True,
            passage_retrieval=True,
            llm_model_func=use_model_func,
            memory=memory
        )
        history_retrieved_info_context.append(retrieved_info_context)
        history_retrieved_objects.append(retrieved_objects_dict)
        #entities_ids, use_relationships_ids, use_text_chunks_ids = retrieved_objects_dict["entities_ids"], retrieved_objects_dict["relationships_ids"], retrieved_objects_dict["text_chunks_ids"]

        memory_state = {"before": await memory.get_memory_context(format_dict["object_delimiter"])}
        inserted_points, updated_points = await memory.evolve(retrieved_info_context, knowledge_graph_inst, entities_vdb, relationships_vdb,
                                                              use_model_func, query, cur_subqueries, query_param, format_dict, retrieved_text_chunks_ids=retrieved_objects_dict["text_chunks_ids"])

        if len(history_memory_states) > 0:
            await memory.reorganize_memory(knowledge_graph_inst, entities_vdb, relationships_vdb, use_model_func, query, format_dict)
        memory_state["inserted"] = inserted_points
        memory_state["updated"] = updated_points
        history_memory_states.append(memory_state)

        #memory_context = await memory.get_memory_context(format_dict["object_delimiter"])
        memory_history_subqueries_context = memory.get_history_subqueries_context().strip()
        if not memory_history_subqueries_context:
            memory_history_subqueries_context = "<None>"
        memory_points_context = await memory.get_memory_points_context()
        extended_info = await memory.get_extended_info(knowledge_graph_inst, entities_vdb, query, query_param, llm_model_func=use_model_func)

        context_base = dict(query=query, memory=memory_points_context, num_concerns=3, extended_info=extended_info, **format_dict)
        planning_prompt = PROMPTS["planning_with_hint"].format(**context_base)
        planning_result = await use_model_func(planning_prompt)

        judgement, raised_concerns = post_process_planning_results(planning_result)
        history_judgements.append(judgement)

        if judgement == "1":
            cur_subqueries = None
        else:
            all_concerns = []
            for concern_idx, concern_dict in enumerate(raised_concerns):
                concern = concern_dict["concern"]
                related_memory_points_indices = concern_dict["related_memory_points_indices"]
                if related_memory_points_indices is not None:
                    concern_str = f"Concern {concern_idx}: {concern} ---- Relate to memory point {related_memory_points_indices}"
                else:
                    concern_str = f"Concern {concern_idx}: {concern}"
                all_concerns.append(concern_str)
            concerns_str = "\n".join(all_concerns)
            history_concerns.append(all_concerns)

            context_base = dict(query=query, memory=memory_points_context, concerns=concerns_str,
                                extended_info=extended_info, history_subqueries=memory_history_subqueries_context, **format_dict)
            generate_subquery_prompt = PROMPTS["generate_subqueries"].format(**context_base)
            generate_subquery_result = await use_model_func(generate_subquery_prompt)
            cur_subqueries = post_process_generate_subqueries_results(generate_subquery_result)
        num_turn += 1


    #TODO 利用Memory和相关的信息进行最终作答
    final_memory_state = {"final": await memory.get_memory_context(format_dict["object_delimiter"])}
    history_memory_states.append(final_memory_state)
    memory_related_info, pointwise_memory_related_chunks_ids, final_selected_chunks_ids = \
        await memory.get_memory_pointwise_related_info(knowledge_graph_inst, text_chunks_db, text_chunks_vdb, query, query_param, history_retrieved_objects)

    if query_param.only_need_context:
        return memory_related_info
    if memory_related_info is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["rag_response_with_memory_system_prompt"]
    sys_prompt = sys_prompt_temp.format(
        memory=memory_points_context, context_data=memory_related_info
    )
    if query_param.only_need_prompt:
        return sys_prompt

    user_prompt_temp = PROMPTS["rag_response_with_memory_user_prompt"]
    user_prompt = user_prompt_temp.format(query=query, response_type=query_param.response_type)

    response = await use_model_func(user_prompt, system_prompt=sys_prompt, stream=query_param.stream)
    if isinstance(response, str) and len(response) > len(user_prompt):
        response = (
            response.replace(user_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    if query_param.save_to_cache:
        # Save to cache
        misc = {
            "retrieved_objects": history_retrieved_objects,
            "judgements": history_judgements,
            "concerns": history_concerns,
            "subqueries": history_subqueries,
            "memory_states": history_memory_states,
            "pointwise_memory_related_chunks_ids": pointwise_memory_related_chunks_ids,
            "final_selected_chunks_ids": final_selected_chunks_ids,
        }
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=response,
                prompt=sys_prompt,
                query=query,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=query_param.mode,
                misc=misc
            ),
        )

    await memory.clear_memory()
    print(f"------------------{query}------------------")
    print(response)
    print(f"-------------{len(response.split())}-------------{len(encode_string_by_tiktoken(response))}-------------")
    return response
