import asyncio
import re
from collections import defaultdict

from .utils import (
    split_string_by_multi_markers, clean_str, truncate_list_by_token_size, build_text_chunks_context,
    build_entities_context, truncate_attribute_by_token_size, compute_mdhash_id, maybe_truncate_description,
    convert_to_entity_vdb_ids_dict,
)
from .prompt import PROMPTS, GRAPH_FIELD_SEP
#from .prompt_dev import PROMPTS_DEV


def postprocess_evolve_memory(response, format_dict):
    object_delimiter = format_dict["object_delimiter"]
    tuple_delimiter = format_dict["tuple_delimiter"]
    record_delimiter = format_dict["record_delimiter"]
    completion_delimiter = format_dict["completion_delimiter"]

    response = response.strip()

    loc_updated_memory_points = response.find("[Updated Memory Points]:")
    if loc_updated_memory_points == -1:
        loc_updated_memory_points = response.find(completion_delimiter) + len(completion_delimiter)
    assert loc_updated_memory_points != -1
    inserted_memory_points_str = response[:loc_updated_memory_points].replace("[Inserted Memory Points]:", "").strip()
    records = split_string_by_multi_markers(
        inserted_memory_points_str,
        [record_delimiter, completion_delimiter],
    )

    def _handle_single_memory_point_extraction(record_attributes):
        if len(record_attributes) < 3 or record_attributes[0] != 'point':
            return None

        point_involved_objects = clean_str(record_attributes[1].upper())
        involved_objects = point_involved_objects.split(object_delimiter)
        if not involved_objects:
            return None
        point_description = clean_str(record_attributes[2])
        entity_dict = {"involved_objects": involved_objects, "description": point_description}
        return entity_dict

    maybe_inserted_points = []
    for record in records:
        record = re.search(r"\((.*)\)", record)
        if record is None:
            continue
        record = record.group(1)
        record_attributes = split_string_by_multi_markers(
            record, [tuple_delimiter]
        )
        if_point = _handle_single_memory_point_extraction(record_attributes)
        if if_point is not None:
            maybe_inserted_points.append(if_point)

    updated_memory_points_str = response[loc_updated_memory_points:].replace("[Updated Memory Points]:", "").strip()
    records = split_string_by_multi_markers(
        updated_memory_points_str,
        [record_delimiter, completion_delimiter],
    )
    maybe_updated_points = []
    for record in records:
        mp_index_str = record.split(",")[0].replace("(", "").replace(")", "").strip()
        record = re.search(r"\((.*)\)", record)
        if record is None or (not mp_index_str.isdigit()):
            continue
        record = record.group(1)
        record_attributes = split_string_by_multi_markers(
            record, [tuple_delimiter]
        )
        if_point = _handle_single_memory_point_extraction(
            record_attributes
        )
        if if_point is not None:
            maybe_updated_points.append((int(mp_index_str), if_point))

    return maybe_inserted_points, maybe_updated_points


def postprocess_summarize_absent_entities_relationships(response, format_dict):
    object_delimiter = format_dict["object_delimiter"]
    tuple_delimiter = format_dict["tuple_delimiter"]
    record_delimiter = format_dict["record_delimiter"]
    completion_delimiter = format_dict["completion_delimiter"]

    response = response.strip()
    records = split_string_by_multi_markers(response, [record_delimiter, completion_delimiter])

    def _handle_single_entity_summarization(record_attributes: list[str]):
        if len(record_attributes) < 4 or record_attributes[0] != 'entity':
            return None

        # add this record as a node in the G
        entity_name = clean_str(record_attributes[1].upper())
        if not entity_name.strip():
            return None
        entity_type = clean_str(record_attributes[2].upper())
        entity_description = clean_str(record_attributes[3])
        entity_dict = {"entity_name": entity_name, "entity_type": entity_type, "description": entity_description, "source_id": "", "state": "temporary"}
        return entity_dict

    def _handle_single_relationship_summarization(record_attributes: list[str]):
        if len(record_attributes) < 5 or record_attributes[0] != 'relationship':
            return None

        # add this record as edge
        source = clean_str(record_attributes[1].upper())
        target = clean_str(record_attributes[2].upper())
        relationship_description = clean_str(record_attributes[3])
        relationship_keywords = clean_str(record_attributes[4])
        relationship_dict = {"src_id": source, "tgt_id": target, "description": relationship_description, "keywords": relationship_keywords, "source_id": "", "state": "temporary"}
        return relationship_dict


    maybe_entities = {}
    maybe_relationships = {}
    for record in records:
        record = re.search(r"\((.*)\)", record)
        if record is None:
            continue
        record = record.group(1)
        record_attributes = split_string_by_multi_markers(record, [tuple_delimiter])
        if_entity = _handle_single_entity_summarization(record_attributes)
        if if_entity is not None:
            maybe_entities[if_entity["entity_name"]] = if_entity
            continue
        #TODO
        if_relationship = _handle_single_relationship_summarization(record_attributes)
        if if_relationship is not None:
            maybe_relationships[(if_relationship["src_id"], if_relationship["tgt_id"])] = if_relationship

    return maybe_entities, maybe_relationships


def postprocess_reorganize_memory(response, format_dict):
    object_delimiter = format_dict["object_delimiter"]
    tuple_delimiter = format_dict["tuple_delimiter"]
    record_delimiter = format_dict["record_delimiter"]
    completion_delimiter = format_dict["completion_delimiter"]

    loc_points_to_merge = response.find("[Points_to_Merge]:")
    assert loc_points_to_merge != -1
    merged_memory_points_str = response[loc_points_to_merge:].strip()
    if "<None>" in merged_memory_points_str:
        return []

    records = split_string_by_multi_markers(
        merged_memory_points_str,
        [record_delimiter, completion_delimiter],
    )

    def _handle_single_memory_point_extraction(record_attributes):
        if len(record_attributes) < 2:
            return None
        memory_points_indices = [int(index_str) for index_str in record_attributes[0].replace(" ", "").split(",")]
        if len(memory_points_indices) <= 1:
            return None
        point_description = clean_str(record_attributes[1])
        entity_dict = {"memory_points_indices": memory_points_indices, "description": point_description}
        return entity_dict

    reorganized_memory_points = []
    for record in records:
        record = re.search(r"\((.*)\)", record)
        if record is None:
            continue
        record = record.group(1)
        record_attributes = split_string_by_multi_markers(
            record, [tuple_delimiter]
        )
        if_point = _handle_single_memory_point_extraction(record_attributes)
        if if_point is not None:
            reorganized_memory_points.append(if_point)

    return reorganized_memory_points


async def add_absent_entities_to_graph_and_vdb(knowledge_graph_inst, entities_vdb, collected_absent_entities, remaining_absent_entities, entity_description_func):
    data_for_entities_vdb = {}
    for entity_name, entity_info in collected_absent_entities.items():
        await knowledge_graph_inst.upsert_node(entity_name, entity_info)
        dp = {
            "entity_name": entity_name,
            "description": entity_info["description"],
            "entity_type": entity_info["entity_type"],
            "source_id": entity_info["source_id"]
        }
        key = compute_mdhash_id(dp["entity_name"], prefix="ent-")
        data_for_entities_vdb[key] = {
            "content": entity_description_func(dp["entity_name"], dp["description"]),
            "entity_name": dp["entity_name"],
        }
    for entity_name in remaining_absent_entities:
        entity_info = {"entity_name": entity_name, "entity_type": "", "description": "", "source_id": "", "state": "missing"}
        await knowledge_graph_inst.upsert_node(entity_name, entity_info)
        dp = {
            "entity_name": entity_name,
            "description": entity_info["description"],
            "entity_type": entity_info["entity_type"],
            "source_id": entity_info["source_id"]
        }
        key = compute_mdhash_id(dp["entity_name"], prefix="ent-")
        data_for_entities_vdb[key] = {
            "content": entity_description_func(dp["entity_name"], dp["description"]),
            "entity_name": dp["entity_name"],
        }
    await entities_vdb.upsert(data_for_entities_vdb)


async def add_absent_relationships_to_graph_and_vdb(knowledge_graph_inst, relationships_vdb, collected_absent_relationships, remaining_absent_relationships, relationship_description_func):
    data_for_relationships_vdb = {}
    for relationship_identifier, relationship_info in collected_absent_relationships.items():
        await knowledge_graph_inst.upsert_edge(relationship_identifier[0], relationship_identifier[1], relationship_info)
        dp = {
            "src_id": relationship_identifier[0],
            "tgt_id": relationship_identifier[1],
            "description": relationship_info["description"],
            "keywords": relationship_info["keywords"],
            "source_id": relationship_info["source_id"]
        }
        key = compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-")
        value = {
            "src_id": dp["src_id"],
            "tgt_id": dp["tgt_id"],
            "content": relationship_description_func(dp["keywords"], dp["src_id"], dp["tgt_id"], dp["description"]),
        }
        data_for_relationships_vdb[key] = value
    for relationship_identifier in remaining_absent_relationships:
        relationship_info = {"src_id": relationship_identifier[0], "tgt_id": relationship_identifier[1],
                             "description": "", "keywords": "", "source_id": "", "state": "missing"}
        await knowledge_graph_inst.upsert_edge(relationship_identifier[0], relationship_identifier[1], relationship_info)
        dp = {
            "src_id": relationship_identifier[0],
            "tgt_id": relationship_identifier[1],
            "description": relationship_info["description"],
            "keywords": relationship_info["keywords"],
            "source_id": relationship_info["source_id"]
        }
        key = compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-")
        value = {
            "src_id": dp["src_id"],
            "tgt_id": dp["tgt_id"],
            "content": relationship_description_func(dp["keywords"], dp["src_id"], dp["tgt_id"], dp["description"]),
        }
        data_for_relationships_vdb[key] = value
    await relationships_vdb.upsert(data_for_relationships_vdb)


async def collect_absent_entities_relationships(absent_entities_hyperedges_kv, info, knowledge_graph_inst,
                                                entities_vdb, relationships_vdb, llm_model_func, format_dict,
                                                entity_description_func, relationship_description_func):
    language = format_dict["language"]
    entity_types = format_dict["entity_types"]
    object_delimiter = format_dict["object_delimiter"]
    tuple_delimiter = format_dict["tuple_delimiter"]
    record_delimiter = format_dict["record_delimiter"]
    completion_delimiter = format_dict["completion_delimiter"]

    #absent_entities_relationships_kv = {k: list(set(v)) for k, v in absent_entities_relationships_kv.items()}
    absent_unique_relationships = []
    for entity, relevant_hyperedges in absent_entities_hyperedges_kv.items():
        for hyperedge in relevant_hyperedges:
            for rel_entity in hyperedge:
                if entity != rel_entity and (entity, rel_entity) not in absent_unique_relationships and (rel_entity, entity) not in absent_unique_relationships:
                    absent_unique_relationships.append((entity, rel_entity))
    collected_absent_entities = {}
    collected_absent_relationships = {}
    remaining_absent_entities = list(absent_entities_hyperedges_kv.keys())
    remaining_absent_relationships = absent_unique_relationships
    num_collect, max_num_collect = 0, 3
    while (len(remaining_absent_entities) > 0 or len(remaining_absent_relationships)) and num_collect < max_num_collect:
        target_entities = "\n".join([f"- {entity}" for entity in remaining_absent_entities])
        target_relationships = "\n".join([f"- {relationship[0]} --&-- {relationship[1]}" for relationship in remaining_absent_relationships])
        context_base = dict(
            language=language,
            entity_types=entity_types,
            tuple_delimiter=tuple_delimiter,
            record_delimiter=record_delimiter,
            object_delimiter=object_delimiter,
            completion_delimiter=completion_delimiter,
            target_entities=target_entities,
            target_relationships=target_relationships,
            info=info
        )
        summarize_absent_entities_relationships_prompt = PROMPTS["summarize_absent_entities_relationships"]
        summarize_absent_entities_relationships_prompt = summarize_absent_entities_relationships_prompt.format(**context_base)
        response_summarize_absent_entities_relationships = await llm_model_func(summarize_absent_entities_relationships_prompt)

        summarized_absent_entities_info, summarized_absent_relationships_info = postprocess_summarize_absent_entities_relationships(response_summarize_absent_entities_relationships, format_dict)
        collected_absent_entities.update({entity_name: entity_info for entity_name, entity_info in summarized_absent_entities_info.items() if entity_name in absent_entities_hyperedges_kv})
        collected_absent_relationships.update({relationship_identifier: relationship_info for relationship_identifier, relationship_info in summarized_absent_relationships_info.items() if relationship_identifier in absent_unique_relationships})
        remaining_absent_entities = [entity_name for entity_name in absent_entities_hyperedges_kv.keys() if entity_name not in collected_absent_entities]
        remaining_absent_relationships = [relationship_identifier for relationship_identifier in absent_unique_relationships if relationship_identifier not in collected_absent_relationships]
        num_collect += 1

    await add_absent_entities_to_graph_and_vdb(knowledge_graph_inst, entities_vdb, collected_absent_entities, remaining_absent_entities, entity_description_func)
    await add_absent_relationships_to_graph_and_vdb(knowledge_graph_inst, relationships_vdb, collected_absent_relationships, remaining_absent_relationships, relationship_description_func)


def postprocess_select_entities(response):
    loc_selected_entities = response.find("[Selected]:")
    if loc_selected_entities == -1:
        print(f"Format error: {response}")
    selected_entities_str = response[loc_selected_entities:].replace("[Selected]:", "").strip()
    if "<None>" in selected_entities_str:
        return []
    else:
        try:
            selected_entities_indices = [int(index_str) for index_str in selected_entities_str.replace(" ", "").split(",")]
        except Exception as e:
            print(response)
            raise ValueError
        return selected_entities_indices


class Memory:
    def __init__(self, hypergraph_storage_cls, global_config, embedding_func):
        self.global_config = global_config
        self.memory_hypergraph = hypergraph_storage_cls(
            namespace="memory",
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._memory_points = []

        self._history_subqueries = []
        self._history_retrieved_text_chunks_ids = []

    def get_history_subqueries_context(self, include_first_query=False):
        history_subqueries = self._history_subqueries
        if not include_first_query:
            history_subqueries = history_subqueries[1:]

        tiled_history_subqueries = []
        for subqueries in history_subqueries:
            assert type(subqueries) is list, f"Error data type: {subqueries}"
            tiled_history_subqueries.extend(subqueries)

        history_subqueries_context = "\n".join([f"- {subquery}" for subquery in tiled_history_subqueries])
        return history_subqueries_context

    def get_memory_points(self):
        return self._memory_points

    async def get_memory_point_info(self, mp_identifier):
        memory_point = await self.memory_hypergraph.get_hyperedge(mp_identifier)
        return memory_point['description']
    
    async def get_memory_points_context(self, object_delimiter=", "):
        memory_point_details = []
        for mp_idx, mp_identifier in enumerate(self._memory_points, start=0):
            memory_point = await self.memory_hypergraph.get_hyperedge(mp_identifier)
            memory_point_str = f"- Point ({mp_idx})\nInvolved Objects: {object_delimiter.join(mp_identifier)}\nDescription: {memory_point['description']}"
            memory_point_details.append(memory_point_str)
        memory_points_context = "\n".join(memory_point_details)

        return memory_points_context

    async def get_memory_context(self, object_delimiter=", ", include_first_query=False):
        memory_context = f"**Previous Subqueries**\n{self.get_history_subqueries_context(include_first_query=include_first_query)}\n\n" \
        f"**Memory Points**\n{await self.get_memory_points_context(object_delimiter)}"

        return memory_context

    async def evolve(self, retrieved_info_context, knowledge_graph_inst, entities_vdb, relationships_vdb,
                     llm_model_func, main_query, cur_subqueries, query_param, format_dict, retrieved_text_chunks_ids=None):
        language = format_dict["language"]
        entity_types = format_dict["entity_types"]
        object_delimiter = format_dict["object_delimiter"]
        tuple_delimiter = format_dict["tuple_delimiter"]
        record_delimiter = format_dict["record_delimiter"]
        completion_delimiter = format_dict["completion_delimiter"]

        if type(cur_subqueries) == str:
            cur_subqueries = [cur_subqueries]
        self._history_subqueries.append(cur_subqueries)
        self._history_retrieved_text_chunks_ids.append(retrieved_text_chunks_ids)

        context_base = dict(
            tuple_delimiter=tuple_delimiter,
            record_delimiter=record_delimiter,
            object_delimiter=object_delimiter,
            completion_delimiter=completion_delimiter,
            language=language,
        )
        evolve_memory_system_prompt = PROMPTS["evolve_memory_system_prompt"]
        evolve_memory_system_prompt = evolve_memory_system_prompt.format(**context_base)

        cur_subqueries_str = "\n".join([f"- {subquery}" for subquery in cur_subqueries if subquery != main_query])
        context_base = dict(
            tuple_delimiter=tuple_delimiter,
            record_delimiter=record_delimiter,
            object_delimiter=object_delimiter,
            completion_delimiter=completion_delimiter,
            main_query=main_query,
            cur_subqueries=cur_subqueries_str,
            memory=await self.get_memory_points_context(object_delimiter),
            retrieved_info=retrieved_info_context
        )
        evolve_memory_user_prompt = PROMPTS["evolve_memory_user_prompt"]
        evolve_memory_user_prompt = evolve_memory_user_prompt.format(**context_base)
        response_evolve_memory = await llm_model_func(
            evolve_memory_user_prompt,
            system_prompt=evolve_memory_system_prompt,
            stream=query_param.stream,
        )
        maybe_inserted_points, maybe_updated_points = postprocess_evolve_memory(response_evolve_memory, format_dict)

        absent_entities_hyperedges_kv = defaultdict(list)
        for memory_point_to_update in maybe_updated_points:
            mp_index, memory_point_to_update = memory_point_to_update
            involved_objects = memory_point_to_update["involved_objects"]
            memory_point_description = memory_point_to_update["description"]
            for object_name in involved_objects:
                if not (await self.memory_hypergraph.has_vertex(object_name)):
                    if (await knowledge_graph_inst.has_node(object_name)):
                        node_data = await knowledge_graph_inst.get_node(object_name)
                        await self.memory_hypergraph.upsert_vertex(
                            object_name,
                            node_data,
                        )
                    else:
                        absent_entities_hyperedges_kv[object_name].append(involved_objects)
                        await self.memory_hypergraph.upsert_vertex(object_name)
            await self.memory_hypergraph.remove_hyperedge(self._memory_points[mp_index])

            self._memory_points[mp_index] = involved_objects
            hyperedge_data = dict(
                description=memory_point_description,
            )
            await self.memory_hypergraph.upsert_hyperedge(
                involved_objects,
                dict(hyperedge_data),
            )

        for memory_point_to_insert in maybe_inserted_points:
            involved_objects = memory_point_to_insert["involved_objects"]
            memory_point_description = memory_point_to_insert["description"]
            for object_name in involved_objects:
                if not (await self.memory_hypergraph.has_vertex(object_name)):
                    if (await knowledge_graph_inst.has_node(object_name)):
                        node_data = await knowledge_graph_inst.get_node(object_name)
                        await self.memory_hypergraph.upsert_vertex(
                            object_name,
                            node_data,
                        )
                    else:
                        absent_entities_hyperedges_kv[object_name].append(involved_objects)
                        await self.memory_hypergraph.upsert_vertex(object_name)
            hyperedge_data = dict(description=memory_point_description)
            await self.memory_hypergraph.upsert_hyperedge(
                involved_objects,
                dict(hyperedge_data),
            )
            self._memory_points.append(involved_objects)

        """
        absent_unique_relationships = []
        for entity, relevant_hyperedges in absent_entities_hyperedges_kv.items():
            for hyperedge in relevant_hyperedges:
                for rel_entity in hyperedge:
                    if entity != rel_entity and (entity, rel_entity) not in absent_unique_relationships and (rel_entity, entity) not in absent_unique_relationships:
                        absent_unique_relationships.append((entity, rel_entity))
        collected_absent_entities = {}
        collected_absent_relationships = {}
        remaining_absent_entities = list(absent_entities_hyperedges_kv.keys())
        remaining_absent_relationships = absent_unique_relationships
        num_collect, max_num_collect = 0, 3
        while (len(remaining_absent_entities) > 0 or len(remaining_absent_relationships)) and num_collect < max_num_collect:
            target_entities = "\n".join([f"- {entity}" for entity in remaining_absent_entities])
            target_relationships = "\n".join([f"- {relationship[0]} --&-- {relationship[1]}" for relationship in remaining_absent_relationships])
            context_base = dict(
                language=language,
                entity_types=entity_types,
                tuple_delimiter=tuple_delimiter,
                record_delimiter=record_delimiter,
                object_delimiter=object_delimiter,
                completion_delimiter=completion_delimiter,
                target_entities=target_entities,
                target_relationships=target_relationships,
                info=retrieved_info_context
            )
            summarize_absent_entities_relationships_prompt = PROMPTS["summarize_absent_entities_relationships"]
            summarize_absent_entities_relationships_prompt = summarize_absent_entities_relationships_prompt.format(**context_base)
            response_summarize_absent_entities_relationships = await llm_model_func(summarize_absent_entities_relationships_prompt)

            summarized_absent_entities_info, summarized_absent_relationships_info = postprocess_summarize_absent_entities_relationships(response_summarize_absent_entities_relationships, format_dict)
            collected_absent_entities.update({entity_name: entity_info for entity_name, entity_info in summarized_absent_entities_info.items() if entity_name in absent_entities_hyperedges_kv})
            collected_absent_relationships.update({relationship_identifier: relationship_info for relationship_identifier, relationship_info in summarized_absent_relationships_info.items() if relationship_identifier in absent_unique_relationships})
            remaining_absent_entities = [entity_name for entity_name in absent_entities_hyperedges_kv.keys() if entity_name not in collected_absent_entities]
            remaining_absent_relationships = [relationship_identifier for relationship_identifier in absent_unique_relationships if relationship_identifier not in collected_absent_relationships]
            num_collect += 1

        await add_absent_entities_to_graph_and_vdb(knowledge_graph_inst, entities_vdb, collected_absent_entities, remaining_absent_entities, self.global_config["entity_description_func"])
        await add_absent_relationships_to_graph_and_vdb(knowledge_graph_inst, relationships_vdb, collected_absent_relationships, remaining_absent_relationships, self.global_config["relationship_description_func"])
        """
        await collect_absent_entities_relationships(absent_entities_hyperedges_kv, retrieved_info_context,
                                                    knowledge_graph_inst, entities_vdb, relationships_vdb,
                                                    llm_model_func, format_dict,
                                                    self.global_config["entity_description_func"],
                                                    self.global_config["relationship_description_func"])
        return maybe_inserted_points, maybe_updated_points

    async def reorganize_memory(self, knowledge_graph_inst, entities_vdb, relationships_vdb, llm_model_func, main_query, format_dict):
        language = format_dict["language"]
        entity_types = format_dict["entity_types"]
        object_delimiter = format_dict["object_delimiter"]
        tuple_delimiter = format_dict["tuple_delimiter"]
        record_delimiter = format_dict["record_delimiter"]
        completion_delimiter = format_dict["completion_delimiter"]
        original_memory_points_context = await self.get_memory_points_context()

        context_base = dict(
            tuple_delimiter=tuple_delimiter,
            record_delimiter=record_delimiter,
            object_delimiter=object_delimiter,
            completion_delimiter=completion_delimiter,
            language=language,
            main_query=main_query,
            memory=original_memory_points_context
        )
        reorganize_memory_prompt = PROMPTS["reorganize_memory"]
        reorganize_memory_prompt = reorganize_memory_prompt.format(**context_base)

        response_reorganize_memory = await llm_model_func(reorganize_memory_prompt)
        reorganized_memory_points = postprocess_reorganize_memory(response_reorganize_memory, format_dict)

        merged_memory_points = []
        absent_entities_hyperedges_kv = defaultdict(list)
        for reorganized_memory_point in reorganized_memory_points:
            mp_indices_to_merge = reorganized_memory_point["memory_points_indices"]
            mp_description = reorganized_memory_point["description"]
            merged_involved_objects = []
            for mp_index in mp_indices_to_merge:
                merged_involved_objects.extend(self._memory_points[mp_index])
            merged_memory_point = list(set(merged_involved_objects))
            merged_memory_points.append(merged_memory_point)

            for object_name in merged_involved_objects:
                if not (await self.memory_hypergraph.has_vertex(object_name)):
                    if (await knowledge_graph_inst.has_node(object_name)):
                        node_data = await knowledge_graph_inst.get_node(object_name)
                        await self.memory_hypergraph.upsert_vertex(
                            object_name,
                            node_data,
                        )
                    else:
                        absent_entities_hyperedges_kv[object_name].append(merged_involved_objects)
                        await self.memory_hypergraph.upsert_vertex(object_name)
            hyperedge_data = dict(description=mp_description)
            await self.memory_hypergraph.upsert_hyperedge(
                merged_involved_objects,
                dict(hyperedge_data),
            )

        backup_memory_points = self._memory_points
        for reorganized_memory_point in reorganized_memory_points:
            mp_indices_to_merge = reorganized_memory_point["memory_points_indices"]
            for mp_index in mp_indices_to_merge:
                self._memory_points[mp_index] = None
        new_memory_points = [mp for mp in self._memory_points if mp is not None]
        new_memory_points.extend(merged_memory_points)
        self._memory_points = new_memory_points

        await collect_absent_entities_relationships(absent_entities_hyperedges_kv, original_memory_points_context,
                                                    knowledge_graph_inst, entities_vdb, relationships_vdb,
                                                    llm_model_func, format_dict,
                                                    self.global_config["entity_description_func"],
                                                    self.global_config["relationship_description_func"])
        a=1

    async def get_extended_info(self, knowledge_graph_inst, entities_vdb, query, query_param, llm_model_func=None):
        related_entities_data = []
        related_entities_data_dict = {}
        for mp_idx, mp_identifier in enumerate(self._memory_points, start=0):
            for object in mp_identifier:
                if object not in related_entities_data_dict:
                    node_data = {"entity_name": object, **(await knowledge_graph_inst.get_node(object))}
                    related_entities_data.append(node_data)
                    related_entities_data_dict[object] = node_data
        #truncate_attribute_by_token_size(related_entities_data, "description", query_param.max_token_for_entity_description)
        #related_entities_context = build_entities_context(related_entities_data)

        candidate_neighbor_nodes = {}
        for entity_name in related_entities_data_dict.keys():
            neighbor_nodes = await knowledge_graph_inst.get_neighbor_nodes(entity_name)
            #neighbor_nodes = await knowledge_graph_inst.get_k_hop_nodes(entity_name, k=2, mode="nodes_within_k_hops")
            for node_name in neighbor_nodes:
                if node_name not in candidate_neighbor_nodes and node_name not in related_entities_data_dict:
                    entity = await knowledge_graph_inst.get_node(node_name)
                    entity_type = entity["entity_type"]
                    description = maybe_truncate_description(entity["description"], delimiter="<SEP>", max_tokens=query_param.max_token_for_entity_description)
                    entity_detail_str = f"****{node_name}****\nEntity_Type: {entity_type}\nDescription: {description}\n"
                    candidate_neighbor_nodes[node_name] = entity_detail_str
        #TODO
        entity_vdb_ids_dict = convert_to_entity_vdb_ids_dict(candidate_neighbor_nodes.keys())
        filter_lambda = lambda data: data["__id__"] in entity_vdb_ids_dict
        num_extension = min(query_param.top_k_entities, len(candidate_neighbor_nodes))
        results = await entities_vdb.query(query, top_k=num_extension, filter_lambda=filter_lambda)
        candidate_entities_data = await asyncio.gather(
            *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
        )
        candidate_entities_data = [
            {**n, "entity_name": k["entity_name"]}
            for k, n in zip(results, candidate_entities_data)
            if n is not None
        ]
        truncate_attribute_by_token_size(candidate_entities_data, attribute="description", max_token_size=query_param.max_token_for_entity_description)

        entities_data = candidate_entities_data[:query_param.first_k_entities]
        if query_param.llm_selection:
            assert llm_model_func is not None

            entities_details = []
            for entity_idx, entity in enumerate(candidate_entities_data[query_param.first_k_entities:], start=0):
                 entity_str = f"- ({entity_idx}): {entity['entity_name']}\nDescription: {entity['description']}"
                 entities_details.append(entity_str)
            entities_str = "\n".join(entities_details)

            context_base = dict(
                query=query,
                memory=await self.get_memory_points_context(),
                entities_str=entities_str
            )
            select_entities_prompt = PROMPTS["select_entities"]
            select_entities_prompt = select_entities_prompt.format(**context_base)
            response_select_entities = await llm_model_func(select_entities_prompt)
            selected_entities_indices = postprocess_select_entities(response_select_entities)
            llm_selected_entities_data = [candidate_entities_data[query_param.first_k_entities:][index] for index in selected_entities_indices]
            if query_param.llm_select_k_entities >= 0:
                llm_selected_entities_data = llm_selected_entities_data[:query_param.llm_select_k_entities]
            entities_data.extend(llm_selected_entities_data)

        extended_entities_details_str = [candidate_neighbor_nodes[ent["entity_name"]] for ent in entities_data]
        extension_info = "\n".join([cand_detail_str for cand_detail_str in extended_entities_details_str])
        return extension_info

    async def get_memory_pointwise_related_info_full(self, knowledge_graph_inst, text_chunks_db, text_chunks_vdb, query, query_param, verbose=True):
        related_entities_data = []
        related_entities_data_dict = {}
        for mp_identifier in self._memory_points:
            for entity_name in mp_identifier:
                if entity_name not in related_entities_data_dict:
                    node = await knowledge_graph_inst.get_node(entity_name)
                    if node:
                        node_data = {"entity_name": entity_name, **node}
                        related_entities_data.append(node_data)
                        related_entities_data_dict[entity_name] = node_data

        truncate_attribute_by_token_size(related_entities_data, "description", query_param.max_token_for_entity_description)
        entities_context = build_entities_context(related_entities_data)

        pointwise_selected_chunks_ids = []
        final_selected_chunks_ids = set()
        for mp_identifier in self._memory_points:
            num_entities_k = len(mp_identifier)
            if num_entities_k == 0:
                continue

            mp_chunks_ids = set()
            for entity_name in mp_identifier:
                node_data = related_entities_data_dict.get(entity_name)
                if not node_data or not node_data.get("source_id"):
                    continue
                source_ids = set(split_string_by_multi_markers(str(node_data["source_id"]), [GRAPH_FIELD_SEP]))
                mp_chunks_ids.update(source_ids)

            valid_inner_chunks_ids = []
            for cid in mp_chunks_ids:
                if await text_chunks_db.get_by_id(cid) is not None:
                    valid_inner_chunks_ids.append(cid)

            selected_inner_chunks_ids = set()
            if valid_inner_chunks_ids:
                filter_lambda = lambda data: data["__id__"] in valid_inner_chunks_ids
                memory_point_info = await self.get_memory_point_info(mp_identifier)
                selected_inner_chunks = await text_chunks_vdb.query(memory_point_info, top_k=min(query_param.max_inner_chunks_per_memory_point, len(valid_inner_chunks_ids)), filter_lambda=filter_lambda)
                selected_inner_chunks_ids = {chunk["id"] for chunk in selected_inner_chunks}
                final_selected_chunks_ids.update(selected_inner_chunks_ids)

            # 获取outer chunk
            outer_chunks_ids = set()
            for entity_name in mp_identifier:
                node_data = related_entities_data_dict.get(entity_name)
                if not node_data:
                    continue
                # 获取邻居节点
                neighbor_nodes = await knowledge_graph_inst.get_neighbor_nodes(entity_name)
                for neighbor_node in neighbor_nodes:
                    if neighbor_node not in related_entities_data_dict:
                        neighbor_data = await knowledge_graph_inst.get_node(neighbor_node)
                        if neighbor_data and neighbor_data.get("source_id"):
                            neighbor_source_ids = set(
                                split_string_by_multi_markers(str(neighbor_data["source_id"]), [GRAPH_FIELD_SEP]))
                            outer_chunks_ids.update(neighbor_source_ids)

            valid_outer_chunks_ids = []
            for cid in outer_chunks_ids:
                if (await text_chunks_db.get_by_id(cid) is not None and cid not in selected_inner_chunks_ids):
                    valid_outer_chunks_ids.append(cid)

            selected_outer_chunks_ids = set()
            if valid_outer_chunks_ids:
                filter_lambda = lambda data: data["__id__"] in valid_outer_chunks_ids
                selected_outer_chunks = await text_chunks_vdb.query(query, top_k=query_param.max_outer_chunks_per_memory_point, filter_lambda=filter_lambda)
                selected_outer_chunks_ids = {chunk["id"] for chunk in selected_outer_chunks}
                final_selected_chunks_ids.update(selected_outer_chunks_ids)

            pointwise_selected_chunks_ids.append({"inner_chunks": list(selected_inner_chunks_ids), "outer_chunks": list(selected_outer_chunks_ids)})

        if len(final_selected_chunks_ids) > query_param.max_text_chunks:
            filter_lambda = lambda data: data["__id__"] in final_selected_chunks_ids
            all_query = self.get_history_subqueries_context(include_first_query=True)
            final_selected_chunks = await text_chunks_vdb.query(all_query, top_k=query_param.max_text_chunks, filter_lambda=filter_lambda)
            final_selected_chunks_ids = [text_chunk["id"] for text_chunk in final_selected_chunks]

        if not final_selected_chunks_ids:
            all_text_chunks = []
        else:
            # Asynchronously fetch all unique chunks data.
            chunk_data_tasks = [text_chunks_db.get_by_id(c_id) for c_id in final_selected_chunks_ids]
            fetched_chunks_data = await asyncio.gather(*chunk_data_tasks)

            valid_text_chunks = [
                {"id": c_id, "data": data}
                for c_id, data in zip(list(final_selected_chunks_ids), fetched_chunks_data)
                if data and "content" in data
            ]

            # Sort chunks for coherent presentation (e.g., by document and position).
            valid_text_chunks.sort(key=lambda x: (x["data"].get("full_doc_id", ""), x["data"].get("chunk_order_index", 0)))

            # Truncate the list to fit within the token limit.
            truncated_valid_text_chunks = truncate_list_by_token_size(
                valid_text_chunks,
                key=lambda x: x["data"]["content"],
                max_token_size=query_param.max_token_for_final_text_chunks,
            )
            all_text_chunks = [t["data"] for t in truncated_valid_text_chunks]

        text_chunks_context = build_text_chunks_context(all_text_chunks)

        memory_related_info = f"""-----Entities-----\n""" \
                              f"""```csv\n{entities_context}\n```\n""" \
                              f"""-----Sources-----\n""" \
                              f"""```csv\n""" \
                              f"""{text_chunks_context}\n""" \
                              f"""```\n"""
        if verbose:
            return memory_related_info, pointwise_selected_chunks_ids, final_selected_chunks_ids
        return memory_related_info

    async def get_memory_pointwise_related_info(self, knowledge_graph_inst, text_chunks_db, text_chunks_vdb, query, query_param, history_retrieved_objects=None, verbose=True):
        history_retrieved_chunks_ids = []
        for per_turn_retrieved_objects in history_retrieved_objects:
            per_turn_retrieved_text_chunks_ids = []
            for pre_query_retrieved_chunks_ids in per_turn_retrieved_objects["text_chunks_ids"]:
                per_turn_retrieved_text_chunks_ids.extend(pre_query_retrieved_chunks_ids)
            history_retrieved_chunks_ids.extend(per_turn_retrieved_text_chunks_ids)
        history_retrieved_chunks_ids = set(history_retrieved_chunks_ids)

        related_entities_data = []
        related_entities_data_dict = {}
        for mp_identifier in self._memory_points:
            for entity_name in mp_identifier:
                if entity_name not in related_entities_data_dict:
                    node = await knowledge_graph_inst.get_node(entity_name)
                    if node:
                        node_data = {"entity_name": entity_name, **node}
                        related_entities_data.append(node_data)
                        related_entities_data_dict[entity_name] = node_data

        truncate_attribute_by_token_size(related_entities_data, "description", query_param.max_token_for_entity_description)
        entities_context = build_entities_context(related_entities_data)

        pointwise_selected_chunks_ids = []
        final_selected_chunks_ids = set()
        for mp_identifier in self._memory_points:
            memory_point_info = await self.get_memory_point_info(mp_identifier)
            num_entities_k = len(mp_identifier)
            if num_entities_k == 0:
                continue

            mp_chunks_ids = set()
            for entity_name in mp_identifier:
                node_data = related_entities_data_dict.get(entity_name)
                if not node_data or not node_data.get("source_id"):
                    continue
                source_ids = set(split_string_by_multi_markers(str(node_data["source_id"]), [GRAPH_FIELD_SEP]))
                mp_chunks_ids.update(source_ids)

            valid_inner_chunks_ids = []
            for cid in mp_chunks_ids:
                if await text_chunks_db.get_by_id(cid) is not None and cid in history_retrieved_chunks_ids:
                    valid_inner_chunks_ids.append(cid)

            selected_inner_chunks_ids = set()
            if valid_inner_chunks_ids:
                filter_lambda = lambda data: data["__id__"] in valid_inner_chunks_ids
                selected_inner_chunks = await text_chunks_vdb.query(memory_point_info, top_k=min(query_param.max_inner_chunks_per_memory_point, len(valid_inner_chunks_ids)), filter_lambda=filter_lambda)
                selected_inner_chunks_ids = {chunk["id"] for chunk in selected_inner_chunks}
                final_selected_chunks_ids.update(selected_inner_chunks_ids)

            # 获取outer chunk
            outer_chunks_ids = set()
            for entity_name in mp_identifier:
                node_data = related_entities_data_dict.get(entity_name)
                if not node_data:
                    continue
                # 获取邻居节点
                neighbor_nodes = await knowledge_graph_inst.get_neighbor_nodes(entity_name)
                for neighbor_node in neighbor_nodes:
                    if neighbor_node not in related_entities_data_dict:
                        neighbor_data = await knowledge_graph_inst.get_node(neighbor_node)
                        if neighbor_data and neighbor_data.get("source_id"):
                            neighbor_source_ids = set(split_string_by_multi_markers(str(neighbor_data["source_id"]), [GRAPH_FIELD_SEP]))
                            outer_chunks_ids.update(neighbor_source_ids)

            valid_outer_chunks_ids = []
            for cid in outer_chunks_ids:
                if (await text_chunks_db.get_by_id(cid) is not None and cid not in selected_inner_chunks_ids and cid in history_retrieved_chunks_ids):
                    valid_outer_chunks_ids.append(cid)

            selected_outer_chunks_ids = set()
            if valid_outer_chunks_ids:
                filter_lambda = lambda data: data["__id__"] in valid_outer_chunks_ids
                selected_outer_chunks = await text_chunks_vdb.query(memory_point_info, top_k=query_param.max_outer_chunks_per_memory_point, filter_lambda=filter_lambda)
                selected_outer_chunks_ids = {chunk["id"] for chunk in selected_outer_chunks}
                final_selected_chunks_ids.update(selected_outer_chunks_ids)

            pointwise_selected_chunks_ids.append({"inner_chunks": list(selected_inner_chunks_ids), "outer_chunks": list(selected_outer_chunks_ids)})

        if len(final_selected_chunks_ids) > query_param.max_text_chunks:
            filter_lambda = lambda data: data["__id__"] in final_selected_chunks_ids
            all_query = self.get_history_subqueries_context(include_first_query=True)
            final_selected_chunks = await text_chunks_vdb.query(all_query, top_k=query_param.max_text_chunks, filter_lambda=filter_lambda)
            final_selected_chunks_ids = [text_chunk["id"] for text_chunk in final_selected_chunks]

        if not final_selected_chunks_ids:
            all_text_chunks = []
        else:
            # Asynchronously fetch all unique chunks data.
            chunk_data_tasks = [text_chunks_db.get_by_id(c_id) for c_id in final_selected_chunks_ids]
            fetched_chunks_data = await asyncio.gather(*chunk_data_tasks)

            valid_text_chunks = [
                {"id": c_id, "data": data}
                for c_id, data in zip(list(final_selected_chunks_ids), fetched_chunks_data)
                if data and "content" in data
            ]

            # Sort chunks for coherent presentation (e.g., by document and position).
            valid_text_chunks.sort(key=lambda x: (x["data"].get("full_doc_id", ""), x["data"].get("chunk_order_index", 0)))

            # Truncate the list to fit within the token limit.
            truncated_valid_text_chunks = truncate_list_by_token_size(
                valid_text_chunks,
                key=lambda x: x["data"]["content"],
                max_token_size=query_param.max_token_for_final_text_chunks,
            )
            all_text_chunks = [t["data"] for t in truncated_valid_text_chunks]

        text_chunks_context = build_text_chunks_context(all_text_chunks)

        memory_related_info = f"""-----Entities-----\n""" \
                              f"""```csv\n{entities_context}\n```\n""" \
                              f"""-----Sources-----\n""" \
                              f"""```csv\n""" \
                              f"""{text_chunks_context}\n""" \
                              f"""```\n"""
        if verbose:
            return memory_related_info, pointwise_selected_chunks_ids, list(final_selected_chunks_ids)
        return memory_related_info

    async def clear_memory(self):
        all_vertices = await self.memory_hypergraph.get_all_vertices()
        all_hyperedges = await self.memory_hypergraph.get_all_hyperedges()
        for hyperedge in all_hyperedges:
            await self.memory_hypergraph.remove_hyperedge(hyperedge)
        for vertex in all_vertices:
            await self.memory_hypergraph.remove_vertex(vertex)
        self._memory_points = []
        self._history_subqueries = []
        self._history_retrieved_text_chunks_ids = []
