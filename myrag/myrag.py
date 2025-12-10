import asyncio
import os

from .memory import Memory

cur_dir = os.path.dirname(__file__)
import yaml
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast, Union

from .llm import (
    gpt_4o_mini_complete,
    openai_embedding,
)
from .operate import (
    chunking_by_token_size,
    extract_entities,
    node_indexing, chunk_indexing,
    naive_query, direct_query, hgmem_query,
)

from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger, set_logger,
    default_entity_description_func, default_relationship_description_func,
)
from .base import (
    BaseKVStorage,
    BaseVectorStorage,
    BaseGraphStorage,
    BaseHypergraphStorage,
    StorageNameSpace,
    QueryParam,
)

from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage, HypergraphStorage,
)

# future KG integrations

# from .kg.ArangoDB_impl import (
#     GraphStorage as ArangoDBStorage
# )


def lazy_external_import(module_name: str, class_name: str):
    """Lazily import a class from an external module based on the package of the caller."""

    # Get the caller's module and package
    import inspect

    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args, **kwargs):
        import importlib

        # Import the module using importlib
        module = importlib.import_module(module_name, package=package)

        # Get the class from the module and instantiate it
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


Neo4JStorage = lazy_external_import(".kg.neo4j_impl", "Neo4JStorage")
OracleKVStorage = lazy_external_import(".kg.oracle_impl", "OracleKVStorage")
OracleGraphStorage = lazy_external_import(".kg.oracle_impl", "OracleGraphStorage")
OracleVectorDBStorage = lazy_external_import(".kg.oracle_impl", "OracleVectorDBStorage")
MilvusVectorDBStorge = lazy_external_import(".kg.milvus_impl", "MilvusVectorDBStorge")
MongoKVStorage = lazy_external_import(".kg.mongo_impl", "MongoKVStorage")
ChromaVectorDBStorage = lazy_external_import(".kg.chroma_impl", "ChromaVectorDBStorage")
TiDBKVStorage = lazy_external_import(".kg.tidb_impl", "TiDBKVStorage")
TiDBVectorDBStorage = lazy_external_import(".kg.tidb_impl", "TiDBVectorDBStorage")
TiDBGraphStorage = lazy_external_import(".kg.tidb_impl", "TiDBGraphStorage")
AGEStorage = lazy_external_import(".kg.age_impl", "AGEStorage")
GremlinStorage = lazy_external_import(".kg.gremlin_impl", "GremlinStorage")


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


@dataclass
class MyRAG:
    working_dir: str = field(
        default_factory=lambda: f"./myrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # Default not to use embedding cache
    embedding_cache_config: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    kv_storage: str = field(default="JsonKVStorage")
    #vector_storage: str = field(default="NanoVectorDBStorage")
    entity_vector_storage: Union[str, None] = field(default="NanoVectorDBStorage")
    relationship_vector_storage: Union[str, None] = field(default="NanoVectorDBStorage")
    chunk_vector_storage: Union[str, None] = field(default="NanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")
    hypergraph_storage: str = field(default="HypergraphStorage")

    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    tiktoken_model_name: str = "gpt-4o"
    # text chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100

    # entity extraction
    max_extra_entity_gleaning: int = 0
    entity_summary_max_tokens: int = 500

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # embedding_func: EmbeddingFunc = field(default_factory=lambda:hf_embedding)
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # LLM
    llm_model_func: callable = gpt_4o_mini_complete  # hf_model_complete#
    llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"  #'meta-llama/Llama-3.2-1B'#'google/gemma-2-2b-it'
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16
    llm_model_kwargs: dict = field(default_factory=dict)

    # Retriever
    #retriever = #TODO

    # storage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    enable_llm_cache: bool = True
    entity_description_func: callable = default_entity_description_func
    relationship_description_func: callable = default_relationship_description_func

    # extension
    config_path: str = os.path.join(cur_dir, "rag_config.yaml")
    config: dict = field(default_factory=dict)
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self):
        log_file = os.path.join(self.working_dir, "myrag.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)

        logger.info(f"Logger initialized for working directory: {self.working_dir}")

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"MyRAG init with param:\n  {_print_config}\n")

        print(self.config_path)
        with open(self.config_path, 'r', encoding='utf-8') as yf:
            self.config = yaml.safe_load(yf.read())
        if "TEXT_CHUNKING" in self.config:
            if "chunk_token_size" in self.config["TEXT_CHUNKING"]:
                self.chunk_token_size = self.config["TEXT_CHUNKING"]["chunk_token_size"]
            if "chunk_overlap_token_size" in self.config["TEXT_CHUNKING"]:
                self.chunk_overlap_token_size = self.config["TEXT_CHUNKING"]["chunk_overlap_token_size"]
        if "ENTITY_EXTRACTION" in self.config:
            if "max_extra_entity_gleaning" in self.config["ENTITY_EXTRACTION"]:
                self.max_extra_entity_gleaning = self.config["ENTITY_EXTRACTION"]["max_extra_entity_gleaning"]
            if "entity_summary_max_tokens" in self.config["ENTITY_EXTRACTION"]:
                self.entity_summary_max_tokens = self.config["ENTITY_EXTRACTION"]["entity_summary_max_tokens"]

        # @TODO: should move all storage setup here to leverage initial start params attached to self.

        self.key_string_value_json_storage_cls: Type[BaseKVStorage] = (
            self._get_storage_class()[self.kv_storage]
        )
        self.entity_vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class()[
            self.entity_vector_storage
        ] if self.entity_vector_storage is not None else None
        self.relationship_vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class()[
            self.relationship_vector_storage
        ] if self.relationship_vector_storage is not None else None
        self.chunk_vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class()[
            self.chunk_vector_storage
        ] if self.chunk_vector_storage is not None else None
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class()[
            self.graph_storage
        ]
        self.hypergraph_storage_cls: Type[BaseHypergraphStorage] = self._get_storage_class()[
            self.hypergraph_storage
        ]

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache",
                global_config=asdict(self),
                embedding_func=None,
            )
            if self.enable_llm_cache
            else None
        )
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(self.embedding_func)

        ####
        # entities, relationships and text_chunks kv dbs
        ####
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        ####
        # entities, relationships and text_chunks vector dbs
        ####
        self.entities_vdb = self.entity_vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        ) if self.entity_vector_db_storage_cls is not None else None
        self.relationships_vdb = self.relationship_vector_db_storage_cls(
            namespace="relationships",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id"},
        ) if self.relationship_vector_db_storage_cls is not None else None
        self.text_chunks_vdb = self.chunk_vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        ) if self.chunk_vector_db_storage_cls is not None else None

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace="llm_response_cache",
                    global_config=asdict(self),
                    embedding_func=None,
                ),
                **self.llm_model_kwargs,
            )
        )

        for attr_name in dir(self):
            if attr_name in self.addon_params:
                setattr(self, attr_name, self.addon_params[attr_name])
        #add memory graph storage
        self.memory = Memory(self.hypergraph_storage_cls, asdict(self), self.embedding_func)

    def _get_storage_class(self) -> Type[BaseGraphStorage]:
        return {
            # kv storage
            "JsonKVStorage": JsonKVStorage,
            "OracleKVStorage": OracleKVStorage,
            "MongoKVStorage": MongoKVStorage,
            "TiDBKVStorage": TiDBKVStorage,
            # vector storage
            "NanoVectorDBStorage": NanoVectorDBStorage,
            "OracleVectorDBStorage": OracleVectorDBStorage,
            "MilvusVectorDBStorge": MilvusVectorDBStorge,
            "ChromaVectorDBStorage": ChromaVectorDBStorage,
            "TiDBVectorDBStorage": TiDBVectorDBStorage,
            # graph storage
            "NetworkXStorage": NetworkXStorage,
            "Neo4JStorage": Neo4JStorage,
            "OracleGraphStorage": OracleGraphStorage,
            "AGEStorage": AGEStorage,
            "TiDBGraphStorage": TiDBGraphStorage,
            "GremlinStorage": GremlinStorage,
            # "ArangoDBStorage": ArangoDBStorage
            "HypergraphStorage": HypergraphStorage,
        }

    async def chunkize_docs(self, docs):
        final_chunks = {}
        for doc_key, doc in tqdm_async(
                docs.items(), desc="Chunking documents", unit="doc"
        ):
            chunks = {
                compute_mdhash_id(dp["content"], prefix="chunk-"): {
                    **dp,
                    "full_doc_id": doc_key,
                }
                for dp in chunking_by_token_size(
                    doc["content"],
                    overlap_token_size=self.chunk_overlap_token_size,
                    max_token_size=self.chunk_token_size,
                    tiktoken_model=self.tiktoken_model_name,
                )
            }
            final_chunks.update(chunks)

        return final_chunks

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    async def ainsert(self, string_or_strings):
        update_storage = False
        try:
            ########################################################
            logger.info(f"[Doc Insertion]")
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            update_storage = True
            logger.info(f"Inserting {len(new_docs)} docs")
            await self.full_docs.upsert(new_docs)
            logger.info(f"Finish inserting documents")
            ########################################################
            logger.info(f"[Chunk Insertion]")
            chunks_to_insert = await self.chunkize_docs(new_docs)
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(chunks_to_insert.keys())
            )
            chunks_to_insert = {
                k: v for k, v in chunks_to_insert.items() if k in _add_chunk_keys
            }
            if not len(chunks_to_insert):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"Inserting {len(chunks_to_insert)} chunks")
            await self.text_chunks.upsert(chunks_to_insert)
            if self.text_chunks_vdb is not None:
                await self.text_chunks_vdb.upsert(chunks_to_insert)
            logger.info(f"Finish inserting chunks")
            ########################################################
            #"""
            logger.info("[Entity Extraction]...")
            maybe_new_kg = await extract_entities(
                chunks_to_insert,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                logger.warning("No new entities and relationships found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg
            #"""
        finally:
            if update_storage:
                await self._insert_done()

    def insert_from_custom_chunks(self, chunkized_doc):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert_from_custom_chunks(chunkized_doc))

    async def ainsert_from_custom_chunks(self, chunkized_docs):
        update_storage = False
        try:
            ########################################################
            logger.info(f"[Doc Insertion]")
            if not isinstance(chunkized_docs, list):
                chunkized_docs = [chunkized_docs]
            new_docs = {}
            for c in chunkized_docs:
                if "doc_id" in c:
                    new_doc = {
                        c['doc_id']: {"content": c["content"].strip(), "chunks": c["chunks"]}
                    }
                else:
                    new_doc = {
                        compute_mdhash_id(c["content"].strip(), prefix="doc-"): {"content": c["content"].strip(),
                                                                                 "chunks": c["chunks"]}
                    }
                new_docs.update(new_doc)
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            update_storage = True
            logger.info(f"Inserting {len(new_docs)} docs")
            await self.full_docs.upsert(new_docs)
            logger.info(f"Finish inserting documents")
            ########################################################
            logger.info(f"[Chunk Insertion]")
            chunks_to_insert = {}
            for doc_key, doc in tqdm_async(
                new_docs.items(), desc="Chunking documents", unit="doc"
            ):
                chunks = {dp["chunk_id"]: {**dp, "full_doc_id": doc_key} for dp in doc["chunks"]}
                chunks_to_insert.update(chunks)
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(chunks_to_insert.keys())
            )
            chunks_to_insert = {
                k: v for k, v in chunks_to_insert.items() if k in _add_chunk_keys
            }
            if not len(chunks_to_insert):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"Inserting {len(chunks_to_insert)} chunks")
            await self.text_chunks.upsert(chunks_to_insert)
            if self.text_chunks_vdb is not None:
                await self.text_chunks_vdb.upsert(chunks_to_insert)
            logger.info(f"Finish inserting chunks")
            ########################################################
            #"""
            logger.info("[Entity Extraction]...")
            maybe_new_kg = await extract_entities(
                chunks_to_insert,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                logger.warning("No new entities and relationships found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg
            #"""
        finally:
            if update_storage:
                await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.relationships_vdb,
            self.text_chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def insert_custom_kg(self, custom_kg: dict):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert_custom_kg(custom_kg))

    async def ainsert_custom_kg(self, custom_kg: dict):
        update_storage = False
        try:
            # Insert chunks into vector storage
            all_chunks_data = {}
            chunk_to_source_map = {}
            for chunk_data in custom_kg.get("chunks", []):
                chunk_content = chunk_data["content"]
                source_id = chunk_data["source_id"]
                chunk_id = compute_mdhash_id(chunk_content.strip(), prefix="chunk-")

                chunk_entry = {"content": chunk_content.strip(), "source_id": source_id}
                all_chunks_data[chunk_id] = chunk_entry
                chunk_to_source_map[source_id] = chunk_id
                update_storage = True

            if self.text_chunks_vdb is not None and all_chunks_data:
                await self.text_chunks_vdb.upsert(all_chunks_data)
            if self.text_chunks is not None and all_chunks_data:
                await self.text_chunks.upsert(all_chunks_data)

            # Insert entities into knowledge graph
            all_entities_data = []
            for entity_data in custom_kg.get("entities", []):
                entity_name = entity_data["entity_name"].upper()
                entity_type = entity_data.get("entity_type", "UNKNOWN")
                description = entity_data.get("description", "No description provided")
                # source_id = entity_data["source_id"]
                source_chunk_id = entity_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Entity '{entity_name}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Prepare node data
                node_data = {
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_id,
                }
                # Insert node data into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_node(
                    entity_name, node_data=node_data
                )
                node_data["entity_name"] = entity_name
                all_entities_data.append(node_data)
                update_storage = True

            # Insert relationships into knowledge graph
            all_relationships_data = []
            for relationship_data in custom_kg.get("relationships", []):
                src_id = relationship_data["src_id"].upper()
                tgt_id = relationship_data["tgt_id"].upper()
                description = relationship_data["description"]
                keywords = relationship_data["keywords"]
                weight = relationship_data.get("weight", 1.0)
                # source_id = relationship_data["source_id"]
                source_chunk_id = relationship_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Relationship from '{src_id}' to '{tgt_id}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Check if nodes exist in the knowledge graph
                for need_insert_id in [src_id, tgt_id]:
                    if not (
                        await self.chunk_entity_relation_graph.has_node(need_insert_id)
                    ):
                        await self.chunk_entity_relation_graph.upsert_node(
                            need_insert_id,
                            node_data={
                                "source_id": source_id,
                                "description": "UNKNOWN",
                                "entity_type": "UNKNOWN",
                            },
                        )

                # Insert edge into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_edge(
                    src_id,
                    tgt_id,
                    edge_data={
                        "weight": weight,
                        "description": description,
                        "keywords": keywords,
                        "source_id": source_id,
                    },
                )
                edge_data = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                }
                all_relationships_data.append(edge_data)
                update_storage = True

            # Insert entities into vector storage if needed
            if self.entities_vdb is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                        "content": self.entity_description_func(dp["entity_name"], dp["description"]), #dp["entity_name"] + dp["description"],
                        "entity_name": dp["entity_name"],
                    }
                    for dp in all_entities_data
                }
                await self.entities_vdb.upsert(data_for_vdb)

            # Insert relationships into vector storage if needed
            if self.relationships_vdb is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                        "src_id": dp["src_id"],
                        "tgt_id": dp["tgt_id"],
                        "content": dp["keywords"]
                        + dp["src_id"]
                        + dp["tgt_id"]
                        + dp["description"],
                    }
                    for dp in all_relationships_data
                }
                await self.relationships_vdb.upsert(data_for_vdb)
        finally:
            if update_storage:
                await self._insert_done()

    def query(self, query: str, query_param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, query_param))

    async def aquery(self, query: str, query_param: QueryParam = QueryParam()):
        print(query_param)
        if query_param.mode == "hgmem":
            num_try = 0
            while num_try < 5:
                try:
                    response = await hgmem_query(
                        query,
                        self.chunk_entity_relation_graph,
                        self.entities_vdb,
                        self.relationships_vdb,
                        self.text_chunks_vdb,
                        self.text_chunks,
                        query_param,
                        asdict(self),
                        memory=self.memory,
                        hashing_kv=self.llm_response_cache
                        if self.llm_response_cache and hasattr(self.llm_response_cache, "global_config")
                        else self.key_string_value_json_storage_cls(
                            namespace="llm_response_cache",
                            global_config=asdict(self),
                            embedding_func=None,
                        ),
                    )
                    break
                except Exception as e:
                    print(f"Encountering error: {e}")
                    num_try += 1
                finally:
                    await self.aclear_memory()
                if num_try >= 5:
                    return "Failed"

        elif query_param.mode == "naive":
            response = await naive_query(
                query,
                self.text_chunks_vdb,
                self.text_chunks,
                query_param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace="llm_response_cache",
                    global_config=asdict(self),
                    embedding_func=None,
                ),
            )
        elif query_param.mode == "light-direct":
            response = await direct_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.text_chunks,
                query_param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace="llm_response_cache",
                    global_config=asdict(self),
                    embedding_func=None,
                ),
                relationships_vdb=self.relationships_vdb,
                text_chunks_vdb=self.text_chunks_vdb
            )
        else:
            raise ValueError(f"Unknown mode {query_param.mode}")
        await self._query_done()
        return response

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def delete_by_entity(self, entity_name: str):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str):
        entity_name = entity_name.upper()

        try:
            await self.entities_vdb.delete_entity(entity_name)
            await self.relationships_vdb.delete_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self):
        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def update_graph_entity_representation(self):
        all_entities_data = self.chunk_entity_relation_graph.list_all_nodes(data=True)

        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": self.entity_description_func(dp["entity_name"], dp["description"]), #dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await self.entities_vdb.upsert(data_for_vdb)

    def indexing(self, query: str, query_param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aindexing(query, query_param))

    async def aindexing(self, query: str, query_param: QueryParam = QueryParam()):
        if query_param.mode == "node_indexing":
            indexed_data = await node_indexing(
                query,
                self.entities_vdb,
                self.chunk_entity_relation_graph,
                query_param
            )
        elif query_param.mode == "chunk_indexing":
            indexed_data = await chunk_indexing(
                query,
                self.text_chunks_vdb,
                self.text_chunks,
                query_param
            )
        else:
            raise ValueError(f"Unknown mode {query_param.mode}")

        return indexed_data

    def clear_memory(self):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aclear_memory())

    async def aclear_memory(self):
        await self.memory.clear_memory()
