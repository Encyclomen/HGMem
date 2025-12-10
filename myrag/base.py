from dataclasses import dataclass, field
from typing import TypedDict, Union, Literal, Generic, TypeVar, List, Set, Tuple, Any, Optional, Dict

import numpy as np

from .utils import EmbeddingFunc

TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)

T = TypeVar("T")


@dataclass
class QueryParam:
    mode: str = "naive"
    ignore_cache: bool = False
    save_to_cache: bool = True
    only_need_context: bool = False
    only_need_prompt: bool = False
    response_type: str = "Multiple Paragraphs"
    stream: bool = False
    max_num_turns: int = 2
    # Number of top-k items to retrieve.
    top_k: int = 60
    # Number of entities to retrieve and number of tokens for the entity descriptions
    top_k_entities: int = -1
    first_k_entities: int = -1
    llm_selection: bool = False
    llm_select_k_entities: int = -1
    max_token_for_entity_description: int = 100
    max_token_for_local_context: int = 1500
    # Number of relationships to retrieve and number of tokens for the relationship descriptions
    top_k_relationships: int = -1
    llm_select_k_relationships: int = -1
    max_token_for_relationship_description: int = 100
    max_token_for_global_context: int = 1500
    # Number of document chunks to retrieve and number of tokens for the original text chunks.
    top_k_chunks: int = -1
    llm_select_k_chunks: int = -1
    max_token_for_text_chunks: int = 4000
    max_token_for_final_text_chunks: int = 10000

    # Number of document chunks to use for final response
    max_chunks_per_memory_point: int = -1
    max_inner_chunks_per_memory_point: int = -1
    max_outer_chunks_per_memory_point: int = -1
    max_text_chunks: int = 50
    padding_to_max_text_chunks: bool = False
    max_inner_text_chunks: int = 10
    max_outer_text_chunks: int = 10


    include_node_degree: bool = True
    return_text_chunks_context: bool = True

    def __post_init__(self):
        if self.top_k_entities == -1:
            self.top_k_entities = self.top_k
        if self.top_k_relationships == -1:
            self.top_k_relationships = self.top_k
        if self.top_k_chunks == -1:
            self.top_k_chunks = self.top_k
        if self.first_k_entities == -1:
            self.first_k_entities = self.top_k_entities
        if self.max_token_for_final_text_chunks == -1:
            self.max_token_for_final_text_chunks = self.max_token_for_text_chunks


@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    async def index_done_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def query_done_callback(self):
        """commit the storage operations after querying"""
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    embedding_func: EmbeddingFunc

    async def all_keys(self) -> list[str]:
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    async def drop(self):
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc = None

    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        raise NotImplementedError

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        raise NotImplementedError

    async def delete_node(self, node_id: str):
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("Node embedding is not used in lightrag.")

    async def list_all_nodes(self, data=False):
        raise NotImplementedError("Not implemented yet")

    async def list_all_edges(self, data=False):
        raise NotImplementedError("Not implemented yet")


@dataclass
class BaseHypergraphStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc = None
    async def has_vertex(self, v_id: Any) -> bool:
        raise NotImplementedError

    async def has_hyperedge(self, e_tuple: Union[List, Set, Tuple]) -> bool:
        raise NotImplementedError

    async def get_vertex(self, v_id: str, default: Any = None) :
        raise NotImplementedError

    async def get_hyperedge(self, e_tuple: Union[List, Set, Tuple], default: Any = None) :
        raise NotImplementedError

    async def get_all_vertices(self):
        raise NotImplementedError

    async def get_all_hyperedges(self):
        raise NotImplementedError

    async def get_num_of_vertices(self):
        raise NotImplementedError

    async def get_num_of_hyperedges(self):
        raise NotImplementedError

    async def upsert_vertex(self, v_id: Any, v_data: Optional[Dict] = None) :
        raise NotImplementedError

    async def upsert_hyperedge(self, e_tuple: Union[List, Set, Tuple], e_data: Optional[Dict] = None) :
        raise NotImplementedError

    async def remove_vertex(self, v_id: Any) :
        raise NotImplementedError

    async def remove_hyperedge(self, e_tuple: Union[List, Set, Tuple]) :
        raise NotImplementedError

    async def vertex_degree(self, v_id: Any) -> int:
        raise NotImplementedError

    async def hyperedge_degree(self, e_tuple: Union[List, Set, Tuple]) -> int:
        raise NotImplementedError

    async def get_nbr_e_of_vertex(self, e_tuple: Union[List, Set, Tuple]) -> list:
        raise NotImplementedError

    async def get_nbr_v_of_hyperedge(self, v_id: Any, exclude_self=True) -> list:
        raise NotImplementedError

    async def get_nbr_v_of_vertex(self, v_id: Any, exclude_self=True) -> list:
        raise NotImplementedError
