import asyncio
import html
import os
from queue import Queue

from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import dataclass
from typing import Any, Union, cast, Tuple, List, Dict, Set, Optional
import networkx as nx
import numpy as np
from nano_vectordb import NanoVectorDB
from hyperdb import HypergraphDB

from .utils import (
    logger,
    load_json,
    write_json,
    compute_mdhash_id,
)

from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    BaseHypergraphStorage,
)


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        left_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left_data)
        return left_data

    async def drop(self):
        self._data = {}


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i: i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        async def wrapped_task(batch):
            result = await self.embedding_func(batch)
            pbar.update(1)
            return result

        embedding_tasks = [wrapped_task(batch) for batch in batches]
        pbar = tqdm_async(
            total=len(embedding_tasks), desc="Generating embeddings", unit="batch"
        )
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        if len(embeddings) == len(list_data):
            for i, d in enumerate(list_data):
                d["__vector__"] = embeddings[i]
            results = self._client.upsert(datas=list_data)
            return results
        else:
            # sometimes the embedding is not returned correctly. just log it.
            logger.error(
                f"embedding is not 1-1 with data, {len(embeddings)} != {len(list_data)}"
            )

    async def query(self, query: str, top_k=5, filter_lambda=None):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
            filter_lambda=filter_lambda
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def get(self, ids):
        return self._client.get(ids)

    @property
    def client_storage(self):
        return getattr(self._client, "_NanoVectorDB__storage")

    def upsert_matrix(self, data, matrix):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]

        if len(matrix) == len(list_data):
            for i, d in enumerate(list_data):
                d["__vector__"] = matrix[i]
            results = self._client.upsert(datas=list_data)
            return results
        else:
            # sometimes the embedding is not returned correctly. just log it.
            logger.error(
                f"embedding is not 1-1 with data, {len(matrix)} != {len(list_data)}"
            )

    async def delete_entity(self, entity_name: str):
        try:
            entity_id = [compute_mdhash_id(entity_name, prefix="ent-")]

            if self._client.get(entity_id):
                self._client.delete(entity_id)
                logger.info(f"Entity {entity_name} have been deleted.")
            else:
                logger.info(f"No entity found with name {entity_name}.")
        except Exception as e:
            logger.error(f"Error while deleting entity {entity_name}: {e}")

    async def delete_relation(self, entity_name: str):
        try:
            relations = [
                dp
                for dp in self.client_storage["data"]
                if dp["src_id"] == entity_name or dp["tgt_id"] == entity_name
            ]
            ids_to_delete = [relation["__id__"] for relation in relations]

            if ids_to_delete:
                self._client.delete(ids_to_delete)
                logger.info(
                    f"All relations related to entity {entity_name} have been deleted."
                )
            else:
                logger.info(f"No relations found for entity {entity_name}.")
        except Exception as e:
            logger.error(
                f"Error while deleting relations for entity {entity_name}: {e}"
            )

    async def index_done_callback(self):
        self._client.save()


@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {
            node: html.unescape(node.upper().strip()) for node in graph.nodes()
        }  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        return self._graph.degree(node_id)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return self._graph.degree(src_id) + self._graph.degree(tgt_id)

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def delete_node(self, node_id: str):
        """
        Delete a node from the graph based on the specified node_id.

        :param node_id: The node_id to delete
        """
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)
            logger.info(f"Node {node_id} deleted from the graph.")
        else:
            logger.warning(f"Node {node_id} not found in the graph for deletion.")

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def list_all_nodes(self, data=False):
        return self._graph.nodes(data=data)

    async def list_all_edges(self, data=False):
        return self._graph.edges(data=data)

    async def get_neighbor_nodes(self, node_id: str) -> List[str]:
        return list(self._graph.neighbors(node_id))

    async def get_k_hop_nodes(self, node_id: str, k=1, mode="nodes_at_the_k_hop") -> List[str] | Dict[str | int, int | List[str]]:
        if k < 0:
            raise ValueError(f"Invalid k, {k} < 0")
        already_visited = {node_id: 0, 0: [node_id]}
        for i in range(k):
            nodes_of_next_hop = []
            for node_id in already_visited[i]:
                for neighbor in self._graph.neighbors(node_id):
                    if neighbor not in already_visited:
                        already_visited[neighbor] = i+1
                        nodes_of_next_hop.append(neighbor)
            if len(nodes_of_next_hop) == 0:
                break
            already_visited[i+1] = nodes_of_next_hop

        if mode == "nodes_at_the_k_hop":
            return already_visited[k]
        elif mode == "nodes_within_k_hops":
            all_nodes_within_k_hop = []
            for i in range(k+1):
                if i not in already_visited or len(already_visited[i]) == 0:
                    break
                all_nodes_within_k_hop.extend(already_visited[i])
            return all_nodes_within_k_hop
        elif mode == "nodes_mapping_within_k_hops":
            return already_visited
        else:
            raise NotImplementedError(f"Invalid mode: {mode}")

    async def get_all_reachable_nodes(self, node_id: str) -> List[str]:
        reachable_node_ids = {node_id: 1}
        queue = Queue()
        queue.put(node_id)
        while not queue.empty():
            cur_node_id = queue.get()
            edges = await self.get_node_edges(cur_node_id)
            for edge in edges:
                src_node_id, tgt_node_id = edge[0], edge[1]
                assert src_node_id != tgt_node_id
                new_node_id = tgt_node_id if src_node_id == cur_node_id else src_node_id
                if new_node_id not in reachable_node_ids:
                    queue.put(new_node_id)
                    reachable_node_ids[new_node_id] = 1
        return list(reachable_node_ids.keys())

    async def get_all_components(self, return_mapping=False) -> Tuple[List[List[str]], Dict[str, List[str]]] | List[List[str]]:
        components = []
        node_components_mapping = {}
        for entity_name in (await self.list_all_nodes()):
            if entity_name not in node_components_mapping:
                reachable_node_ids = await self.get_all_reachable_nodes(entity_name)
                components.append(reachable_node_ids)
                for node_id in reachable_node_ids:
                    node_components_mapping[node_id] = reachable_node_ids
        if return_mapping:
            return components, node_components_mapping
        else:
            return components

    # @TODO: NOT USED
    async def _node2vec_embed(self):
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids


@dataclass
class HypergraphStorage(BaseHypergraphStorage):
    @staticmethod
    def load_hypergraph(file_name) -> HypergraphDB:
        if os.path.exists(file_name):
            pre_hypergraph = HypergraphDB()
            pre_hypergraph.load(file_name)
            return pre_hypergraph
        return None

    @staticmethod
    def write_hypergraph(hypergraph: HypergraphDB, file_name):
        logger.info(
            f"Writing hypergraph with {hypergraph.num_v} vertices, {hypergraph.num_e} hyperedges"
        )
        hypergraph.save(file_name)

    def __post_init__(self):
        self._hgdb_file = os.path.join(
            self.global_config["working_dir"], f"hypergraph_{self.namespace}.hgdb"
        )
        preloaded_hypergraph = HypergraphStorage.load_hypergraph(self._hgdb_file)
        if preloaded_hypergraph is not None:
            logger.info(
                f"Loaded hypergraph from {self._hgdb_file} with {preloaded_hypergraph.num_v} vertices, {preloaded_hypergraph.num_e} hyperedges"
            )
        self._hypergraph = preloaded_hypergraph or HypergraphDB()
        # self._node_embed_algorithms = {
        #     "node2vec": self._node2vec_embed,
        # }

    async def index_done_callback(self):
        HypergraphStorage.write_hypergraph(self._hypergraph, self._hgdb_file)

    async def has_vertex(self, v_id: Any) -> bool:
        return self._hypergraph.has_v(v_id)

    async def has_hyperedge(self, e_tuple: Union[List, Set, Tuple]) -> bool:
        return self._hypergraph.has_e(e_tuple)

    async def get_vertex(self, v_id: str, default: Any = None) :
        return self._hypergraph.v(v_id)

    async def get_hyperedge(self, e_tuple: Union[List, Set, Tuple], default: Any = None) :
        return self._hypergraph.e(e_tuple)

    async def get_all_vertices(self):
        return self._hypergraph.all_v

    async def get_all_hyperedges(self):
        return self._hypergraph.all_e

    async def get_num_of_vertices(self):
        return self._hypergraph.num_v

    async def get_num_of_hyperedges(self):
        return self._hypergraph.num_e

    async def upsert_vertex(self, v_id: Any, v_data: Optional[Dict] = None) :
        return self._hypergraph.add_v(v_id, v_data)

    async def upsert_hyperedge(self, e_tuple: Union[List, Set, Tuple], e_data: Optional[Dict] = None) :
        return self._hypergraph.add_e(e_tuple, e_data)

    async def remove_vertex(self, v_id: Any) :
        return self._hypergraph.remove_v(v_id)

    async def remove_hyperedge(self, e_tuple: Union[List, Set, Tuple]) :
        return self._hypergraph.remove_e(e_tuple)

    async def vertex_degree(self, v_id: Any) -> int:
        return self._hypergraph.degree_v(v_id)

    async def hyperedge_degree(self, e_tuple: Union[List, Set, Tuple]) -> int:
        return self._hypergraph.degree_e(e_tuple)

    async def get_nbr_e_of_vertex(self, v_id: Any) -> list:
        """
            Return the incident hyperedges of the vertex.
        """
        return self._hypergraph.nbr_e_of_v(v_id)

    async def get_nbr_v_of_hyperedge(self, v_id: Any, exclude_self=True) -> list:
        """
            Return the incident vertices of the hyperedge.
        """
        return self._hypergraph.nbr_v_of_e(v_id)

    async def get_nbr_v_of_vertex(self, v_id: Any, exclude_self=True) -> list:
        """
            Return the neighbors of the vertex.
        """
        return self._hypergraph.nbr_v(v_id)

    # async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
    #     if algorithm not in self._node_embed_algorithms:
    #         raise ValueError(f"Node embedding algorithm {algorithm} not supported")
    #     return await self._node_embed_algorithms[algorithm]()
    #
    # async def _node2vec_embed(self):
    #     from graspologic import embed
    #
    #     embeddings, nodes = embed.node2vec_embed(
    #         self._graph,
    #         **self.global_config["node2vec_params"],
    #     )
    #
    #     nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
    #     return embeddings, nodes_ids
