# retriever.py
# -*- coding: utf-8 -*-

from llama_index.core.schema import QueryBundle
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BGEM3SparseEmbeddingFunction
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from datetime import datetime
import numpy as np
import aiofiles
import asyncio
import json
import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'


class Retriever():

    def __init__(
        self,
        embedding_model: str = 'BAAI/bge-m3',
        embedding_batch_size: int = 8,
        db_url: str = 'http://localhost:19530',
        db_collection_name: str = 'default',
        rerank_model: str = 'BAAI/bge-reranker-v2-m3',
    ):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.dense_embedding = HuggingFaceEmbedding(
            model_name=embedding_model,
            embed_batch_size=embedding_batch_size,
            device=self.device,
            trust_remote_code=True,
        )

        self.vector_store = MilvusVectorStore(
            uri=db_url,
            collection_name=db_collection_name,
            dim=len(self.dense_embedding.get_query_embedding("test")),
            similarity_metric='COSINE',
            enable_dense=True,
            enable_sparse=True,
            sparse_embedding_function=BGEM3SparseEmbeddingFunction(),
        )

        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.dense_embedding
        )

        self.hybrid_retriever = self.index.as_retriever(
            vector_store_query_mode="hybrid",
            hybrid_ranker="RRFRanker",
            similarity_top_k=20,
        )

        self.dense_retriever = self.index.as_retriever(
            vector_store_query_mode="default",
            similarity_top_k=20,
        )

        self.reranker = SentenceTransformerRerank(
            model=rerank_model,
            top_n=10
        )

    def _merge_nodes(self, nodes1, nodes2):

        seen = {}

        for n in nodes1 + nodes2:
            node_id = n.node.node_id
            if node_id not in seen:
                seen[node_id] = n
            else:
                seen[node_id].score = max(
                    seen[node_id].score,
                    n.score
                )

        return list(seen.values())

    def _is_good(self, nodes, threshold):

        if not nodes:
            return False

        max_score = max([n.score for n in nodes])
        return max_score >= threshold

    async def retrieving_with_median_filter(self, query):
        '''多层检索+中位数过滤'''

        dense_nodes = await asyncio.to_thread(self.dense_retriever.retrieve, query)
        if self._is_good(dense_nodes, 0.9):
            print(f'直接检索完成!')
            median = np.median([n.score for n in dense_nodes])
            return [n for n in dense_nodes if n.score >= median]

        hybrid_nodes = await asyncio.to_thread(self.hybrid_retriever.retrieve, query)
        hybrid_nodes = self._merge_nodes(dense_nodes, hybrid_nodes)
        if self._is_good(hybrid_nodes, 0.7):
            print(f'混合检索完成!')
            median = np.median([n.score for n in hybrid_nodes])
            return [n for n in hybrid_nodes if n.score >= median]

        rerank_nodes = await asyncio.to_thread(self.reranker.postprocess_nodes, hybrid_nodes, QueryBundle(query_str=query))
        print(f'混合+重排检索完成!')
        median = np.median([n.score for n in rerank_nodes])
        return [n for n in rerank_nodes if n.score >= median]

    def sampling(self, nodes, temperature=0.2, top_p=0.8):

        scores = torch.tensor([n.score for n in nodes], dtype=torch.float32)
        probs = torch.softmax(scores / temperature, dim=0)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        sorted_nodes = [nodes[i] for i in sorted_indices.tolist()]

        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        cutoff = (cumulative_probs >= top_p).nonzero(as_tuple=True)[0][0]
        selected = sorted_nodes[:cutoff + 1]

        return selected

    async def retrieving_with_sampling(self, query):
        '''多层检索+温度核采样'''

        dense_nodes = await asyncio.to_thread(self.dense_retriever.retrieve, query)
        if self._is_good(dense_nodes, 0.9):
            print(f'直接检索完成!')
            return self.sampling(dense_nodes)

        hybrid_nodes = await asyncio.to_thread(self.hybrid_retriever.retrieve, query)
        hybrid_nodes = self._merge_nodes(dense_nodes, hybrid_nodes)
        if self._is_good(hybrid_nodes, 0.7):
            print(f'混合检索完成!')
            return self.sampling(hybrid_nodes)

        rerank_nodes = await asyncio.to_thread(self.reranker.postprocess_nodes, hybrid_nodes, QueryBundle(query_str=query))
        print(f'混合+重排检索完成!')
        return self.sampling(rerank_nodes)


query_engine = Retriever(
    embedding_model=os.getenv("EMBEDDING_MODEL"),
    embedding_batch_size=8,
    db_url=f"http://{os.getenv('MILVUS_HOST', 'localhost')}:{os.getenv('MILVUS_PORT', '19530')}",
    db_collection_name="default",
    rerank_model=os.getenv("RERANK_MODEL"),
)


_log_write_lock = asyncio.Lock()


async def main(collection_name: str, query: str):

    query_engine.vector_store.collection_name = collection_name

    nodes = await query_engine.retrieving_with_sampling(query=query)

    results = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "node_id": [n.id_ for n in nodes],
        "text": [n.text for n in nodes],
        "metadata": [n.metadata for n in nodes],
        "score": [n.score for n in nodes],
    }

    async with _log_write_lock:
        async with aiofiles.open(f"/app/logs/openai_agents_retrieve_logs.jsonl", "a", encoding="utf-8") as f:
            await f.write(json.dumps(results, ensure_ascii=False) + "\n")

    return results


if __name__ == '__main__':
    asyncio.run(main(collection_name="disney", query="上海迪士尼的营业时间"))
