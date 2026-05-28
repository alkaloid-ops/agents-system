# src/ingestion.py
# -*- coding: utf-8 -*-

from llama_index.core.schema import TextNode, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus.utils import BGEM3SparseEmbeddingFunction
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from pymilvus import MilvusClient
import asyncio
import tiktoken
import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# from llama_index.core.node_parser import SentenceWindowNodeParser


class ChunkingEmbeddingStoring:

    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-m3",
        embedding_batch_size: int = 8,
        chunk_size: int = None,
        overlap: int = None,
        db_url: str = "http://localhost:19530",
        db_collection_name: str = "default",
        store_batch_size: int = 256,
        overwrite: bool = False,
    ):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.embedding_model = HuggingFaceEmbedding(
            model_name=embedding_model_name,
            embed_batch_size=embedding_batch_size,
            device=self.device,
            trust_remote_code=True,
        )

        self.sparse_function = BGEM3SparseEmbeddingFunction()

        self.dimensions = len(self.embedding_model.get_query_embedding("test"))

        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            paragraph_separator="\n\n\n",
            secondary_chunking_regex="[^,.;]+[,.;]?",
        )

        self.vector_store = MilvusVectorStore(
            uri=db_url,
            collection_name=db_collection_name,
            dim=self.dimensions,
            batch_size=store_batch_size,
            similarity_metric="COSINE",
            overwrite=overwrite,
            enable_dense=True,
            enable_sparse=True,
            sparse_embedding_function=self.sparse_function,
        )

        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

    async def ingesting(self, documents):

        nodes = self.splitter.get_nodes_from_documents(documents)

        index = await asyncio.to_thread(
            VectorStoreIndex,
            nodes=nodes,
            embed_model=self.embedding_model,
            storage_context=self.storage_context,
            show_progress=True,
        )

        print(f'create {index} successfully!')


async def main():

    documents = SimpleDirectoryReader(
        input_dir='/home/strawberrycharlotte/Agents_system/knowledge_docs/disney',
        recursive=True,
        required_exts=['.txt']
    ).load_data()
    print(f'成功读取 {len(documents)} 个文档.')

    encoder = tiktoken.encoding_for_model('gpt-4o')
    docs_token = [len(encoder.encode(doc.get_content())) for doc in documents]
    print(f'文档 token 数量统计: {docs_token}')

    for doc in documents:

        metadata = doc.metadata or {}

        doc.metadata.update({
            "file_name": metadata.get("file_name"),
            "creation_date": metadata.get("creation_date"),
            "last_modified_date": metadata.get("last_modified_date"),
        })

    # client = MilvusClient("/app/data/milvus.db")
    client = MilvusClient(
        uri="http://localhost:19530")
    collection_name = "disney"

    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
        print(f"Collection {collection_name} 已删除!")

    CES = ChunkingEmbeddingStoring(
        embedding_model_name='/home/strawberrycharlotte/Agents_system/models/bge-m3',
        embedding_batch_size=8,
        chunk_size=512,
        overlap=64,
        db_url="http://localhost:19530",
        # db_url='./feishu_agent/data/milvus.db',
        db_collection_name="disney",
        store_batch_size=256,
        overwrite=False,
    )

    await CES.ingesting(documents=documents)

    stats = client.get_collection_stats(collection_name=collection_name)
    print(f"Collection: {collection_name}, 行数: {stats['row_count']}")


if __name__ == '__main__':
    asyncio.run(main())
