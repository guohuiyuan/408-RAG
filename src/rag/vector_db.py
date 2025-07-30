import json
from pymilvus import CollectionSchema, FieldSchema, MilvusClient, DataType
from tqdm import tqdm
from embedding_apis import OpenAIEmbedding
from langchain.schema import Document
import os


class VectorDatabase:
    def __init__(self, embedding=None, persist_directory=None):
        self.embedding = embedding if embedding else OpenAIEmbedding()
        self.persist_directory = persist_directory
        self.vectordb = None

    def create_from_documents(self, documents, persist_directory=None):
        """从文档创建向量数据库"""
        if persist_directory:
            self.persist_directory = persist_directory

        # 初始化Milvus客户端
        self.vectordb = MilvusClient(self.persist_directory)

        # pk为索引键，text为表项内容，sparse_vector和dense_vector为稠密和稀疏两种向量
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=True,
                max_length=100,
            ),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=256),
        ]
        schema = CollectionSchema(fields, "")

        index_params = self.vectordb.prepare_index_params()

        index_params.add_index(
            field_name="embedding", index_type="FLAT", metric_type="IP"
        )

        # 创建集合（如果不存在）
        collection_name = "rag_collection"
        if self.vectordb.has_collection(collection_name=collection_name):
            self.vectordb.drop_collection(collection_name=collection_name)
        self.vectordb.create_collection(
            collection_name=collection_name, schema=schema, index_params=index_params
        )

        embeddings = self.embedding.embed_documents(
            [doc.page_content for doc in documents]
        )
        # 插入文档
        docs = []
        for i, doc in enumerate(documents):
            docs.append(
                {
                    "text": doc.page_content,
                    "embedding": embeddings[i],
                    "metadata": json.dumps(doc.metadata),
                }
            )
        batch_size = 100
        for i in tqdm(range(0, len(docs), batch_size), desc="插入进度"):
            _ = self.vectordb.insert(
                collection_name=collection_name, data=docs[i : i + batch_size]
            )
        return self.vectordb

    def load_existing(self, persist_directory):
        """加载已有的向量数据库"""
        self.persist_directory = persist_directory
        self.vectordb = MilvusClient(self.persist_directory)
        return self.vectordb

    def similarity_search(self, query, k=3):
        """相似度搜索"""
        if not self.vectordb:
            raise ValueError("Vector database not initialized")

        query_embedding = self.embedding.embed_query(query)
        results = self.vectordb.search(
            collection_name="rag_collection",
            data=[query_embedding],
            limit=k,
            output_fields=["text", "metadata"],
        )
        return [
            Document(page_content=hit["text"], metadata=json.loads(hit["metadata"]))
            for hit in results[0]
        ]

    def mmr_search(self, query, k=3):
        """最大边际相关性搜索"""
        if not self.vectordb:
            raise ValueError("Vector database not initialized")

        # Milvus暂不支持MMR搜索，使用相似度搜索替代
        return self.similarity_search(query, k=k)

    def get_collection_count(self):
        """获取向量库中的文档数量"""
        if not self.vectordb:
            raise ValueError("Vector database not initialized")
        stats = self.vectordb.get_collection_stats("rag_collection")
        return stats["row_count"]
