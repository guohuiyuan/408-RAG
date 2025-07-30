import json
from pymilvus import CollectionSchema, FieldSchema, MilvusClient, DataType
from tqdm import tqdm
from embedding_apis import OpenAIEmbedding
from langchain.schema import Document


class VectorDatabase:
    def __init__(self, embedding=None, persist_directory=None):
        self.embedding = embedding if embedding else OpenAIEmbedding()
        self.persist_directory = persist_directory
        self.vectordb = None

    def create_from_documents(
        self, documents, collection_name="rag_collection", persist_directory=None
    ):
        """从文档创建向量数据库"""
        if persist_directory:
            self.persist_directory = persist_directory

        # 初始化Milvus客户端
        self.vectordb = MilvusClient(self.persist_directory)

        # 定义字段
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
        schema = CollectionSchema(fields, "RAG Collection")

        index_params = self.vectordb.prepare_index_params()
        index_params.add_index(
            field_name="embedding", index_type="FLAT", metric_type="IP"
        )

        # 创建集合（如果不存在）
        if self.vectordb.has_collection(collection_name=collection_name):
            self.vectordb.drop_collection(collection_name=collection_name)
        self.vectordb.create_collection(
            collection_name=collection_name, schema=schema, index_params=index_params
        )

        embeddings = self.embedding.embed_documents(
            [doc.page_content for doc in documents]
        )
        # 插入文档
        docs_to_insert = [
            {
                "text": doc.page_content,
                "embedding": embeddings[i],
                "metadata": json.dumps(doc.metadata),
            }
            for i, doc in enumerate(documents)
        ]

        batch_size = 100
        for i in tqdm(range(0, len(docs_to_insert), batch_size), desc="插入进度"):
            batch = docs_to_insert[i : i + batch_size]
            self.vectordb.insert(collection_name=collection_name, data=batch)

        return self.vectordb

    def load_existing(self, persist_directory):
        """加载已有的向量数据库"""
        self.persist_directory = persist_directory
        self.vectordb = MilvusClient(self.persist_directory)
        return self.vectordb

    def similarity_search(self, query, k=3, collection_name="rag_collection"):
        """相似度搜索"""
        if not self.vectordb:
            raise ValueError("Vector database not initialized")

        query_embedding = self.embedding.embed_query(query)
        results = self.vectordb.search(
            collection_name=collection_name,
            data=[query_embedding],
            limit=k,
            output_fields=["text", "metadata"],
        )
        return [
            Document(page_content=hit["text"], metadata=json.loads(hit["metadata"]))
            for res in results
            for hit in res
        ]

    def get_collection_count(self, collection_name="rag_collection"):
        """获取向量库中的文档数量"""
        if not self.vectordb:
            raise ValueError("Vector database not initialized")
        stats = self.vectordb.get_collection_stats(collection_name)
        return stats["row_count"]
