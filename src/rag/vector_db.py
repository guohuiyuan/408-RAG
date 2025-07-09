from langchain_community.vectorstores import Chroma
from embedding_apis import ZhipuEmbedding
from embedding_apis import OpenAIEmbedding
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
        
        # 确保目录存在
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.vectordb = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding,
            persist_directory=self.persist_directory
        )
        return self.vectordb

    def load_existing(self, persist_directory):
        """加载已有的向量数据库"""
        self.persist_directory = persist_directory
        self.vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding
        )
        return self.vectordb

    def similarity_search(self, query, k=3):
        """相似度搜索"""
        if not self.vectordb:
            raise ValueError("Vector database not initialized")
        return self.vectordb.similarity_search(query, k=k)

    def mmr_search(self, query, k=3):
        """最大边际相关性搜索"""
        if not self.vectordb:
            raise ValueError("Vector database not initialized")
        return self.vectordb.max_marginal_relevance_search(query, k=k)

    def get_collection_count(self):
        """获取向量库中的文档数量"""
        if not self.vectordb:
            raise ValueError("Vector database not initialized")
        return self.vectordb._collection.count()
