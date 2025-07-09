import os
from dotenv import load_dotenv, find_dotenv
from document_processor import DocumentProcessor
from vector_db import VectorDatabase

# 加载环境变量
_ = load_dotenv(find_dotenv())

class RAGSystem:
    def __init__(self, persist_dir='../../data_base/vector_db/chroma'):
        self.document_processor = DocumentProcessor()
        self.vector_db = VectorDatabase(persist_directory=persist_dir)
        self.persist_dir = persist_dir

    def build_knowledge_base(self, data_dir='../../data_base/knowledge_db'):
        """构建知识库"""
        # 获取所有文档路径
        file_paths = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
        
        # 处理文档
        processed_docs = self.document_processor.process_documents(file_paths)
        
        # 构建向量数据库
        self.vector_db.create_from_documents(processed_docs)
        print(f"知识库构建完成，包含 {self.vector_db.get_collection_count()} 个文档块")

    def query(self, question, use_mmr=False, k=3):
        """查询知识库"""
        if not os.path.exists(self.persist_dir):
            raise ValueError("知识库不存在，请先构建知识库")
            
        if not self.vector_db.vectordb:
            self.vector_db.load_existing(self.persist_dir)
        
        if use_mmr:
            results = self.vector_db.mmr_search(question, k=k)
        else:
            results = self.vector_db.similarity_search(question, k=k)
        
        print(f"找到 {len(results)} 个相关文档块:")
        for i, doc in enumerate(results):
            print(f"\n文档块 {i+1}:")
            print(doc.page_content[:200] + "...")
        
        return results

if __name__ == "__main__":
    rag = RAGSystem()
    # 示例用法:
    rag.build_knowledge_base()
    rag.query("什么是线性表", use_mmr=True)
