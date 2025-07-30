import os
import logging
from dotenv import load_dotenv, find_dotenv
from document_processor import DocumentProcessor
from vector_db import VectorDatabase
from llm_apis import LLMClient

# 配置日志记录
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 加载环境变量
load_dotenv(find_dotenv())


class RAGSystem:
    def __init__(self, persist_dir):
        self.document_processor = DocumentProcessor()
        self.vector_db = VectorDatabase(persist_directory=persist_dir)
        self.llm_client = LLMClient()
        self.persist_dir = persist_dir

    def build_knowledge_base(self, data_dir):
        """构建知识库"""
        # 获取所有文档路径
        file_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(data_dir)
            for file in files
        ]

        # 处理文档
        processed_docs = self.document_processor.process_documents(file_paths)

        # 构建向量数据库
        self.vector_db.create_from_documents(processed_docs)
        logging.info(
            f"知识库构建完成，包含 {self.vector_db.get_collection_count()} 个文档块"
        )

    def query(self, question, k=3):
        """查询知识库并生成答案"""
        if not os.path.exists(self.persist_dir):
            raise ValueError("知识库不存在，请先构建知识库")

        if not self.vector_db.vectordb:
            self.vector_db.load_existing(self.persist_dir)

        # 检索相关文档
        retrieved_docs = self.vector_db.similarity_search(question, k=k)
        context = [doc.page_content for doc in retrieved_docs]

        logging.info(f"找到 {len(retrieved_docs)} 个相关文档块.")

        # 生成答案
        answer = self.llm_client.generate_answer(question, context)
        logging.info(f"生成的答案: {answer}")

        return answer


if __name__ == "__main__":
    # 定义项目根目录
    project_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # 定义数据路径
    persist_directory = os.path.join(project_dir, "data_base/vector_db/408.db")
    knowledge_base_dir = os.path.join(project_dir, "data_base/knowledge_db")

    # 确保目录存在
    os.makedirs(os.path.dirname(persist_directory), exist_ok=True)

    # 初始化RAG系统
    rag_system = RAGSystem(persist_dir=persist_directory)

    # 构建知识库
    logging.info("开始构建知识库...")
    rag_system.build_knowledge_base(data_dir=knowledge_base_dir)
    logging.info("知识库构建完成。")

    # 执行查询
    logging.info("执行示例查询...")
    answer = rag_system.query("什么是操作系统？")
    logging.info(f"最终答案: {answer}")
    logging.info("查询完成。")
