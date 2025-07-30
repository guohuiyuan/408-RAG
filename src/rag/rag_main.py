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
    def __init__(self, persist_dir, strategy="default"):
        self.strategy = strategy
        self.document_processor = DocumentProcessor(strategy=self.strategy)
        self.vector_db = VectorDatabase(persist_directory=persist_dir)
        self.llm_client = LLMClient()
        self.persist_dir = persist_dir

    def build_knowledge_base(self, data_dir):
        """构建知识库"""
        if len(os.listdir(os.path.dirname(self.persist_dir))) > 0:
            logging.info("知识库已存在，跳过构建。")
            return
        # 获取所有文档路径
        file_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(data_dir)
            for file in files
        ]

        # 处理文档
        processed_docs = self.document_processor.process_documents(file_paths)

        # 保存处理后的文档以供检查
        output_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "output",
            self.strategy,
        )

        # Clear existing files in the directory
        if os.path.exists(output_dir):
            for root, dirs, files in os.walk(output_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))

        os.makedirs(output_dir, exist_ok=True)

        for doc in processed_docs:
            source_filename = os.path.basename(
                doc.metadata.get("source", "unknown_file")
            )
            filename_base, _ = os.path.splitext(source_filename)

            file_output_dir = os.path.join(output_dir, filename_base)
            os.makedirs(file_output_dir, exist_ok=True)

            # Find the next available chunk number
            chunk_num = 1
            while os.path.exists(
                os.path.join(file_output_dir, f"chunk_{chunk_num}.txt")
            ):
                chunk_num += 1

            with open(
                os.path.join(file_output_dir, f"chunk_{chunk_num}.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(doc.page_content)

        logging.info(f"切割后的文档已保存到 {output_dir}")

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
    # strategy: "default", "paper", "chapter"
    # "default": 默认切割方式，使用RecursiveCharacterTextSplitter
    # "paper": 按论文结构切割，使用PaperTextSplitter
    # "chapter": 按章节标题切割，使用ChapterTitleSplitter
    rag_system = RAGSystem(persist_dir=persist_directory, strategy="chapter")

    # 构建知识库
    logging.info("开始构建知识库...")
    rag_system.build_knowledge_base(data_dir=knowledge_base_dir)
    logging.info("知识库构建完成。")

    # 执行查询
    logging.info("执行示例查询...")
    answer = rag_system.query("什么是操作系统？")
    logging.info(f"最终答案: {answer}")
    logging.info("查询完成。")
