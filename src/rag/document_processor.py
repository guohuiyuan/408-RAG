import re
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.loaders = {
            "pdf": PyMuPDFLoader,
            "md": UnstructuredMarkdownLoader,
        }

    def load_documents(self, file_paths):
        """加载多种格式的文档"""
        documents = []
        for file_path in file_paths:
            file_extension = file_path.split(".")[-1]
            loader_class = self.loaders.get(file_extension)
            if loader_class:
                loader = loader_class(file_path)
                documents.extend(loader.load())
        return documents

    def clean_text(self, text):
        """清洗文本数据"""
        # 移除中日韩字符之间的换行符
        text = re.sub(r"([^\u4e00-\u9fa5\n])\n([^\u4e00-\u9fa5\n])", r"\1 \2", text)
        # 移除特殊符号和多余的空格
        text = text.replace("•", "").replace(" ", "").replace("\n\n", "\n")
        return text

    def process_documents(self, file_paths):
        """完整文档处理流程"""
        # 1. 加载文档
        docs = self.load_documents(file_paths)

        # 2. 清洗文档内容
        for doc in docs:
            doc.page_content = self.clean_text(doc.page_content)

        # 3. 分割文档
        split_docs = self.text_splitter.split_documents(docs)

        return split_docs
