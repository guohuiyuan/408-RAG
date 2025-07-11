import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def load_documents(self, file_paths):
        """加载多种格式的文档"""
        documents = []
        for file_path in file_paths:
            file_type = file_path.split(".")[-1]
            if file_type == "pdf":
                loader = PyMuPDFLoader(file_path)
            elif file_type == "md":
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                continue
            documents.extend(loader.load())
        return documents

    def clean_text(self, text):
        """清洗文本数据"""
        # 处理PDF中的多余换行
        pattern = re.compile(r"[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]", re.DOTALL)
        text = re.sub(pattern, lambda match: match.group(0).replace("\n", ""), text)

        # 去除特殊符号和多余空格
        text = text.replace("•", "").replace(" ", "")

        # 处理Markdown中的多余换行
        text = text.replace("\n\n", "\n")

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
