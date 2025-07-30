import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
_ = load_dotenv(find_dotenv())


class OpenAIEmbedding:
    def __init__(self, model="BAAI/bge-m3", batch_size=64):
        self.model = model
        self.batch_size = batch_size
        self.client = OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def embed_documents(self, texts):
        """批量生成文档向量"""
        result = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            embeddings = self.client.embeddings.create(
                input=batch_texts, model=self.model
            )
            result.extend([data.embedding for data in embeddings.data])
        return result

    # 可选：补充单句嵌入方法（如需单独处理查询）
    def embed_query(self, text):
        return self.embed_documents([text])[0]
