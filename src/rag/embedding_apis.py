import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import httpx

# 加载环境变量
_ = load_dotenv(find_dotenv())


class OpenAIEmbedding:
    def __init__(self, model="BAAI/bge-m3"):
        self.model = model
        self.client = OpenAI(
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
            http_client=httpx.Client(verify=False),
        )

    # 新增：批量生成文档向量的方法（Chroma必需）
    def embed_documents(self, texts):
        result = []
        for i in range(0, len(texts), 64):
            embeddings = self.client.embeddings.create(
                input=texts[i : i + 64], model=self.model
            )
            result.extend([data.embedding for data in embeddings.data])
        return result

    # 可选：补充单句嵌入方法（如需单独处理查询）
    def embed_query(self, text):
        return self.embed_documents([text])[0]
