import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import requests
import json
from sparkai.embedding.spark_embedding import Embeddingmodel
from zhipuai import ZhipuAI
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

    def embed(self, text: str):
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding
    
    # 新增：批量生成文档向量的方法（Chroma必需）
    def embed_documents(self, texts):
        # 调用OpenAI API批量生成嵌入
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        # 提取嵌入向量并返回
        return [data.embedding for data in response.data]

    # 可选：补充单句嵌入方法（如需单独处理查询）
    def embed_query(self, text):
        return self.embed_documents([text])[0]


class WenxinEmbedding:
    def __init__(self):
        self.api_key = os.environ["QIANFAN_AK"]
        self.secret_key = os.environ["QIANFAN_SK"]

    def embed(self, text: str):
        url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={0}&client_secret={1}".format(
            self.api_key, self.secret_key
        )
        response = requests.request(
            "POST", url, headers={"Content-Type": "application/json"}
        )

        url = (
            "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1?access_token="
            + str(response.json().get("access_token"))
        )
        payload = json.dumps({"input": [text]})
        response = requests.request(
            "POST", url, headers={"Content-Type": "application/json"}, data=payload
        )
        return json.loads(response.text)["data"][0]["embedding"]


class XunfeiEmbedding:
    def __init__(self, domain="para"):
        self.embedding = Embeddingmodel(
            spark_embedding_app_id=os.environ["IFLYTEK_SPARK_APP_ID"],
            spark_embedding_api_key=os.environ["IFLYTEK_SPARK_API_KEY"],
            spark_embedding_api_secret=os.environ["IFLYTEK_SPARK_API_SECRET"],
            spark_embedding_domain=domain,
        )

    def embed(self, text: str):
        return self.embedding.embedding(text={"content": text, "role": "user"})


class ZhipuEmbedding:
    def __init__(self):
        self.client = ZhipuAI(api_key=os.environ["ZHIPUAI_API_KEY"])

    def embed(self, text: str):
        response = self.client.embeddings.create(
            model="embedding-3",
            input=text,
        )
        return response.data[0].embedding
