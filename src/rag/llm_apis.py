import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv


class LLMClient:
    def __init__(self):
        load_dotenv(find_dotenv())
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")
        self.model_name = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3-8B")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_answer(self, question, context):
        """
        Generates an answer using the LLM based on the provided question and context.
        """
        context_str = "\n\n".join(context)
        prompt = f"请根据以下提供的知识回答问题：\n\n{context_str}\n\n问题：{question}"
        print(f"LLM Input: {prompt}")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个问答机器人，请根据提供的背景知识回答问题。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
