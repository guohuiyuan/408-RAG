from openai import OpenAI
import json
from tqdm import tqdm
import random
import os
import copy

if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

client = OpenAI(api_key="YOUR_API_KEY", base_url="http://localhost:8000/v1")
model_name = client.models.list().data[0].id

test_path = "data/test_data/computer-408-exam-questions-400.json"

output_folder = "output/400_question"
os.makedirs(output_folder, exist_ok=True)


# 系统消息模板
system_message = {
    "role": "system",
    "content": """你作为精通计算机知识的专业答题助手，需依据计算机相关知识点对选择题进行解答。本次任务的题目为计算机领域选择题，包含题干、选项及相关计算机背景信息。请严格遵循以下规则：
1、答案仅限从题目给定的选项中选取，禁止脱离选项范围进行选择
2、输出结果需包含两个部分：
① 第一部分明确列出正确的答案选项
② 第二部分以计算机知识逻辑形式阐述选择该答案的依据
两个部分以 "&&" 键连接
输出示例：B&&CPU（中央处理器）是计算机的核心部件，主要功能是执行指令，进行算术运算、逻辑运算等数据处理操作。选项 A 数据存储由内存、硬盘等存储设备负责；选项 C 数据输入由键盘、鼠标等输入设备完成；选项 D 数据输出由显示器、打印机等输出设备实现。因此，CPU 的核心功能是数据处理，答案选 B。
""",
}


def history_chat(messages):
    """使用已设置的system消息和用户问题进行对话"""
    # 合并system消息和用户消息
    full_messages = [system_message] + messages
    print(full_messages)

    # 调用API
    response = client.chat.completions.create(
        model=model_name,
        messages=full_messages,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    return response.choices[0].message.content


# 读取输入数据
test_data = []
all_data = []
with open(f"{test_path}", "r", encoding="utf-8") as file:
    all_data = json.load(file)
# print(data)
data_len = len(all_data)  # len(all_data)
random.shuffle(all_data)
test_data = copy.deepcopy(all_data)
for item in test_data:
    # 先获取所有键的列表（副本），再遍历删除
    for key in list(item.keys()):
        if key not in ["question", "options"]:  # 仅保留question和options字段
            item.pop(key, None)
# 处理并写入结果


with open(
    os.path.join(output_folder, "qwen3_8b_400_question.jsonl"), "a", encoding="utf-8"
) as f:
    for index, item in tqdm(enumerate(test_data, start=1), total=data_len):
        print(f"\n处理第 {index} 条记录")

        # 只构建用户问题，不包含system模板
        user_question = [{"role": "user", "content": f"问题和选项:{item}"}]

        # 调用API获取回答
        answer = history_chat(user_question)
        print(answer)
        # 移除前后的空白字符
        first_part, second_part = answer.split("&&", 1)
        print(first_part)
        print(second_part)
        output = {
            "序号": index,
            "问题": all_data[index - 1]["question"],
            "选项": all_data[index - 1]["options"],
            "模型结果": first_part,
            "正确答案": all_data[index - 1]["answer"],
            "模型依据": second_part,
        }
        # 写入结果
        json.dump(output, f, ensure_ascii=False)
        f.write("\n")

    print(f"共写入{index}条记录")
