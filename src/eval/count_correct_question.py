import json
import os
import re


# 检查答案是否正确
def check_ans(input_item):
    return {
        "序号": input_item["序号"],
        "测试结果": input_item["模型结果"] == input_item["正确答案"],
        "问题": input_item["问题"],
        "选项": input_item["选项"],
        "模型结果": input_item["模型结果"],
        "正确答案": input_item["正确答案"],
        "模型依据": input_item["模型依据"],
    }
    print(f"没有找到病案号为{id}的诊断结果！")


answer_result = []
output_folder = "output/400_question"
os.makedirs(output_folder, exist_ok=True)
# 读模型输出
with open(
    os.path.join(output_folder, "qwen3_8b_400_question.jsonl"), "r", encoding="utf-8"
) as file:
    for line in file:
        item = json.loads(line)
        answer_result.append(check_ans(item))

with open(
    os.path.join(output_folder, "qwen3_8b_400_question_result.json"),
    "w",
    encoding="utf-8",
) as f_result:
    correct_number = 0
    for item in answer_result:
        if item["测试结果"] is True:
            correct_number = correct_number + 1
    result_dic = {
        "正确个数": correct_number,
        "正确率": correct_number / len(answer_result),
        "详细结果": answer_result,
    }
    print(result_dic)
    f_result.write(json.dumps(result_dic, ensure_ascii=False, indent=4))
