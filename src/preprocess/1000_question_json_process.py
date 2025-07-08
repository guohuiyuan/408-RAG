import json
import re
from collections import defaultdict


def process_sample_json(input_path, output_path):
    """Process sample.json into the target question format with enhanced features"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    current_question = None
    current_options = {}
    current_subject = "计算机综合"
    current_knowledge = []
    page_images = defaultdict(list)
    answer_keys = {}

    # First pass: collect images and answer keys
    for item in data:
        # Collect images by page
        if item["type"] == "image":
            page_images[item["page_idx"]].append(item["img_path"])

    # Second pass: process questions
    for item in data:
        if item["type"] != "text":
            continue

        text = item["text"].strip()
        if not text:
            continue

        # Update current subject from section headers
        if text in ["数据结构", "计算机网络", "操作系统", "计算机组成原理"]:
            current_subject = text
            current_knowledge = []

        # Update knowledge points from chapter headers
        if re.match(r"^第\w+章\s*", text):
            current_knowledge = [text]

        # Check for question number pattern (e.g. "11.")
        q_match = re.match(r"^(\d+)\.\s*(.*)", text)
        if q_match:
            if current_question:
                questions.append(current_question)

            q_id = int(q_match.group(1))
            q_text = q_match.group(2)

            current_question = {
                "question_id": f"Q{q_id}",
                "question": q_text,
                "options": {},
                "answer": answer_keys.get(q_id, ""),
                "analysis": "",
                "knowledge_points": current_knowledge.copy(),
                "images": page_images.get(item["page_idx"], []),
                "page": item["page_idx"] + 1,  # Convert 0-based to 1-based
                "subject_category": current_subject,
            }
            current_options = {}

        # Check for options (A:, B:, etc.)
        elif current_question and re.match(r"^[A-E][.．、]\s*", text):
            opt_matches = re.findall(r"([A-E])[.．、]\s*(.*?)(?=[A-E][.．、]|$)", text)
            for opt_match in opt_matches:
                opt_key = opt_match[0]  # 选项字母（A/B/C/D等）
                opt_val = opt_match[1].strip()  # 选项内容
                current_question["options"][opt_key] = opt_val

        # Check for answer (override table answer if present)
        elif current_question and "答案：" in text:
            ans_match = re.search(r"答案[:：]\s*([A-E])", text)
            if ans_match:
                current_question["answer"] = ans_match.group(1)

        # Check for analysis
        elif current_question and "【解析】" in text:
            ana_match = re.search(r"【解析】\s*(.+)", text)
            if ana_match:
                current_question["analysis"] = ana_match.group(1)

    # Add last question if exists
    if current_question:
        questions.append(current_question)

    # Save to output file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    input_file = "data/408_1000/408 1000题（答案册）_v3_content_list.json"
    output_file = "data/test_data/computer_408_exam_questions_slice_1000.json"
    process_sample_json(input_file, output_file)
