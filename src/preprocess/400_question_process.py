import os
import json
import re
from unstructured.partition.docx import partition_docx

def process_docx_to_json(source_dir, output_dir):
    """
    处理源目录中的docx文件，转换为JSON格式并保存到输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith(".docx"):
            source_path = os.path.join(source_dir, filename)
            try:
                elements = partition_docx(filename=source_path)
                content_list = [el.to_dict() for el in elements]
                base_filename = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, f"{base_filename}.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(content_list, f, ensure_ascii=False, indent=4)
                print(f"成功处理 {filename} 并保存至 {output_path}")
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

def combine_qa_from_json(source_dir, output_file):
    """
    从JSON文件中提取题目、选项和答案，并添加学科分类信息
    """
    all_qa = []
    # 定义文档前缀与学科分类的映射
    subject_mapping = {
        "ComputerArchitecture": "计算机组成原理",
        "ComputerNetwork": "计算机网络",
        "DataStructure": "数据结构",
        "OperatingSystem": "操作系统"
    }
    
    for filename in os.listdir(source_dir):
        if filename.endswith(".json"):
            source_path = os.path.join(source_dir, filename)
            with open(source_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 从文件名提取文档前缀
            doc_prefix = os.path.splitext(filename)[0]
            
            # 获取对应学科分类，默认未知类型
            subject_category = subject_mapping.get(doc_prefix, "未知类型")
            
            i = 0
            while i < len(data):
                element = data[i]
                text = element.get("text", "").strip()
                
                # 识别题号模式
                if re.match(r'^\d+[.、]', text):
                    question = text
                    options = {}
                    answer = ""
                    
                    # 提取选项
                    if i + 1 < len(data):
                        options_text = data[i+1].get("text", "").strip()
                        if options_text and options_text[0].upper() in ['A', 'B', 'C', 'D']:
                            option_pattern = r'([A-E])[.\s]+(.+?)(?=[B-E][.\s]|$)'
                            options_matches = re.findall(option_pattern, options_text, re.DOTALL)
                            for letter, content in options_matches:
                                options[letter] = content.strip()
                    
                    # 提取答案
                    answer_index = None
                    for offset in [2, 3]:
                        if i + offset < len(data):
                            answer_text = data[i+offset].get("text", "").strip()
                            if answer_text.startswith("**答案："):
                                answer = answer_text.replace("**答案：", "").replace("**", "").strip()
                                answer_index = i + offset
                                break
                    
                    if answer_index:
                        i = answer_index
                    
                    if question and options and answer:
                        all_qa.append({
                            "question": question,
                            "options": options,
                            "answer": answer,
                            "subject_category": subject_category  # 蛇形命名
                        })
                
                i += 1
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=4)
    
    print(f"成功合并所有题目到 {output_file}")

if __name__ == "__main__":
    source_directory = "data/docx_data"
    output_directory = "output/processed_data"
    combined_output_file = "data/test_data/computer_408_exam_questions_400.json"
    
    process_docx_to_json(source_directory, output_directory)
    combine_qa_from_json(output_directory, combined_output_file)