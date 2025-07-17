import os
import json
import re
import pdfplumber
from PIL import Image
import io
import logging

# 配置日志记录
logging.basicConfig(
    filename='pdf_processing_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def is_valid_image(data):
    """检查数据是否为有效的图片格式"""
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # 验证图片完整性
        return True
    except Exception as e:
        logging.error(f"无效图片数据: {str(e)}")
        return False

def get_image_format(data):
    """获取图片格式"""
    try:
        img = Image.open(io.BytesIO(data))
        return img.format.lower() if img.format else 'png'
    except Exception:
        return 'png'

def extract_images_from_pdf(pdf_path, output_dir):
    """从PDF中提取图片并保存到指定目录（针对习题册图片特点优化）"""
    images = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            print(f"处理第 {page_num} 页图片...")
            page_images = page.images
            print(f"  找到 {len(page_images)} 个图像对象")
            
            for img_idx, img in enumerate(page_images):
                try:
                    # 针对习题册中可能的矢量图/嵌入式图片处理
                    img_data = img["stream"].get_data()
                    
                    # 过滤无效图片（习题册中可能存在的空白图片）
                    if len(img_data) < 100:  # 过滤极小无效数据
                        print(f"  跳过过小图像 {img_idx+1}（大小：{len(img_data)}字节）")
                        continue
                    
                    # 验证图片有效性
                    if not is_valid_image(img_data):
                        print(f"  图像 {img_idx+1} 数据无效，跳过")
                        error_path = os.path.join(output_dir, f"error_page_{page_num}_img_{img_idx}.bin")
                        with open(error_path, 'wb') as f:
                            f.write(img_data)
                        continue
                    
                    # 确定图片格式
                    img_format = get_image_format(img_data)
                    img_name = f"P{page_num}_Img{img_idx+1}.{img_format}"
                    img_path = os.path.join(output_dir, img_name)
                    
                    # 保存图片
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                    
                    images.append({
                        "name": img_name,
                        "page": page_num,
                        "path": img_path
                    })
                    print(f"  成功保存图片: {img_name}")
                    
                except Exception as e:
                    print(f"保存图片时出错: {e}")
                    logging.error(f"页面 {page_num} 图像 {img_idx+1} 保存失败: {str(e)}")
    
    print(f"总共提取出 {len(images)} 张有效图片")
    return images

def process_pdf_to_json(pdf_path, output_file):
    """处理习题册PDF，优化文本解析规则"""
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    
    # 提取图片
    image_dir = os.path.join(os.path.dirname(output_file), "../images/network_images")
    os.makedirs(image_dir, exist_ok=True)
    print(f"开始从PDF提取图片...")
    images = extract_images_from_pdf(pdf_path, image_dir)
    
    all_qa = []
    current_question = None
    current_page = 0  # 当前处理的页码
    question_pattern = re.compile(r'^\d+[.、]\s*')  # 匹配题号（支持全角符号）
    option_pattern = re.compile(r'^[A-E][.．\s]\s*')  # 匹配选项（支持全角句号）
    answer_pattern = re.compile(r'【答案】\s*([A-Z])')  # 匹配答案
    analysis_pattern = re.compile(r'【解析】\s*(.+)')  # 匹配解析
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            current_page = page_num
            text = page.extract_text()
            if not text:
                continue
                
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            for line in lines:
                line = line.replace('．', '.').replace('：', ':')
                
                if question_pattern.match(line):
                    if current_question:
                        all_qa.append(current_question)
                    
                    q_id = re.match(r'^(\d+)[.、]', line).group(1)
                    current_question = {
                        "question_id": f"N{q_id}",
                        "question": line,
                        "options": {},
                        "answer": "",
                        "analysis": "",
                        "images": [],
                        "page": page_num,
                        "subject_category": "计算机网络"
                    }
                
                elif current_question and option_pattern.match(line):
                    opt_match = re.match(r'^([A-E])[.．\s](.+)', line)
                    if opt_match:
                        opt_key = opt_match.group(1)
                        opt_val = opt_match.group(2).strip()
                        current_question["options"][opt_key] = opt_val
                
                elif current_question and "【答案】" in line:
                    ans_match = answer_pattern.search(line)
                    if ans_match:
                        current_question["answer"] = ans_match.group(1)
                
                elif current_question and "【解析】" in line:
                    ana_match = analysis_pattern.search(line)
                    if ana_match:
                        current_question["analysis"] += ana_match.group(1).strip()

    if current_question:
        all_qa.append(current_question)
    
    for qa in all_qa:
        qa["images"] = [
            img["name"] for img in images 
            if img["page"] == qa["page"]
        ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=4)
    
    print(f"成功处理PDF并保存至 {output_file}")

if __name__ == "__main__":
    pdf_path = "data/pdf_data/深度浅出计算机网络习题解答.pdf"
    output_file = "data/test_data/network_questions.json"
    
    process_pdf_to_json(pdf_path, output_file)
