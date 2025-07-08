import json

with open(
    "data/pdf_data/408 1000题（答案册）_v3_content_list.json", "r", encoding="utf-8"
) as file:
    data = json.load(file)
data = data[:200]
with open("data/pdf_data/sample.json", "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
