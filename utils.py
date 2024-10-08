import json


def load_json(path: str) -> dict:
    with open(path, 'r') as fp:
        obj = json.load(fp)
    return obj


def save_json(obj: dict, path: str) -> None:
    with open(path, "w") as fp:
        json.dump(obj, fp, indent=4)
    return

from typing import List


def process_mc_data(data: dict, context: List[str], answer: bool = False) -> dict:
    data_mc = {}
    data_mc["id"] = data.get("id", 0)
    data_mc["sent1"] = data.get("question", None)
    data_mc["sent2"] = ""

    for i in range(4):
        data_mc[f"ending{i}"] = context[data.get("paragraphs", None)[i]]
    
    if answer:
        data_mc["label"] = data.get("paragraphs", None).index(data.get("relevant", None))
    
    return data_mc


def process_qa_data(data: dict, context: List[str], answer: bool = False) -> dict:
    data_qa = {}
    data_qa["id"] = data.get("id", 0)
    data_qa["title"] = data.get("id", 0)
    data_qa["context"] = context[data.get("relevant", None)]
    data_qa["question"] = data.get("question", None)

    if answer:
        data_qa["answers"] = {
            "text": [data.get("answer", None).get("text", None)],
            "answer_start": [data.get("answer", None).get("start", None)]
        }
    
    return data_qa
