import argparse
import pandas as pd
from tqdm import tqdm
import json
import random
from pymilvus.model.reranker import BGERerankFunction
import subprocess

# 랜덤 시드 설정
random.seed(42)

def main(model_name, tsv_file_path, jsonl_file_path):
    # BGERerankFunction 설정
    bge_rf = BGERerankFunction(
        model_name=model_name,
        device="cuda:0"
    )

    candidate_list = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for n, line in enumerate(file):
            candidate_list.append(json.loads(line))

    result = []

    for candidate in tqdm(candidate_list):
        top_k = bge_rf(
                query=candidate["query"],
                documents=candidate["passage"],
                top_k=100,
            )
        for n, i in enumerate(top_k):
            result.append([int(candidate["qid"]), candidate["pid"][i.index], n + 1])

    result_df = pd.DataFrame(result)
    result_df.to_csv(tsv_file_path, sep='\t', index=False)

    try:
        # evaluatiojn
        command = ["python", "/home/livin/rimo/llm/msmarco/ms_marco_eval.py", "/home/livin/rimo/llm/msmarco_test/data/test_qrels.tsv", tsv_file_path] 
        result = subprocess.run(command, capture_output=True, text=True)
        stdout = result.stdout
        stdout_value = float(stdout.split("\n")[1].split(" ")[-1])

        # log save
        log_df = pd.read_csv("/home/livin/rimo/llm/msmarco_test/result/result/log.csv")
        log_step = int(model_name.split("/")[-1].split("-")[-1])
        finetuning_type = model_name.split("/")[-2]

        log_df.loc[log_df['step'] == log_step, finetuning_type] = stdout_value
        log_df.to_csv("/home/livin/rimo/llm/msmarco_test/result/result/log.csv", index=False)
        print("successfully saved log")
    except:
        print("log save error")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name', type=str, required=True, help='Model name for BGERerankFunction')
    parser.add_argument('--tsv_file_path', type=str, required=True, help='Output TSV file path')
    parser.add_argument('--jsonl_file_path', type=str, required=True, help='Input JSONL file path')

    args = parser.parse_args()
    main(args.model_name, args.tsv_file_path, args.jsonl_file_path)
