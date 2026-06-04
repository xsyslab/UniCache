# SPDX-License-Identifier: Apache-2.0
import argparse

from transformers import AutoTokenizer

from vllm.simulator.data_parser import prepare_simulator_data_streaming_jsonl


def parser_args():
    args = argparse.ArgumentParser()
    args.add_argument("--raw_data_path",
                      type=str,
                      default="ShareGPT_cleaned_valid.json")
    args.add_argument("--save_data_path",
                      type=str,
                      default="ShareGPT_cleaned_valid_sessoin_data.json")
    args.add_argument("--tokenizer_path",
                      type=str,
                      default="Llama-3.2-1B-Instruct")
    return args.parse_args()


if __name__ == "__main__":
    args = parser_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    prepare_simulator_data_streaming_jsonl(args.raw_data_path, tokenizer,
                                           args.save_data_path)
