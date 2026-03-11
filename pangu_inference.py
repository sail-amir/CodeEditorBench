import argparse
import asyncio
import importlib
import json
import jsonlines
import os
import sys
import time

import aiohttp
import torch
from tqdm import tqdm

from dataset import JsonlDataset, my_collate_fn

sys.path.append("./prompt_function/")


SUPPORTED_PROMPT_STYLES = {
    "wizardcoder",
    "magicoder",
    "octocoder",
    "codefuse",
    "deepseek",
    "phind",
    "codellama-inst",
    "codellama",
    "bloom",
}


def infer_prompt_style(base_model, prompt_type):
    if "Wizard" in base_model:
        return "wizardcoder", "group1"
    if "Magic" in base_model:
        return "magicoder", "group1"
    if "octo" in base_model:
        return "octocoder", "group1"
    if "codefuse" in base_model:
        return "codefuse", "group1"
    if "deepseek" in base_model:
        return "deepseek", "group1"
    if "Phind" in base_model:
        return "phind", "group1"
    if "Instruct-hf" in base_model:
        return "codellama-inst", "cot" if prompt_type == "cot" else "group1"
    if "CodeLlama-34b-hf" in base_model:
        return "codellama", "group1"
    if "bloom" in base_model:
        return "bloom", "group1"
    if "OpenCode" in base_model:
        return "deepseek", "group1"
    raise ValueError(
        "Unable to infer prompt style from --base_model. "
        "Pass --prompt_style explicitly."
    )


def extract_code_from_markdown(markdown):
    code_block_start = markdown.find("```")
    if code_block_start == -1:
        return None

    code_start = code_block_start + 3
    code_block_end = markdown.find("```", code_start)
    if code_block_end == -1:
        return None

    code = markdown[code_start:code_block_end].strip()
    first_newline = code.find("\n")
    if first_newline > 0:
        code = code[first_newline + 1 :]
    return code.strip()


def build_request_headers(args):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.auth_token}",
    }
    if args.csb_token:
        headers["csb-token"] = args.csb_token
    return headers


def build_request_payload(args, prompt, stream):
    payload = {
        "model": args.base_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "stream": stream,
    }
    if args.top_k is not None and args.top_k >= 0:
        payload["top_k"] = args.top_k
    return payload


async def generate_non_streaming(session, args, prompt):
    payload = build_request_payload(args, prompt, stream=False)

    async with session.post(
        args.api_url, json=payload, headers=build_request_headers(args)
    ) as response:
        if response.status != 200:
            error_text = await response.text()
            raise RuntimeError(f"API Error {response.status}: {error_text}")

        result = await response.json()
        content = result["choices"][0]["message"]["content"]
        code = extract_code_from_markdown(content)
        return code if code else content


async def generate_streaming(session, args, prompt):
    payload = build_request_payload(args, prompt, stream=True)

    full_content = ""
    async with session.post(
        args.api_url, json=payload, headers=build_request_headers(args)
    ) as response:
        if response.status != 200:
            error_text = await response.text()
            raise RuntimeError(f"API Error {response.status}: {error_text}")

        async for raw_line in response.content:
            line = raw_line.decode("utf-8").strip()
            if not line or line == "data: [DONE]":
                continue
            if not line.startswith("data: "):
                continue
            try:
                chunk = json.loads(line[6:])
            except Exception:
                continue
            if "choices" not in chunk or not chunk["choices"]:
                continue
            delta = chunk["choices"][0].get("delta", {})
            full_content += delta.get("content", "")

    code = extract_code_from_markdown(full_content)
    return code if code else full_content


async def generate_with_retry(session, args, prompt):
    for attempt in range(1, args.max_retries + 1):
        try:
            async with asyncio.timeout(args.request_timeout):
                if args.stream:
                    return await generate_streaming(session, args, prompt)
                return await generate_non_streaming(session, args, prompt)
        except Exception as error:
            if attempt == args.max_retries:
                return f"Error: Max attempts reached. Last error: {error}"
            await asyncio.sleep(args.retry_backoff * attempt)


async def generate_batch(batch_prompts, session, args):
    semaphore = asyncio.Semaphore(args.request_concurrency)

    async def run_single(prompt):
        async with semaphore:
            return await generate_with_retry(session, args, prompt)

    tasks = []
    prompt_index = []
    for idx, prompt in enumerate(batch_prompts):
        for _ in range(args.num_of_sequences):
            tasks.append(run_single(prompt))
            prompt_index.append(idx)

    flat_outputs = await asyncio.gather(*tasks)
    grouped_outputs = [[] for _ in batch_prompts]
    for idx, output in zip(prompt_index, flat_outputs):
        grouped_outputs[idx].append(output)
    return grouped_outputs


def build_record(args, batch, batch_prompts, idx, outputs):
    if args.dataset == "debug":
        return {
            "problem_id": batch["idx"][idx].item(),
            "completion_id": 0,
            "language": batch["code_language"][idx],
            "error_type": batch["type"][idx],
            "difficulty": batch["difficulty"][idx],
            "prompt": batch_prompts[idx],
            "code": outputs,
        }
    if args.dataset == "translate":
        return {
            "problem_id": batch["idx"][idx].item(),
            "completion_id": 0,
            "source_lang": batch["source_lang"][idx],
            "target_lang": batch["target_lang"][idx],
            "difficulty": batch["difficulty"][idx],
            "prompt": batch_prompts[idx],
            "code": outputs,
        }
    if args.dataset == "polishment":
        return {
            "problem_id": batch["idx"][idx].item(),
            "completion_id": 0,
            "language": batch["source_lang"][idx],
            "difficulty": batch["difficulty"][idx],
            "prompt": batch_prompts[idx],
            "code": outputs,
        }
    if args.dataset == "switch":
        return {
            "problem_id": batch["idx"][idx].item(),
            "completion_id": 0,
            "language": batch["language"][idx],
            "pair": batch["pair_id"][idx],
            "prompt": batch_prompts[idx],
            "code": outputs,
        }
    raise ValueError("Invalid dataset type.")


async def run_inference(args):
    module = importlib.import_module(f"prompt_function.prompt_{args.dataset}")
    if args.prompt_style:
        if args.prompt_style not in SUPPORTED_PROMPT_STYLES:
            raise ValueError(
                f"Invalid --prompt_style: {args.prompt_style}. "
                f"Choose from: {sorted(SUPPORTED_PROMPT_STYLES)}"
            )
        model_choice = args.prompt_style
        group = "cot" if args.prompt_type == "cot" and model_choice == "codellama-inst" else "group1"
    else:
        model_choice, group = infer_prompt_style(args.base_model, args.prompt_type)

    get_prompt_function_name = f"generate_prompt_{group}"
    prompt_function = getattr(module, get_prompt_function_name)

    input_data_path = args.input_data_dir + f"code_{args.dataset}_primary.jsonl"
    model_name = args.output_model_name or args.base_model.split("/")[-1].replace("-", "_").replace(".", "_")

    end = args.end_idx if args.end_idx != -1 else "end"
    if args.prompt_type == "zero":
        output_data_path = args.output_data_dir + f"code_{args.dataset}/{model_name}_{args.start_idx}_{end}.jsonl"
    elif args.prompt_type == "three":
        output_data_path = args.output_data_dir + f"code_{args.dataset}/Few_Shot_{model_name}_{args.start_idx}_{end}.jsonl"
    elif args.prompt_type == "cot":
        output_data_path = args.output_data_dir + f"code_{args.dataset}/Cot_{model_name}_{args.start_idx}_{end}.jsonl"
    else:
        raise ValueError("Invalid prompt type.")

    print(f"Input file: {input_data_path}")
    print(f"Output file: {output_data_path}")

    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)

    meta_data_flag = False
    if os.path.exists(output_data_path):
        output_data = jsonlines.open(output_data_path, mode="a", flush=True)
        with open(output_data_path, "r") as f:
            line_count = sum(1 for _ in f)
            print(f"Output file exists. Appending to {output_data_path}. Line count: {line_count}")
        if line_count >= 1:
            meta_data_flag = True
        args.start_idx = max(0, line_count - 1)
    else:
        output_data = jsonlines.open(output_data_path, mode="w", flush=True)

    if args.end_idx == -1:
        args.end_idx = None
    dataset = JsonlDataset(input_data_path)[args.start_idx : args.end_idx]
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=my_collate_fn,
    )

    if len(dataloader) == 0:
        print("No data in the dataset.")
        return

    timeout = aiohttp.ClientTimeout(total=args.request_timeout + 10)
    start_time = time.time()

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for batch in tqdm(dataloader, desc="Inference"):
            if args.prompt_type == "zero":
                batch_prompts = prompt_function(batch, model_choice, "zero")
            elif args.prompt_type == "three":
                batch_prompts = prompt_function(batch, model_choice, "three")
            elif args.prompt_type == "cot":
                batch_prompts = prompt_function(batch, model_choice, "cot")
            else:
                raise ValueError("Invalid prompt type.")

            batch_outputs = await generate_batch(batch_prompts, session, args)

            if not meta_data_flag:
                meta_data_flag = True
                meta_data = {
                    "model": args.base_model,
                    "model_url": args.api_url,
                    "greedy_search_decoding": args.num_of_sequences == 1 and args.temperature == 0.0,
                    "num_output": args.num_of_sequences,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "max_tokens": args.max_tokens,
                    "stream": args.stream,
                    "prompt_style": model_choice,
                }
                output_data.write(meta_data)

            for idx, outputs in enumerate(batch_outputs):
                output_data.write(build_record(args, batch, batch_prompts, idx, outputs))

    end_time = time.time()
    print("Time used: ", end_time - start_time)
    print("Each batch time: ", (end_time - start_time) / len(dataloader))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CodeEditorBench inference against a chat-completions endpoint."
    )
    parser.add_argument("--api_url", required=True, type=str, help="Full chat completions endpoint URL.")
    parser.add_argument("--base_model", default="pangu_auto", type=str, help="Model name sent in request payload.")
    parser.add_argument("--output_model_name", default=None, type=str, help="Optional model name used only for output filenames.")
    parser.add_argument("--prompt_style", default=None, type=str, help="Prompt template style for CodeEditorBench prompt builders.")
    parser.add_argument("--csb_token", default=None, type=str, help="Optional csb-token header value.")
    parser.add_argument("--auth_token", default="nokey", type=str, help='Bearer token for Authorization header.')
    parser.add_argument("--stream", action="store_true", help="Use streaming responses.")
    parser.add_argument("--dataset", default="debug", type=str, help="Name of dataset.")
    parser.add_argument("--input_data_dir", default="Input.jsonl", type=str, help="Path to input data directory.")
    parser.add_argument("--output_data_dir", default="Output.jsonl", type=str, help="Path to output data directory.")
    parser.add_argument("--batch_size", default=16, type=int, help="Number of problems processed per loop iteration.")
    parser.add_argument("--request_concurrency", default=16, type=int, help="Maximum concurrent endpoint requests.")
    parser.add_argument("--num_of_sequences", default=1, type=int, help="Number of generations per prompt.")
    parser.add_argument("--prompt_type", default="zero", type=str, help="Type of prompts: zero, three, or cot.")
    parser.add_argument("--start_idx", default=0, type=int, help="Start index for data processing.")
    parser.add_argument("--end_idx", default=-1, type=int, help="End index for data processing.")
    parser.add_argument("--temperature", default=None, type=float, help="Sampling temperature.")
    parser.add_argument("--top_p", default=None, type=float, help="Top-p sampling parameter.")
    parser.add_argument("--top_k", default=None, type=int, help="Top-k sampling parameter.")
    parser.add_argument("--max_tokens", default=2048, type=int, help="Maximum tokens to generate.")
    parser.add_argument("--request_timeout", default=60, type=int, help="Per-request timeout in seconds.")
    parser.add_argument("--max_retries", default=3, type=int, help="Maximum retry attempts per request.")
    parser.add_argument("--retry_backoff", default=1.5, type=float, help="Linear backoff multiplier in seconds.")
    args = parser.parse_args()

    if args.temperature is None:
        args.temperature = 0.0 if args.num_of_sequences == 1 else 0.8
    if args.top_p is None:
        args.top_p = 0.9 if args.num_of_sequences > 1 else 1.0
    if args.top_k is None:
        args.top_k = 40 if args.num_of_sequences > 1 else -1

    return args


def main():
    args = parse_args()
    print(args)
    asyncio.run(run_inference(args))


if __name__ == "__main__":
    main()
