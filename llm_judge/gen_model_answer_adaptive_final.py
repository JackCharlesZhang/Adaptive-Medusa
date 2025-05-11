"""Generate answers with local models using adaptive Medusa decoding.

Usage:
python3 gen_model_answer_adaptive.py --model-path FasterDecoding/medusa-vicuna-7b-v1.3 --model-id medusa-vicuna-7b-v1.3-adaptive
"""
import argparse
import json
import os
import time
import shortuuid
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import get_conversation_template

# Medusa imports
from medusa.model.medusa_model_adaptive import MedusaModel
from medusa.model.medusa_choices import *
from medusa.model.utils import *
from medusa.model.kv_cache import initialize_past_key_values


def filter_medusa_choices(medusa_choices, max_length):
    return [choice for choice in medusa_choices if len(choice) <= max_length]

def filter_and_cache_medusa_buffers(model, medusa_choices, max_length=5):
    cached_buffers = {}
    cached_choices = {}
    
    # For each possible head length from 1 to max_length
    for length in [1,2,3,4,5]:
        
        # Filter choices to only include tuples up to the current length
        filtered_choices = filter_medusa_choices(medusa_choices, length)
        cached_choices[length] = filtered_choices
        
        # Generate and cache the buffers for this set of filtered choices
        cached_buffers[length] = generate_medusa_buffers(
            filtered_choices, device=model.base_model.device
        )

    return cached_buffers, cached_choices

@torch.inference_mode()
def predict_accept_length(
    logits: torch.Tensor,      # shape: [n_heads, 1, 1, vocab_size]
    alpha: float,              # weight for log(top-1 prob)
    beta: float,               # weight for entropy
    threshold: float,          # score cutoff for early rejection
    decay_lambda: float,       # decay factor for head position (e.g. 0.9)
    gamma: float,              # EMA momentum (e.g. 0.9)
    entropy_cap: float,        # hard cutoff to abort if entropy too high (e.g. 6.5)
) -> int:
    """
    Predict how many speculative tokens to decode (i.e., number of Medusa heads),
    using a fast EMA-smoothed scoring function based on entropy and top-1 probability.

    Returns:
        predicted_length (int): number of speculative tokens to decode (min 1).
    """
    if logits.numel() == 0:
        return 1  # fallback — always decode at least one token

    # === Step 1: reduce logits to [n_heads, vocab_size]
    logits = logits[:, 0, 0, :]  # squeeze intermediate dims

    # === Step 2: compute softmax once
    probs = torch.softmax(logits, dim=-1)
    top_probs, _ = torch.max(probs, dim=-1)

    # Cast to float32 for entropy calculation to avoid NaNs
    probs_fp32 = probs.to(torch.float32)
    log_probs = torch.log(probs_fp32 + 1e-9)
    entropies = -torch.sum(probs_fp32 * log_probs, dim=-1)                 # shape: [n_heads]

    # === Step 4: early exit if entropy is too high
    # this prevents decoding very uncertain heads
    above_cap = (entropies > entropy_cap)
    if torch.any(above_cap):
        cutoff = torch.nonzero(above_cap)[0].item()
    else:
        cutoff = len(entropies)  # usually 5 heads max

    # === Step 5: compute headwise score = alpha * log(p1) - beta * entropy
    scores = alpha * torch.log(top_probs[:cutoff] + 1e-9) - beta * entropies[:cutoff]

    # === Step 6: apply decay: weight early heads more (e.g. 1.0, 0.9, 0.81, ...)
    decay_weights = decay_lambda ** torch.arange(cutoff, device=scores.device)
    scores = scores * decay_weights

    # === Step 7: apply EMA smoothing to score
    score_ema = 0.0
    for i in range(cutoff):
        score_ema = gamma * score_ema + (1 - gamma) * scores[i]
        if score_ema <= threshold:
            return i  # accept up to i heads (0-based => i tokens)

    # === Step 8: if all scores pass threshold
    return cutoff  # accept all heads up to cutoff

def medusa_forward(input_ids, model, tokenizer, medusa_choices, temperature, posterior_threshold, posterior_alpha, top_p=0.8, sampling='typical', fast=True, max_steps=512):
    """
    Forward pass using Medusa's adaptive decoding strategy.
    Directly implements adaptive decoding logic similar to the original version.
    
    Args:
        input_ids: Input token IDs
        model: Medusa model
        tokenizer: Tokenizer for the model
        medusa_choices: Tree structures for Medusa decoding
        temperature: Temperature for sampling
        posterior_threshold: Threshold for posterior validation
        posterior_alpha: Another threshold parameter
        top_p: Top-p for nucleus sampling
        sampling: Sampling strategy ('typical' or 'nucleus')
        fast: Whether to use fast decoding
        max_steps: Maximum number of decoding steps
        
    Returns:
        Tuple of (output_ids, new_token_count, steps_taken)
    """
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()

    #     
    # Precompute all relevant buffers if not already cached
    if not hasattr(model, "adaptive_medusa_buffers") or model.adaptive_medusa_buffers is None:
        model.adaptive_medusa_buffers, model.adaptive_medusa_choices = filter_and_cache_medusa_buffers(
            model, medusa_choices, max_length=5 
        )

    # Initialize the past key and value states
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.base_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]

    reset_medusa_mode(model)
    
    # Initialize with maximum number of heads for initial processing
    max_heads = model.medusa
    initial_buffers = model.adaptive_medusa_buffers[max_heads]
    
    # Process prefill tokens and initialize
    medusa_logits_all, logits = initialize_medusa(
        input_ids, model, initial_buffers["medusa_attn_mask"], past_key_values
    )

    new_token = 0

    last_k = -1
    cached_medusa_buffers = None
    cached_medusa_logits = None
        
    # Main generation loop
    for idx in range(max_steps):
        # Choose number of heads dynamically
        # For fixed head count for testing:
        current_k = predict_accept_length(
            medusa_logits_all[:, :, -1:, :],
            alpha=4.9722,
            beta=4.9985,
            threshold=-3.5936,
            decay_lambda=0.9995,
            gamma=0.952,
            entropy_cap=6.7334,
            )
         
        
        if current_k == 0:
            # sample next_token
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            new_token += 1

            # --- zero‐case initialize ---
            last_token = next_token.unsqueeze(0) if next_token.ndim == 1 else next_token
            medusa_logits_all, logits = initialize_medusa(
                last_token,
                model,
                initial_buffers["medusa_attn_mask"],
                past_key_values
            )

        else:

            # Could be replaced with adaptive head selection logic
            if current_k != last_k:
                cached_medusa_buffers = model.adaptive_medusa_buffers[current_k]
                cached_medusa_logits = medusa_logits_all[:current_k]
                last_k = current_k
            medusa_buffers = cached_medusa_buffers
            medusa_logits = cached_medusa_logits
            # Get buffers for current head count
            medusa_buffers = model.adaptive_medusa_buffers[current_k]
            
            # Slice only the logits needed for this iteration
            medusa_logits = medusa_logits_all[:current_k]
            
            # Generate candidates using current tree structure
            candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                medusa_buffers["tree_indices"],
                medusa_buffers["retrieve_indices"],
                temperature, posterior_threshold, posterior_alpha, top_p, sampling, fast
            )
            
            # Tree decoding with appropriate attention mask for current head count
            medusa_logits_all, logits, outputs = tree_decoding(
                model,
                tree_candidates,
                past_key_values,
                medusa_buffers["medusa_position_ids"],
                input_ids,
                medusa_buffers["retrieve_indices"],
                medusa_attn_mask=medusa_buffers["medusa_attn_mask"]
            )
            
            
            # Determine which tokens to accept
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p, sampling, fast
            )
            
            # Update input_ids and related state variables
            input_ids, logits, medusa_logits_all, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                outputs,
                logits,
                medusa_logits_all,
                new_token,
                past_key_values_data,
                current_length_data,
            )
        
        # Check for EOS token
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
            
        # Safety check for max tokens
        if new_token > 1024:
            break
            
    return input_ids, new_token, idx

def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    temperature,
    posterior_threshold,
    posterior_alpha,
    top_p,
    sampling,
    fast,
    medusa_choices,
    adaptive=True,  # Added adaptive flag
):
    questions = load_questions(question_file, question_begin, question_end)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                posterior_threshold,
                posterior_alpha,
                sampling,
                top_p,
                fast,
                medusa_choices,
                adaptive,  # Pass adaptive flag
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    temperature,
    posterior_threshold,
    posterior_alpha,
    sampling,
    top_p,
    fast,
    medusa_choices,
    adaptive=True,  # Added adaptive flag
):
    # Medusa model setup
    num_heads = -1
    for choice in medusa_choices:
        if len(choice) > num_heads:
            num_heads = len(choice)

    model = MedusaModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = model.get_tokenizer()
    
    model.eval()
    print('Check model training state:', model.training)
    
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)
    print('Using adaptive Medusa decoding:', adaptive)
    
    question = questions[0]

    # warmup
    print('Starting warmup...')
    for _ in range(3):
        conv = get_conversation_template(model_id)
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer([prompt]).input_ids

            # some models may error out when generating long outputs
            try:
                torch.cuda.synchronize()
                start_time = time.time()
                output_ids, new_token, idx = medusa_forward(
                    torch.as_tensor(input_ids).cuda(),
                    model,
                    tokenizer,
                    medusa_choices,
                    0.7,  # Fixed warmup temperature
                    posterior_threshold,
                    posterior_alpha,
                    top_p=top_p,
                    sampling=sampling,
                    fast=fast,
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                output_ids = output_ids[0][len(input_ids[0]) :]
                # be consistent with the template's stop_token_ids
                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                if conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()
            except RuntimeError as e:
                print(e)
                print("ERROR question ID: ", question["question_id"])
                output = "ERROR"

            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            conv.messages[-1][-1] = output
    print('Warmup done')

    # Main evaluation loop
    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temp = temperature_config[question["category"]]
        else:
            temp = temperature if temperature > 0 else 0.7

        choices = []
        for i in range(num_choices):
            conv = get_conversation_template(model_id)
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids

                # some models may error out when generating long outputs
                try:
                    torch.cuda.synchronize()
                    start_time = time.time()
                    output_ids, new_token, idx = medusa_forward(
                        torch.as_tensor(input_ids).cuda(),
                        model,
                        tokenizer,
                        medusa_choices,
                        temp,
                        posterior_threshold,
                        posterior_alpha,
                        top_p=top_p,
                        sampling=sampling,
                        fast=fast,
                    )
                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    
                    output_ids = output_ids[0][len(input_ids[0]) :]

                    # be consistent with the template's stop_token_ids
                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                conv.messages[-1][-1] = output
            
            choices.append({
                "index": i, 
                "turns": turns, 
                "idxs": idxs, 
                "new_tokens": new_tokens, 
                "wall_time": wall_time
            })

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    # Medusa args
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for medusa sampling.",
    )
    parser.add_argument(
        "--posterior-threshold",
        type=float,
        default=0.09,
        help="The posterior threshold for medusa sampling.",
    )
    parser.add_argument(
        "--posterior-alpha",
        type=float,
        default=0.3,
        help="The posterior alpha for medusa sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="The top-p for medusa sampling.",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="typical",
        help="The sampling method for medusa sampling.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Whether to use fast decoding.",
    )
    parser.add_argument(
        "--medusa-choices",
        type=str,
        default="mc_sim_7b_63",
        help="The medusa choices for medusa sampling.",
    )
    # Add new adaptive flag
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Whether to use adaptive Medusa decoding with dynamic head switching.",
    )

    parser.add_argument("--n",
    type=int,
    default="5",
    help="The number medusa heads",
    )

    args = parser.parse_args()

    # Include adaptive in the model ID if enabled
    adaptive_tag = "-adaptive" if args.adaptive else ""
    args.model_id = (args.model_id + 
                   "-temperature-" + str(args.temperature) + 
                   "-posterior_threshold-" + str(args.posterior_threshold) + 
                   "-posterior_alpha-" + str(args.posterior_alpha) + 
                   "-top_p-" + str(args.top_p) + 
                   "-sampling-" + args.sampling + 
                   "-fast-" + str(args.fast) + 
                   adaptive_tag)
    
    args.medusa_choices = eval(args.medusa_choices)
    
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray
        ray.init()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args.posterior_threshold,
        args.posterior_alpha,
        args.top_p,
        args.sampling,
        args.fast,
        args.medusa_choices,
        args.adaptive,  # Pass the adaptive flag
    )

    reorg_answer_file(answer_file)
