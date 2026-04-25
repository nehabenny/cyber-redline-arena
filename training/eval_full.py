#!/usr/bin/env python3
"""
Re-run behavioral comparison with longer generation (400 tokens)
to capture the actual action decisions the model makes.
Uses the SAVED LoRA adapters for post-DPO inference.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")

WIN_BASE = "/mnt/c/Users/markj/Desktop/metaprelim/cyber_arena"
sys.path.insert(0, WIN_BASE)

LORA_PATH = os.path.join(WIN_BASE, "training", "qwen-cyber-dpo-lora")
HF_TOKEN  = os.environ.get("HF_TOKEN", "")
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

from server.env import CyberRedlineEnv, SCENARIOS
from server.generate_dpo_dataset import state_to_natural_language

EVAL_SCENARIOS = ["CORPORATE_BREACH", "APT_CAMPAIGN", "FINANCIAL_HEIST"]

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

print("Loading base Qwen3.5-4B in 4-bit...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.bfloat16,
                          bnb_4bit_use_double_quant=True)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", token=HF_TOKEN, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-4B", quantization_config=bnb,
    device_map="auto", trust_remote_code=True, token=HF_TOKEN
)

def get_response(mdl, prompt, max_new=400):
    mdl.eval()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1800).to("cuda")
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=max_new,
                           temperature=0.1, do_sample=True,
                           pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

# ── PRE-DPO: base model (no LoRA) ───────────────────────────────────────────
print("\n=== PRE-DPO RESPONSES (base model, 400 tokens) ===")
pre_responses = {}
for scenario in EVAL_SCENARIOS:
    env = CyberRedlineEnv(fixed_scenario=scenario)
    obs = env.reset()
    prompt = state_to_natural_language(obs, SCENARIOS[scenario]["description"])
    resp = get_response(base_model, prompt)
    pre_responses[scenario] = resp
    print(f"\n[PRE {scenario}]:\n{resp}\n{'—'*60}")

# ── POST-DPO: load LoRA adapters ─────────────────────────────────────────────
print("\n\nLoading LoRA adapters from:", LORA_PATH)
finetuned_model = PeftModel.from_pretrained(base_model, LORA_PATH)
finetuned_model.eval()

print("\n=== POST-DPO RESPONSES (fine-tuned, 400 tokens) ===")
post_responses = {}
for scenario in EVAL_SCENARIOS:
    env = CyberRedlineEnv(fixed_scenario=scenario)
    obs = env.reset()
    prompt = state_to_natural_language(obs, SCENARIOS[scenario]["description"])
    resp = get_response(finetuned_model, prompt)
    post_responses[scenario] = resp
    print(f"\n[POST {scenario}]:\n{resp}\n{'—'*60}")

# ── Save updated responses ───────────────────────────────────────────────────
with open(os.path.join(WIN_BASE, "training", "pre_dpo_responses_full.json"), "w", encoding="utf-8") as f:
    json.dump(pre_responses, f, indent=2, ensure_ascii=False)
with open(os.path.join(WIN_BASE, "training", "post_dpo_responses_full.json"), "w", encoding="utf-8") as f:
    json.dump(post_responses, f, indent=2, ensure_ascii=False)

print("\n\n=== SUMMARY ===")
keywords_good = ["http_get", "probe", "stealth", "execute_exploit"]
keywords_bad  = ["nmap", "honeypot attack", "attack mfa"]
for scenario in EVAL_SCENARIOS:
    pre  = pre_responses[scenario].lower()
    post = post_responses[scenario].lower()
    print(f"\n{scenario}:")
    print(f"  PRE  good-kw: {[k for k in keywords_good if k in pre]} | bad-kw: {[k for k in keywords_bad if k in pre]}")
    print(f"  POST good-kw: {[k for k in keywords_good if k in post]} | bad-kw: {[k for k in keywords_bad if k in post]}")
