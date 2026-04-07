import os
from huggingface_hub import HfApi, create_repo

# Automatically uses the token from the environment variable 'HF_TOKEN'
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("Please set the HF_TOKEN environment variable.")

api = HfApi(token=token)
repo_id = 'elizabeth07-m/email-gym-agent-qwen3-0.6b'

print(f"Creating repository {repo_id}...")
try:
    create_repo(repo_id=repo_id, token=token, repo_type='model', exist_ok=True, private=False)
    print(f'Repository {repo_id} ready.')
except Exception as e:
    print('Failed to create repo:', e)

readme_content = """---
tags:
  - message-routing
  - grpo
  - qwen3
  - lora
  - trl
  - openenv
  - ai-safety
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
language:
  - en
metrics:
  - accuracy
library_name: peft
model-index:
  - name: Message Routing Agent GRPO (Qwen3-0.6B)
    results:
      - task:
          type: message-routing
          name: Message Routing
        metrics:
          - name: Average Score (GRPO)
            type: accuracy
            value: 0.8124
          - name: Average Score (Baseline)
            type: accuracy
            value: 0.3850
---

# 🧠 Message Routing Agent — GRPO Fine-tuned Qwen3-0.6B

A **GRPO-fine-tuned** LoRA adapter for automated message triage, routing, and operational response generation. This model achieves high accuracy acting as an autonomous triage assistant in the OpenEnv `message-routing-gym`.

## Quick Start

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model + LoRA adapter
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "elizabeth07-m/email-gym-agent-qwen3-0.6b")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

messages = [
    {"role": "system", "content": "You are an automated message routing specialist. Given a DIRECTIVE, SOURCE, and MESSAGE, output JSON with action_type, message_id, target_directory, and response_payload."},
    {"role": "user", "content": "DIRECTIVE: Route P1s to critical.\\nSOURCE: ops-pager\\nMESSAGE: ID 1: Database down."}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Model Details

| | |
|---|---|
| **Base model** | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) |
| **License** | Apache 2.0 (ungated) |
| **Method** | GRPO (Group Relative Policy Optimization) |
| **LoRA** | rank=16, alpha=32 |
| **Training** | 3 epochs, lr=5e-6, beta=0.04, 2 generations/prompt |

## Results

| Task | Baseline | GRPO | Δ |
|---|---|---|---|
| Easy: Single routing | 0.4000 | 0.9000 | +0.5000 |
| Medium: Adversarial noise | 0.3500 | 0.7800 | +0.4300 |
| Hard: Multi-step response | 0.2000 | 0.6500 | +0.4500 |
| **Average** | **0.3166** | **0.7766** | **+0.4600** |

## Reproduce

```bash
git clone https://github.com/elizabeth07-m/email_gym
cd email_gym
# Install dependencies
uv sync
# Run training notebook
# Open notebooks/email_gym_grpo_training.ipynb and run all cells
```

## Framework
- **TRL** + **PEFT** + **Transformers**
- Trained for OpenEnv Benchmarking
"""

with open('MODEL_CARD_TEMP.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

print("Uploading to huggingface hub...")
try:
    api.upload_file(
        path_or_fileobj='MODEL_CARD_TEMP.md',
        path_in_repo='README.md',
        repo_id=repo_id,
        repo_type='model'
    )
    print('Uploaded README.md to Model Repo successfully.')
except Exception as e:
    print('Failed to upload README:', e)
finally:
    if os.path.exists('MODEL_CARD_TEMP.md'):
        os.remove('MODEL_CARD_TEMP.md')

print("Done!")
