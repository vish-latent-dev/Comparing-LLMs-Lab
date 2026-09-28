# Comparing LLMs Lab

**Names:** Vishwajit Narayanan, Arjun Mysani
**Job:** General Everyday Assistant

## Models

| | Model | Source |
|---|---|---|
| Model A | Qwen3.5-4B (Q4_K_M) | `bartowski/Qwen_Qwen3.5-4B-GGUF` |
| Model B | Gemma-2-2B-it (Q4_K_M) | `bartowski/gemma-2-2b-it-GGUF` |

## Setup (in the codespace terminal)

```bash
bash setup.sh            # installs llama-cpp-python + huggingface_hub, downloads the class model
python chat_runner.py    # chat with the model; type exit to leave
```
