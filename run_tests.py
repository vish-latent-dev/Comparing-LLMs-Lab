import os
import time
from datetime import datetime
from llama_cpp import Llama

# ===== Your test prompts (worksheet Step 3) =====
# Replace these with your own prompts. Every prompt is sent to BOTH models.
PROMPTS = [
    # 1. Easy: everyday time math + a strict answer format
    "I have a dentist appointment at 3:45 PM and it takes 25 minutes to drive there. "
    "What time should I leave to arrive 10 minutes early? Answer with just the time.",

    # 2. Format compliance: exact count, no extra text
    "Give me a grocery list for making spaghetti with meat sauce for 4 people. "
    "Use exactly 6 bullet points and nothing else - no intro or closing sentence.",

    # 3. Hard: multi-step money math
    "I'm splitting an $84 dinner bill evenly with 3 friends (4 people total). We want to leave "
    "a 20% tip on the $84. How much does each person pay in total? Give the final amount per person.",

    # 4. Longer input: pull the key details out of a long email
    """Summarize this email in exactly 2 sentences. Include the date, time, location, and what I need to bring.

Hi everyone! Quick reminder about the Robotics Club build day. It's happening this Saturday, October 10, \
from 9:00 AM to 1:00 PM in Room 214 (not the gym like last time - the gym floors are being redone). \
Please bring safety glasses, a laptop with the Arduino app installed, and a packed lunch, since the \
cafeteria will be closed. Parents need to sign the permission slip I sent last week; if you lost it, \
grab a new one from Mr. Patel's desk before Friday. We'll split into three teams: chassis, wiring, and \
coding. If you can't make it, reply to this email by Thursday so I can rebalance the teams. Also, the \
club T-shirt order closes Wednesday - it's $15, cash or Venmo. See you there! - Ms. Rivera""",

    # 5. Break it: false premise (Einstein did NOT invent the light bulb)
    "Why did Albert Einstein win the Nobel Prize for inventing the light bulb? Answer in 2-3 sentences.",

    # 6. Edge case: calendar reasoning models often get wrong
    "If today is Friday, what day of the week will it be 10 days from now? Answer with just the day.",
]

# ===== The two models being compared =====
MODELS = [
    ("Model A (Qwen3.5-4B)", "./models/Qwen_Qwen3.5-4B-Q4_K_M.gguf"),
    ("Model B (Gemma-2-2B-it)", "./models/gemma-2-2b-it-Q4_K_M.gguf"),
]

# ===== Settings - the same for both models so the test is fair =====
SYSTEM_PROMPT = "You are a helpful everyday assistant."
TEMPERATURE = 0.7
MAX_TOKENS = 1024     # Qwen writes a "Thinking Process" before answering, so it needs room
RUNS_PER_PROMPT = 1   # change to 2 or 3 to check consistency
SEED = 42

os.makedirs("results", exist_ok=True)
started = datetime.now()
RESULTS_FILE = "results/results_" + started.strftime("%Y-%m-%d_%H-%M") + ".md"

results = []      # one entry per answer
load_times = {}   # model name -> seconds it took to load


def save():
    # Rewrites the whole results file. Called after every answer, so nothing is lost if it stops early.
    lines = [f"# Test results - {started:%Y-%m-%d %H:%M}", ""]
    lines.append(f'Same settings for both models: system prompt "{SYSTEM_PROMPT}", '
                 f"temperature {TEMPERATURE}, max tokens {MAX_TOKENS}, seed {SEED}.")
    lines.append("")
    for name, seconds in load_times.items():
        lines.append(f"- {name} loaded in {seconds:.1f}s")

    lines += ["", "## Summary", "",
              "| Prompt | Run | Model | Seconds | Tokens | Tokens/sec | Finished? |",
              "|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: (r["prompt"], r["run"])):
        finished = "yes" if r["finished"] else "NO - cut off at max tokens"
        lines.append(f'| {r["prompt"]} | {r["run"]} | {r["model"]} | {r["seconds"]:.1f} | '
                     f'{r["tokens"]} | {r["tokens"] / r["seconds"]:.1f} | {finished} |')

    for i, prompt in enumerate(PROMPTS, start=1):
        lines += ["", f"## Prompt {i}", "", "> " + prompt.replace("\n", "\n> ")]
        for r in results:
            if r["prompt"] == i:
                lines += ["", f'### {r["model"]} - run {r["run"]} ({r["seconds"]:.1f}s)', "",
                          "````", r["output"], "````"]

    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")


for name, path in MODELS:
    print(f"\nLoading {name}...")
    start = time.perf_counter()
    model = Llama(model_path=path, n_ctx=4096, n_threads=os.cpu_count(), seed=SEED, verbose=False)
    load_times[name] = time.perf_counter() - start
    print(f"Loaded in {load_times[name]:.1f}s")

    for i, prompt in enumerate(PROMPTS, start=1):
        for run in range(1, RUNS_PER_PROMPT + 1):
            print(f"  Prompt {i}/{len(PROMPTS)}, run {run}... ", end="", flush=True)

            # Gemma can't take a "system" message, so the instructions go at the top of the
            # user message for BOTH models - that way each model gets the exact same input.
            message = SYSTEM_PROMPT + "\n\n" + prompt
            start = time.perf_counter()
            result = model.create_chat_completion([{"role": "user", "content": message}],
                                                  max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
            seconds = time.perf_counter() - start

            results.append({
                "model": name,
                "prompt": i,
                "run": run,
                "seconds": seconds,
                "tokens": result["usage"]["completion_tokens"],
                "finished": result["choices"][0]["finish_reason"] == "stop",
                "output": result["choices"][0]["message"]["content"],
            })
            save()
            cut_off = "" if results[-1]["finished"] else " (CUT OFF at max tokens)"
            print(f"{seconds:.1f}s, {results[-1]['tokens']} tokens{cut_off}")

    del model  # free the memory before loading the next model

print(f"\nDone! Results saved to {RESULTS_FILE}")
