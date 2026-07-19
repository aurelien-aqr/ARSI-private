# RUNBOOK — LoRA fine-tuning of the vlm_05 crop judge (RTX 3080 Ti)

Goal: teach the judge to stop hallucinating on our cameras' crop pairs
(the measured FP templates: "seat cushion shifted", equipment bag, onboard
display) while keeping its recall. Design and rationale: `docs/LORA_PLAN.md`.
Everything below runs on the GPU workstation unless marked LAPTOP.

## 0. One-time setup (workstation)

```bash
# in a fresh venv with CUDA torch already working
pip install "llamafactory[torch,metrics]" bitsandbytes
# llama.cpp for the GGUF conversion (step 5)
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && cmake -B build && cmake --build build -j && cd ..
```

## 1. Label data (LAPTOP, ARSI Studio)

Open a finished job → Results → **Review**: judge every kept box TP/FP,
draw the boxes the model missed, confirm each frame. The staged-objects
video job (20260714-151709) alone yields ~230 samples once reviewed.
Aim for **≥300 crops total with a YES:NO ratio between 1:3 and 3:1**
(`stats.json` will warn otherwise). Reviews of cross-session empty-video
runs are the most valuable NO source — they are the deployment FPs.

## 2. Export the dataset (LAPTOP)

Either click **Export dataset** in the app's LoRA tab (writes
`data/app/lora_dataset/`), or from the CLI:

```bash
venv/bin/python tools/export_lora_dataset.py --out data/app/lora_dataset
cp tools/lora/dataset_info.json data/app/lora_dataset/
```

Do NOT pass `--include-benchmark` unless you accept losing the 29-case
benchmark as an eval set. Copy `data/lora_dataset/` to the workstation
(`scp -r` or a USB stick — it is a few MB of crops).

## 3. Train (~minutes at this dataset size)

```bash
# fix dataset_dir in tools/lora/qwen3vl_lora.yaml first, and verify the
# template name (comment at the top of the yaml)
llamafactory-cli train tools/lora/qwen3vl_lora.yaml
```

Watch eval loss between epochs — with a few hundred samples, 3 epochs is
already generous; stop at the epoch where val loss turns.

## 4. Merge the adapter

QLoRA note: merge onto the **unquantized** base (standard practice — drop
`quantization_bit` at export time, LLaMA-Factory refuses otherwise):

```bash
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
  --adapter_name_or_path out/qwen3vl-arsi-lora \
  --template qwen3_vl --trust_remote_code true \
  --export_dir out/qwen3vl-arsi-merged
```

## 5. Convert to GGUF + reuse the official mmproj

The vision tower and projector were frozen (yaml), so the **official**
Qwen mmproj GGUF stays byte-for-byte valid — only the text backbone
changed:

```bash
python llama.cpp/convert_hf_to_gguf.py out/qwen3vl-arsi-merged --outfile arsi-judge-f16.gguf
llama.cpp/build/bin/llama-quantize arsi-judge-f16.gguf arsi-judge-q4_k_m.gguf Q4_K_M
# mmproj: download once from Qwen/Qwen3-VL-8B-Instruct-GGUF on Hugging Face
```

## 6. Serve — try Ollama, fall back to llama-server

**Ollama import** (KNOWN-FRAGILE for vision models — Gemma3/qwen35moe had
mmproj-import bugs; qwen3-vl is a natively supported arch so it may work):

```bash
cat > Modelfile <<'EOF'
FROM ./arsi-judge-q4_k_m.gguf
FROM ./mmproj-Qwen3-VL-8B-Instruct-f16.gguf
EOF
ollama create arsi-judge -f Modelfile
ollama run arsi-judge "describe this image" ./any.jpg   # MUST mention image content
```

If the answer ignores the image, vision import is broken → **fallback**:

```bash
llama.cpp/build/bin/llama-server -m arsi-judge-q4_k_m.gguf \
  --mmproj mmproj-Qwen3-VL-8B-Instruct-f16.gguf --port 11435
```

and point the pipeline at it via an OpenAI-compat shim for
`arsi_core.OllamaClient(impl=...)` (~30 lines, not written yet — ask
Claude Code when needed).

## 7. Evaluate — the untouched benchmark decides

```bash
# exactly like the base-model sweeps (RUNBOOK_GPU.md): fresh cache, judge = arsi-judge
cd benchmark && python run_benchmark.py   # with MODEL_NAME=arsi-judge
```

Compare against `report_lenient_qwen3vl.md` / the GLM report. **Go/no-go
(docs/LORA_PLAN.md): region precision must gain ≥ +0.10 with object recall
dropping ≤ 0.02 — otherwise the adapter is not worth the serving
complexity.** Also sanity-check output format: 20 random crops must still
answer `YES <label>` / `NO` (a fine-tune that breaks the format silently
zeroes the parser).

## Copy-back discipline

Bring back to the laptop: `out/qwen3vl-arsi-lora/` (adapter, small),
`benchmark/report*.md`, `benchmark/cache.json`, and nothing else — never
overwrite working-dir files wholesale (this bit us on 2026-07-13).
