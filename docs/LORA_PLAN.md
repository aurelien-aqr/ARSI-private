# LoRA plan — fine-tuning the vlm_05 crop judge

Status: exploration (2026-07-19). Execution steps: `RUNBOOK_LORA.md`.
Dataset tooling: `tools/export_lora_dataset.py` + `tools/lora/`.

## Why this, why now

- **The measured bottleneck is judge precision, not localization.** The
  localizer finds 45/45 GT objects; region precision is 0.52–0.66
  depending on judge/prompt, with recurring hallucination TEMPLATES on
  crop pairs ("Blue seat cushion slightly shifted", "Black cable
  snake-like", the equipment bag, the onboard display). Prompt work
  plateaued: the lenient/conservative A/B showed the effect is
  model-dependent and small versus these FP families.
- **The literature says exactly this is what LoRA fixes on compact
  VLMs.** Borodin et al. (J. Imaging 2025, `CCTV_VLM_paper.pdf`): rank-8
  LoRA makes small VLMs competitive with much larger ones on surveillance
  anomaly detection AND collapses prompt sensitivity — our two pain
  points.
- **We now have a labelling machine.** ARSI Studio's Review mode produces
  human TP/FP verdicts + missed boxes per frame; the export tool turns
  them into training pairs rendered EXACTLY like inference
  (`vlm_05.render_crop_pair`, current PROMPT, `YES <label>` / `NO`
  targets).

## What gets trained

The **vlm_05 crop judge only** (base: Qwen3-VL-8B-Instruct). Not the
localizer (it is classical CV and already at 45/45), not the whole-frame
pipelines. Qwen3-VL over GLM as the base because its fine-tuning
toolchain (LLaMA-Factory/Unsloth) and official GGUF+mmproj artifacts are
mature; if the tuned Qwen3-VL beats GLM's region precision, it can also
displace GLM as alarm judge — one model instead of two.

## Dataset design

| Decision | Choice | Reason |
|---|---|---|
| Sample | ref\|insp side-by-side crop + current PROMPT | byte-identical to inference; `stats.json` records the prompt sha |
| Positive target | `YES <human label>` | keeps the label head useful for inventory |
| Negative target | bare `NO` | matches what the parser needs; no invented rationale |
| Sources | human review verdicts + reviewer-drawn missed boxes | model self-verdicts would be distillation, not correction |
| Split | 90/10 by FRAME | crops of one frame never straddle train/val |
| Benchmark GT | **excluded by default** | the 29-case benchmark is the only eval — `--include-benchmark` exists but is a one-way door |
| Balance guard | warn outside YES:NO ∈ [1:5, 5:1] | rush of FP-only reviews would teach "always NO" |

Volume today: reviewing the staged-video vlm_05 job (~226 kept boxes ×
72 frames) plus one cross-session empty-video vlm_05 run gets ~300–500
crops — the paper saw gains at that scale for a narrower task; treat it
as a lower bound and keep labelling through normal Studio use.

## Training recipe (fits the 12 GB 3080 Ti)

QLoRA 4-bit, rank 8, alpha 16, `lora_target: all` (LLM linears),
**vision tower + projector frozen**, batch 1 × grad-accum 8, lr 1e-4
cosine, ≤3 epochs with per-epoch val loss. Freezing vision is both the
recommended recipe for domain adaptation and the trick that keeps the
official mmproj GGUF valid at deployment (only the text backbone
changes). 8B QLoRA at these settings is documented to fit in ~10–12 GB;
Unsloth is the fallback if LLaMA-Factory OOMs.

## Deployment path (the risky part, planned honestly)

merge adapter → fp16 HF → `convert_hf_to_gguf.py` → Q4_K_M → pair with
the **official** Qwen3-VL mmproj GGUF.

1. **Try Ollama import** (two-FROM Modelfile). Risk: Ollama's vision GGUF
   import has a history of silently dropping the projector (Gemma3
   #9967, qwen35moe #14730). Test with an image question; distrust a
   text-only answer.
2. **Fallback: `llama-server --mmproj`** (known-good path) + a ~30-line
   OpenAI-compat shim injected via `OllamaClient(impl=...)` — the
   injection point already exists and is what the tests use.

## Evaluation protocol

Untouched 29-case benchmark, fresh cache, tuned model as vlm_05 judge —
identical to the base-model sweeps, so reports are directly comparable
row-by-row with `report_lenient_qwen3vl.md` and the GLM report.

**Go/no-go: region precision ≥ +0.10 vs base qwen3-vl at object recall
loss ≤ 0.02, and the output format intact on a 20-crop spot check.**
Secondary: frame specificity on the 5 cross-session negatives, and (once
run) the massive-negatives false-alarm rate on the two empty RTX videos.

## Risks / open questions

- **Template name / library drift**: verify `qwen3_vl` template id against
  the installed LLaMA-Factory (checked at the top of the yaml).
- **Tiny dataset**: overfitting shows as val-loss turn + format collapse;
  both are checked in the runbook.
- **Prompt coupling**: the adapter is trained against the current PROMPT;
  changing the prompt later means re-exporting and re-training (sha
  recorded in `stats.json`).
- **QLoRA merge mismatch** (adapter trained on 4-bit, merged onto fp16)
  is standard practice but a known small-quality lottery — the benchmark
  A/B is the arbiter.
- If Ollama import fails AND llama-server is annoying operationally,
  running the judge stage through llama-server only for vlm_05 while the
  rest stays on Ollama is fine — they are separate processes.
