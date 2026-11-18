# retnet-300m: a 310M-parameter RetNet 2.0 for code and math, trained on a single 16 GB GPU

A from-scratch PyTorch implementation of a **~310M-parameter Retentive Network (RetNet)** language model. It has three additions aimed at code and multi-step math:
- **Delta-rule memory** with negative eigenvalues (`β_max = 2`)
- **Short causal convolution** (`K = 4`)
- **Calculator in the loop**: arithmetic is computed exactly at decode time instead of predicted

The model is pretrained on synthetic textbooks, code and step-by-step reasoning data. The whole pipeline is built for a **tight compute budget**: a single 16 GB GPU (Colab T4) in preemptible sessions, with 15 GB of cloud storage for data and checkpoints.

> **Status:** pretraining finished: 15,082 steps, about 494M tokens on one T4, val loss 1.547. Base-model results are below. Latent-reasoning post-training (`scripts/train_latent.py`) is implemented and tested; that run is next.

![RetNet-300M (RetNet 2.0) Architecture](assets/retnet_architecture.png)

## Highlights

- **Three equivalent ways to compute retention**: parallel, chunkwise and recurrent. Tests check they agree to ~1e-9 in float64. Training uses chunkwise mode (memory linear in sequence length). Inference uses recurrent mode: constant memory per token, no KV cache.
- **Delta-rule memory inside chunkwise training**: the within-chunk delta-rule dependencies reduce to one batched unit-lower-triangular solve (`torch.linalg.solve_triangular`). The recurrence is never unrolled.
- **Calculator in the loop**: training data rewrites correct arithmetic as `<<16*47=752>>752`. At inference an AST-based evaluator (no `eval`) fills in the result. Feeding those tokens only extends the recurrent state; no cache is rebuilt.
- **Fault-tolerant training on preemptible GPUs**: versioned checkpoints, fallback past corrupted files, data prep that resumes partway through a source, and an auto-sized micro-batch for whichever GPU the session gets.
- **Ready for post-training**: 16 reserved token ids so latent-thought (`<bot>`/`<eot>`) or pause tokens can be added later without re-tokenizing 1B tokens.

## Results (base model, step 15,082)

Zero-shot, greedy decoding, first 256 GSM8K test problems:

| GSM8K | Accuracy |
|---|---|
| With calculator in the loop | **7.4% ± 1.6%** (3.0 calls per problem) |
| Without calculator | 4.7% ± 1.3% |

The calculator gives about 1.6× more correct answers. In the samples, the model usually sets the problem up right but gets the arithmetic wrong (`168 - 150 = 42`); the calculator fills in the exact result. For scale: GPT-2 Medium (355M, ~30B tokens) and Pythia-410M (300B tokens) score about 1–2% on GSM8K.

## Resource constraints and how the design handles them

| Constraint | Effect | How the project handles it |
|---|---|---|
| **Colab T4**: 16 GB VRAM, Turing (no fast bf16), ~12 GB RAM | A 310M model with AdamW and 1024-token sequences barely fits | fp16 + `GradScaler` on T4 (bf16 on L4/A100); retention core kept in fp32 for stability; the largest micro-batch that fits is measured at startup and activation checkpointing is enabled only if needed. Effective batch stays fixed at 32 × 1024 tokens on any GPU |
| **Preemptible sessions** | Training is cut off every few hours | **Run All** resumes from the newest valid checkpoint, on whatever GPU the next session gets. Checkpoints are saved every 250 steps / 15 min and on stop signals |
| **15 GB of Drive storage** | A 310M checkpoint with optimizer state is ~3.6 GB; deleted files still count in Trash | Each checkpoint is a new step-numbered file. AdamW moments are stored in bf16 (checkpoint ~2.5 GB). Low free space is reported before each save |
| **Disconnects mid-write** | Truncated checkpoints and half-built datasets | Staged on local SSD and fsynced; the loader skips corrupt checkpoints; data prep saves progress every 30 s and resumes from the same row |
| **Compute budget** | Default run is 0.5B tokens, well below the ~6B Chinchilla-optimal for 310M params | Data restricted to dense, curated sources (textbooks, code, worked math) to get more from each token; a 125M preset gives cheaper comparisons |
| **Gated or unfinished datasets** | Some `nampdn-ai/tiny-*` sets are gated; `tiny-math-textbooks` currently serves `[Not ready yet!]` placeholder rows | Each source is probed first and swapped for an ungated fallback (e.g. FineMath-4+) when needed; shards averaging < 32 tokens/doc are rejected |

## Pipeline

```mermaid
flowchart LR
    subgraph Data["Data prep (resumable)"]
        HF[(Hugging Face<br/>streaming)] --> Probe{stub or gated?}
        Probe -- no --> Shard[per-source shards<br/>on Drive]
        Probe -- yes --> FB[ungated fallback] --> Shard
        Shard --> Bin[train.bin / val.bin<br/>1B uint16 tokens]
    end
    subgraph Train["Training (resumable, any GPU)"]
        Bin --> Loop[chunkwise RetNet<br/>fp16/bf16 + auto batch]
        Loop -->|every 250 steps / 15 min| Ckpt[(ckpt_step_*.pt<br/>shared Drive)]
        Ckpt -->|Run All after preemption| Loop
    end
    subgraph Use["Evaluation and inference"]
        Ckpt --> Eval[GSM8K, ARC, HellaSwag,<br/>MBPP, per-source val loss]
        Ckpt --> Chat[recurrent decoding<br/>+ calculator]
    end
```

## Repository layout

| Path | What it holds |
|---|---|
| `retnet/retention.py` | Multi-scale retention: parallel / chunkwise / recurrent, delta rule, short conv |
| `retnet/model.py` | Decoder stack, size presets (300m, 300m-deep, 125m, 30m), generation |
| `retnet/calculator.py` | Safe AST evaluator, calculator-in-the-loop decoding, arithmetic annotation |
| `retnet/checkpoint.py` | Fault-tolerant checkpoint save/load/rotation |
| `scripts/prepare_mix.py` | Resumable data prep and tokenizer training |
| `scripts/train.py`, `scripts/eval.py`, `scripts/chat.py` | Training, evaluation suite, interactive chat |
| `colab/retnet_300m_shared.ipynb` | One-click Colab training notebook (Run All resumes) |
| `colab/retnet_300m_inference.ipynb` | Colab notebook: generate from the latest checkpoint |
| `retnet/latent.py`, `scripts/train_latent.py`, `scripts/eval_latent.py` | Latent reasoning: thought feedback, curriculum training, thought-count sweep |
| `colab/retnet_300m_latent.ipynb` | Colab notebook: latent-reasoning curriculum |
| `tests/` | Mode equivalence, delta memory, calculator, checkpoint recovery, data-prep resume |

---

## 1. Why RetNet 2.0 Features (Delta Memory + Short Conv + Calculator) Are Essential

Plain RetNet (Sun et al., 2023) updates its per-head state via a purely additive recurrence:
$$S_t = \gamma S_{t-1} + k_t^T v_t, \qquad \gamma \in (0, 1)$$
While fast ($O(1)$ recurrent inference, parallel/chunkwise training), plain additive retention has two fundamental weaknesses on coding and reasoning tasks:
1. **Inability to overwrite variables (Associative Recall)**: When code or a reasoning trace updates a variable (`x = 3 ... x = 7`), additive state blends old and new values. **Delta-Rule Memory (`--memory delta`)** uses:
   $$S_t = \gamma S_{t-1} + \beta_t k_t^T (v_t - \gamma k_t S_{t-1})$$
   which erases the old value associated with key $k_t$ before writing $v_t$. During training, chunkwise mode solves the within-chunk dependencies in parallel via a fast batched unit-lower-triangular solve (`torch.linalg.solve_triangular`).
2. **State tracking & negative eigenvalues (`--beta_max 2.0`)**: With write strength $\beta_t \in (0, 2)$, the state transition matrix $(I - \beta_t k_t k_t^T)$ admits eigenvalues in $(-1, 1)$, unlocking parity, counters, boolean toggling, and nested control-flow tracking (Grazzi et al., 2024).
3. **Short Causal Depthwise Convolution (`--short_conv 4`)**: Mixes a 4-token local window before $q, k, v$ projections so keys can bind a variable name token with `=` and its value token.
4. **AST Calculator-in-the-Loop + Digit-Splitting BPE (`retnet/calculator.py`)**:
   - Every digit (`0`–`9`) is tokenized individually and `<<`, `>>`, `=` are isolated tokens.
   - Worked arithmetic in training corpora is rewritten into `<<expr=result>>result` (`annotate_arithmetic`).
   - At inference (`generate_with_calculator` / `generate_batch_greedy`), whenever the model emits `<<expr=`, the safe AST parser computes the exact value and injects `result>>` directly into the recurrent state in $O(1)$ without rebuilding any KV cache.

---

## 2. High-Quality Data Mix (`nampdn-ai/tiny-series` + Reasoning Only)

`scripts/prepare_mix.py` streams strictly textbook-grade, coding, and reasoning datasets (no noisy raw web scrapes):

| Source Key | Hugging Face Dataset | Description | Default Share |
|---|---|---|---|
| `tiny_codes` | `nampdn-ai/tiny-codes` | ~1.6M well-commented code & control-flow reasoning snippets | **24%** |
| `tiny_math_textbooks` | `nampdn-ai/tiny-math-textbooks` | Synthetic math textbooks (calculator-annotated). Currently serves placeholder rows, so `HuggingFaceTB/finemath` (4+) is used automatically | **20%** |
| `tiny_textbooks` | `nampdn-ai/tiny-textbooks` | ~420k foundational synthetic textbooks (`textbook` column extracted) | **14%** |
| `tiny_code_textbooks` | `nampdn-ai/tiny-code-textbooks` | ~207k synthetic programming textbooks explaining code step-by-step | **12%** |
| `metamath` | `meta-math/MetaMathQA` | Step-by-step math Q&A with arithmetic rewritten as `<<a*b=c>>c` | **12%** |
| `tiny_orca_textbooks` | `nampdn-ai/tiny-orca-textbooks` | ~147k step-by-step analytical reasoning textbooks | **10%** |
| `magicoder` | `ise-uiuc/Magicoder-OSS-Instruct-75K` | Algorithmic & software engineering problem-solution pairs | **4%** |
| `tiny_lessons` | `nampdn-ai/tiny-lessons` | Bite-sized structured Markdown lessons & examples | **2%** |
| `gsm8k` | `openai/gsm8k` (`train` split only) | Grade-school math word problems with native `<<a*b=c>>c` calls | **2%** |

*(Optional: `tiny_strange_textbooks` (`nampdn-ai/tiny-strange-textbooks`) can also be enabled via `--shares`.)*

---

## 3. Fault-Tolerant Training on Preemptible Sessions

### Why naive single-file checkpointing (`ckpt.pt`) breaks
1. **Mid-write disconnects**: Writing a ~2.4 GB checkpoint to Google Drive over FUSE takes 20–40 seconds; if the session is preempted mid-write, a single `ckpt.pt` is left corrupted and the run is lost.
2. **Cross-GPU (`T4` fp16 vs `L4`/`A100` bf16) `GradScaler` crash**: Resuming a `bf16` checkpoint (where `GradScaler` was disabled and saved `{}`) on a `T4` (where `GradScaler` is enabled for `fp16`) crashes PyTorch if `scaler.load_state_dict({})` is called unconditionally.
3. **Pruning that outruns the upload**: Drive's `fsync` returns before the upload finishes, and with the quota full a multi-GB upload fails silently while small writes still sync. Deleting (or worse, truncating) the previous checkpoint at that moment leaves no usable checkpoint. This happened in the real run: every checkpoint after step 15,082 was lost, and step 15,082 was only recovered from Drive's 30-day version history.

### How `retnet-300m` handles them
1. **Versioned Step Checkpoints (`ckpt_step_XXXXXXX.pt`) + Multi-Candidate Fallback Loader (`retnet/checkpoint.py`)**:
   - Every checkpoint is staged to local SSD (`/tmp`) first, then written as a new step-numbered file `ckpt_step_0000250.pt` and flushed via `os.fsync`. An existing checkpoint is never overwritten.
   - On resume (`load_latest_checkpoint`), all `ckpt_step_*.pt` files are scanned from highest step to lowest; if the newest file was truncated by a sudden disconnect, training automatically falls back to the previous valid checkpoint.
2. **Anchored Rotation**:
   - Besides the newest `--keep_ckpts`, pruning keeps the newest full-size checkpoint that is at least 20 minutes old (time to finish uploading), deletes nothing if there is no such anchor, and only unlinks files (never truncates), so a pruned file is still recoverable from Trash.
3. **Compact `bfloat16` Optimizer State (`compact_optimizer_state`)**:
   - AdamW moment buffers (`exp_avg`, `exp_avg_sq`) are saved in `bfloat16` on disk and restored to `fp32` on load, shrinking each 300M checkpoint from ~3.6 GB to ~2.4 GB.
4. **Cross-GPU (`T4` $\leftrightarrow$ `L4`/`A100`) & Auto-Batch Compatibility**:
   - `pick_batch_size` measures the largest micro-batch that fits the current GPU (automatically enabling per-block activation checkpointing `--grad_ckpt` on 16 GB T4 GPUs if needed) while keeping `seqs_per_step = 32` constant.
   - `GradScaler` state is only loaded when switching into `fp16` if the checkpoint actually contains a non-empty `scale` state.

---

### Colab Setup

1. **Build the Colab bundle locally**:
   ```bash
   bash scripts/make_colab_bundle.sh   # -> dist/retnet-300m-colab.zip
   ```
2. **Upload it**: in Google Drive, create **`My Drive/retnet-300m/`** and upload **`dist/retnet-300m-colab.zip`** into it.
3. **Run**: open `colab/retnet_300m_shared.ipynb` in Colab, select a GPU runtime (`T4`, `L4`, or `A100`), and click **Runtime $\rightarrow$ Run all**. Data prep also runs on a CPU runtime.
   - After a disconnect, click **Run all** again: it verifies the cached dataset on Drive, finds the latest valid `ckpt_step_XXXXXXX.pt`, adapts the micro-batch to the GPU, and continues training.
4. **Inference**: `colab/retnet_300m_inference.ipynb` loads the latest checkpoint and generates with the calculator in the loop.
5. **After any code change**: rerun `bash scripts/make_colab_bundle.sh` and upload the new zip, replacing the old one on Drive. Also re-open the notebook from `colab/`. The notebook prints `Code bundle: version …, git …` and stops if the zip on Drive is older than it expects.

### Data preparation resumes mid-source
`scripts/prepare_mix.py` tokenizes each source into its own shard in the Drive shard cache (`data/tinymix_shards/`). Every 30 s it fsyncs the partial shard and writes `<source>.progress.json`. After a disconnect, the next Run All continues from that point. It does not restart the source. Sources that return placeholder rows are swapped for their ungated fallback. At the time of writing, `nampdn-ai/tiny-math-textbooks` returns `[Not ready yet!]` placeholder rows. Cached shards that average under 32 tokens per document are rebuilt.

### Reserved tokens for post-training
The tokenizer reserves ids 1–16 (`<|reserved_0|>` … `<|reserved_15|>`, right after `<|endoftext|>` = 0). They never appear in pretraining text. They are kept for post-training, for example latent-thought markers (`reserved_0` = `<bot>`, `reserved_1` = `<eot>`) or pause tokens, so those can be added without changing the tokenizer or re-tokenizing the corpus. `meta.json` lists them under `special_tokens`. A cached tokenizer without them is set aside as `tokenizer.legacy.json` and retrained. Shards record the fingerprint of the tokenizer that built them and are rebuilt if it changes.

## 4. Latent Reasoning (Continuous Thought) on RetNet

`retnet/latent.py` and `scripts/train_latent.py` implement Coconut-style latent reasoning (Hao et al., 2024). After the question the model emits `<bot>`, then for `n` steps it feeds its own last hidden state back in as the next input embedding instead of writing tokens, then `<eot>` switches back to text:

```
Question: ...\nAnswer:<bot> [h1][h2]...[hn] <eot> remaining steps\n#### 42
```

- **Why RetNet fits:** each thought is one recurrent update of a fixed-size state, so thinking for 1,000 steps uses the same memory as 10. A Transformer's KV cache grows with every thought.
- **Curriculum:** stage 0 is step-by-step fine-tuning (the explicit chain-of-thought baseline). At stage `s` the first `s` reasoning steps are replaced by `s × c` thoughts, and the optimizer is reset. Gradients flow back through every thought.
- **Thought input:** `scale · norm_f(h)`, with a learned scalar initialized to the norm of real token embeddings.
- **Evaluation:** `scripts/eval_latent.py` sweeps the number of thoughts at test time (accuracy vs. thinking length).
- **Tests:** with zero thoughts the latent path matches an ordinary forward pass exactly; batched thoughts match per-example ones; a two-stage run stopped mid-stage resumes and finishes.
- **Colab:** `colab/retnet_300m_latent.ipynb`.
