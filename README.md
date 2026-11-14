# retnet-300m: Coding & Reasoning RetNet 2.0 on Tiny-Series Textbooks

A **~310M-parameter RetNet 2.0** language model (`d_model=1024, n_layers=22, n_heads=16`) equipped with **Delta-Rule Memory ($\beta_{\max}=2$)**, **Short Causal Convolution ($K=4$)**, and an **AST Calculator-in-the-Loop**, trained exclusively on **high-quality synthetic textbooks, code explanations, and step-by-step reasoning** (`nampdn-ai/tiny-series` + `MetaMathQA` + `GSM8K` + `Magicoder`), with **multi-account Shared Google Drive checkpointing** for seamless Google Colab execution across accounts.

![RetNet-300M (RetNet 2.0) Architecture](assets/retnet_architecture.png)

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
| `tiny_math_textbooks` | `nampdn-ai/tiny-math-textbooks` | ~635k synthetic math, logic, probability & algebra textbooks (calculator-annotated) | **20%** |
| `tiny_textbooks` | `nampdn-ai/tiny-textbooks` | ~420k foundational synthetic textbooks (`textbook` column extracted) | **14%** |
| `tiny_code_textbooks` | `nampdn-ai/tiny-code-textbooks` | ~207k synthetic programming textbooks explaining code step-by-step | **12%** |
| `metamath` | `meta-math/MetaMathQA` | Step-by-step math Q&A with arithmetic rewritten as `<<a*b=c>>c` | **12%** |
| `tiny_orca_textbooks` | `nampdn-ai/tiny-orca-textbooks` | ~147k step-by-step analytical reasoning textbooks | **10%** |
| `magicoder` | `ise-uiuc/Magicoder-OSS-Instruct-75K` | Algorithmic & software engineering problem-solution pairs | **4%** |
| `tiny_lessons` | `nampdn-ai/tiny-lessons` | Bite-sized structured Markdown lessons & examples | **2%** |
| `gsm8k` | `openai/gsm8k` (`train` split only) | Grade-school math word problems with native `<<a*b=c>>c` calls | **2%** |

*(Optional: `tiny_strange_textbooks` (`nampdn-ai/tiny-strange-textbooks`) can also be enabled via `--shares`.)*

---

## 3. Running Across Multiple Google Colab Accounts via a Shared Drive Folder

### Why naive single-file checkpointing (`ckpt.pt`) breaks across Google accounts
When **Account A** creates `ckpt.pt` in a shared Google Drive folder, Account A is the **file owner**. When **Account B** mounts the same shared folder as an **Editor**:
1. **Cross-account `os.replace` / `unlink` permission error**: Google Drive FUSE often forbids a non-owner Editor from unlinking or `os.replace`-ing a file owned by Account A (`PermissionError: [Errno 13] Permission denied`).
2. **Mid-sync Colab disconnects**: Writing a ~2.4 GB checkpoint over FUSE takes 20–40 seconds; if Account A disconnects mid-write, a single `ckpt.pt` is left corrupted.
3. **Cross-GPU (`T4` fp16 vs `L4`/`A100` bf16) `GradScaler` crash**: Resuming a `bf16` checkpoint (where `GradScaler` was disabled and saved `{}`) on a `T4` (where `GradScaler` is enabled for `fp16`) crashes PyTorch if `scaler.load_state_dict({})` is called unconditionally.
4. **Google Drive 15 GB Quota & Trash bloat**: Deleting multi-GB files without truncating them first fills up the owner's 15 GB Drive Trash.

### How `retnet-300m` solves all four
1. **Versioned Step Checkpoints (`ckpt_step_XXXXXXX.pt`) + Multi-Candidate Fallback Loader (`retnet/checkpoint.py`)**:
   - Every checkpoint is staged to local SSD (`/tmp`) first, then written as a new step-numbered file `ckpt_step_0000250.pt` (owned by the active account, so no cross-account overwrite permission is needed) and flushed via `os.fsync`.
   - On resume (`load_latest_checkpoint`), all `ckpt_step_*.pt` files are scanned from highest step to lowest; if the newest file was truncated by a sudden Colab disconnect, training automatically falls back to the previous valid checkpoint.
2. **Zero-Byte Truncation Before Rotation**:
   - Before pruning checkpoints older than `--keep_ckpts 2`, we first truncate the file to **0 bytes** (`f.truncate(0)` + `os.fsync`). In Google Drive shared folders, Editors *are* permitted to truncate file contents owned by another account even when `unlink` is restricted—and if `unlink` moves the file to Trash, it occupies **0 bytes** of Drive quota.
3. **Compact `bfloat16` Optimizer State (`compact_optimizer_state`)**:
   - AdamW moment buffers (`exp_avg`, `exp_avg_sq`) are saved in `bfloat16` on disk and restored to `fp32` on load, shrinking each 300M checkpoint from ~3.6 GB to ~2.4 GB.
4. **Cross-GPU (`T4` $\leftrightarrow$ `L4`/`A100`) & Auto-Batch Compatibility**:
   - `pick_batch_size` measures the largest micro-batch that fits the current account's GPU (automatically enabling per-block activation checkpointing `--grad_ckpt` on 16 GB T4 GPUs if needed) while keeping `seqs_per_step = 32` constant.
   - `GradScaler` state is only loaded when switching into `fp16` if the checkpoint actually contains a non-empty `scale` state.

---

### Step-by-Step Multi-Account Colab Setup

1. **Build the Colab bundle locally**:
   ```bash
   bash scripts/make_colab_bundle.sh   # -> dist/retnet-300m-colab.zip
   ```
2. **On Account 1 (Owner)**:
   - In Google Drive, create folder **`My Drive/retnet-300m/`**.
   - Upload **`dist/retnet-300m-colab.zip`** into `My Drive/retnet-300m/`.
   - Right-click **`retnet-300m`** $\rightarrow$ **Share** $\rightarrow$ add Account 2, Account 3, ... with **Editor** access.
3. **On Account 2, Account 3, ... (Collaborators)**:
   - Open Google Drive $\rightarrow$ **Shared with me**.
   - Right-click **`retnet-300m`** $\rightarrow$ **Organize** $\rightarrow$ **Add shortcut** $\rightarrow$ select **My Drive** $\rightarrow$ click **Add**.
4. **Run in Google Colab (from any account)**:
   - Open `colab/retnet_300m_shared.ipynb` in Colab, select a GPU runtime (`T4`, `L4`, or `A100`), and click **Runtime $\rightarrow$ Run all**.
   - When an account's GPU session ends, open the notebook on the next account and click **Runtime $\rightarrow$ Run all**—it verifies the cached dataset on Drive, finds the latest valid `ckpt_step_XXXXXXX.pt`, adapts the micro-batch to the new GPU, and continues training.
