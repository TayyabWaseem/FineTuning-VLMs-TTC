# ATML PA3 — Task 1

Task 1 fine-tunes `Qwen/Qwen3-0.6B` on the first 40,000 eligible MetaMathQA
examples with Full FT, Full FT + L2-SP, LoRA-4, LoRA-16, and OPLoRA-16. The
assignment manual is `ATML-PA3.pdf`; engineering decisions and execution
evidence are in `JOURNAL.md`. The PDF report must be written by the student.

## Kaggle execution

1. Publish this code and both notebooks to the public GitHub repository. Confirm
   that Kaggle's cloned commit SHA is the intended revision.
2. Enable a Kaggle GPU and internet access. Run `task1_smoke_test.ipynb` first.
   Its final output must say `"status": "PASS"`; all three adapter corrections
   must be nonzero and no optimizer step may be skipped.
3. Create Kaggle secrets `HF_TOKEN` (Hugging Face write token) and `PA3_HF_REPO`
   (`owner/repo-name`). The full notebook creates or uses a **private** Hugging
   Face dataset repository for recovery states and progress. No token belongs
   in the notebook, Git remote, or repository files.
4. Run `task1_kaggle.ipynb`. Its five training cells can run in separate Kaggle
   sessions. Rerun setup and the interrupted method cell to restore the latest
   completed remote checkpoint. Later evaluation and diagnostic stages also
   restore their saved outputs.

The private repository stores full optimizer recovery states at 9,984, 20,000,
29,984, and 40,000 examples per method. A checkpoint is durable only after its
upload and remote file verification finish. The full notebook fails if code,
configuration, prompts, or key package versions differ across sessions. The
smoke notebook uses small temporary runs and does not provide report scores.

Results are written under `results/task1/`. They are ignored by Git. The final
analysis command creates benchmark, resource, and diagnostic CSV files and an
SFT loss plot from saved machine-readable outputs. Inspect those outputs before
writing the report.

## Code and external sources

- Task definitions, answer filters, and benchmark execution use
  [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
  The HellaSwag prompt preprocessing in `diagnostics.py` follows its
  [HellaSwag utility](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hellaswag/utils.py),
  and the GSM8K diagnostic prompt follows its
  [GSM8K CoT task](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot.yaml).
- Non-thinking chat formatting follows the
  [Qwen3 model guidance](https://github.com/QwenLM/Qwen3).
- The optional private recovery transport uses
  [huggingface_hub](https://huggingface.co/docs/huggingface_hub/guides/upload).

The student is responsible for reviewing and understanding all submitted code.

## Task 2 — vision-language experiment

`task2_kaggle.ipynb` embeds the local Task 2 Python sources and can be uploaded
directly to Kaggle without pushing this repository. Enable Internet and a GPU,
and provide the `HF_TOKEN` and `PA3_HF_REPO` Kaggle secrets for a private
Hugging Face dataset. Its outputs and checkpoint markers are isolated under
`results/task2/`. Run the GPU smoke before the full experiment. Data selection,
verified SigLIP feature shards, and training checkpoints are recoverable across
sessions; training resumes from the last verified 250-update checkpoint.

The notebook prints observed seconds per update; multiply by remaining updates
to estimate training time and add data preparation, development evaluations,
final generation and uploads. Total T4 runtime has not been measured. The
optional 49-token comparison is not included in the main notebook.

Task 2 obtains COCO 2014 captions, instances and images from the
[COCO dataset](https://cocodataset.org/) and VQAv2 questions, annotations and
complementary pairs from the [official VQA downloads](https://visualqa.org/download.html).
SigLIP features use the normalized `last_hidden_state` from
[Hugging Face's SigLIP implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py).
The local answer-normalization rule is applied consistently and documented in
`task2_vlm/normalize.py`; its contraction table is not the complete official
VQA normalization table. The VQA leave-one-out consensus formula follows the
assignment and the [VQA evaluation description](https://visualqa.org/evaluation.html).
The split treats unannotated COCO objects as absent, so annotation omissions
may affect presence labels. The notebook saves evidence; report prose and
interpretation are the student's responsibility.
