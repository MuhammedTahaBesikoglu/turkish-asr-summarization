# Fine-tuning: full fine-tune and LoRA

Modules: `src/training/train_whisper.py`, `src/training/data.py`, `src/training/callbacks.py`,
`src/training/tracking.py` · Configs: `configs/train_full.yaml`, `configs/train_lora.yaml`,
`configs/eval_full.yaml`, `configs/eval_lora.yaml` · Colab: `notebooks/colab_train.ipynb` ·
Tests: `tests/test_training.py`, `tests/test_config.py`

`openai/whisper-small` is fine-tuned on the Common Voice 27.0 `tr` **train** split, either
fully or with LoRA adapters. The two configs differ only in `method`, `lora` and the learning
rate, so their WER, training time and memory can be compared directly. Test-set numbers come
from the same `evaluate_asr.py` used for the zero-shot baseline.

## Results

| Method | Test WER (normalized) | Test CER (normalized) | Test WER (raw) | Training time | Peak GPU memory | Trainable params |
|---|---|---|---|---|---|---|
| Full FT | TBD | TBD | TBD | TBD | TBD | TBD |
| LoRA | TBD | TBD | TBD | TBD | TBD | TBD |

Filled in from `results/finetune/<run>/metrics.json` (test) and `train_summary.json`
(training) after the Colab runs.

## Data and model selection

- Training: the full official `train` split (41,465 clips). The extra clips in
  `validated.tsv` are **not** used (see [data.md](data.md)): they add hours but almost no
  speakers (93.8% from 5 speakers), and the official splits keep results comparable.
- Model selection: every `eval_steps` the model transcribes a fixed random subset of `dev`
  (`data.eval_samples` = 2,000 clips, `data.seed` = 42) with greedy decoding; the checkpoint
  with the lowest normalized WER is kept (`load_best_model_at_end`). A subset keeps each
  evaluation at a few minutes on a T4.
- The `test` split is never loaded by the training script. `validate_config` rejects `test`
  as `train_split` or `eval_split`.
- Labels are the raw sentences (casing and punctuation kept), tokenized with the
  `<|startoftranscript|><|tr|><|transcribe|><|notimestamps|>` prefix. The collator strips the
  leading start token (the model prepends it when shifting labels) and pads with `-100`.
- Audio is decoded and turned into log-mel features on the fly in DataLoader workers; no
  preprocessed copy is written to disk.
- Dev metrics are computed with the same `compute_metrics` (and Turkish normalizer) as the
  test evaluation, against the original sentences.

## Configuration

Both files have the same structure; `training` is passed to
`transformers.Seq2SeqTrainingArguments` as-is.

| Key | Full | LoRA | Notes |
|---|---|---|---|
| `method` | `full` | `lora` | |
| `lora.r` / `lora_alpha` / `lora_dropout` | – | 32 / 64 / 0.05 | Passed to `peft.LoraConfig` |
| `lora.target_modules` | – | `q_proj`, `v_proj` | Attention projections in encoder and decoder |
| `training.learning_rate` | 1e-5 | 1e-3 | HF recipe / PEFT Whisper example |
| `training.per_device_train_batch_size` | 16 | 16 | `gradient_accumulation_steps` 1 |
| `training.max_steps` | 4000 | 4000 | ≈1.5 epochs; `warmup_steps` 500, linear decay |
| `training.fp16` | `true` | `true` | Mixed precision; ignored on CPU |
| `training.activation_checkpointing` | `true` | `true` | Gradient checkpointing (`use_reentrant: false`) |
| `training.eval_steps` = `save_steps` | 500 | 500 | Dev evaluation and checkpoint |
| `training.save_total_limit` | 1 | 1 | Latest and best checkpoint are kept |
| `training.metric_for_best_model` | `wer` | `wer` | Normalized dev WER, lower is better |
| `data.eval_samples` | 2000 | 2000 | Dev subset for evaluation during training |
| `data.num_workers` | 2 | 2 | Colab's free tier has 2 vCPUs |
| `tracking.db` | `mlflow/mlflow.db` | same | One MLflow experiment, `whisper-small-tr` |
| `output.results_dir` | `results/finetune/whisper-small-full` | `.../whisper-small-lora` | `train_summary.json` |

These are starting values from the references, not tuned. If the measured time per step on
the T4 makes 4000 steps impractical on the free tier, `max_steps` is reduced for **both**
runs.

Any key can be overridden on the command line with `--set key.subkey=value` (YAML-parsed),
e.g. `--set training.max_steps=10 --set tracking=null`.

## Running

Colab (free T4), as used for the reported runs, in `notebooks/colab_train.ipynb`:

1. Locally: `python scripts/pack_data.py --splits train dev` writes
   `data/colab/cv-corpus-27.0-2026-09-11-train-dev.tar`. Upload it to Google Drive
   (`MyDrive/turkish-asr/`) next to the test archive from the baseline.
2. In the notebook, set `METHOD` (`"lora"` or `"full"`) and run all cells. It:
   - links `outputs/` to `MyDrive/turkish-asr/outputs` so checkpoints survive disconnects,
   - runs a 20-step smoke run and prints the time per step and the projected run time,
   - trains with `--resume` and `--tracking-backup MyDrive/turkish-asr/mlflow`,
   - evaluates `outputs/whisper-small-<method>/final` on the full test split,
   - copies `results/finetune/whisper-small-<method>/` to Drive.
3. After a disconnect, run all cells again: training resumes from the last checkpoint.
4. Copy `results/finetune/whisper-small-<method>/` from Drive into the repository and commit.

Locally (4 GB GPU), a functional check only:

```bash
# LoRA on whisper-small fits in 4 GB at batch size 2
python -m src.training.train_whisper --config configs/train_lora.yaml \
    --set training.output_dir=/tmp/smoke/out --set output.results_dir=/tmp/smoke/results \
    --set tracking.experiment=smoke --set training.max_steps=10 \
    --set training.eval_steps=5 --set training.save_steps=5 \
    --set training.per_device_train_batch_size=2 --set training.per_device_eval_batch_size=4 \
    --set data.max_train_samples=40 --set data.eval_samples=8
```

A full fine-tune of `whisper-small` does not fit in 4 GB (fp32 weights, gradients and Adam
states alone take about 3.6 GiB); test that code path with
`--set model.name=openai/whisper-tiny`.

Test evaluation of a trained model:

```bash
python -m src.evaluation.evaluate_asr --config configs/eval_lora.yaml
```

## Outputs

`training.output_dir` (`outputs/whisper-small-<method>/`, git-ignored, on Drive in Colab):

- `checkpoint-<step>/`: Trainer checkpoints for `--resume`, each with `resource_usage.json`.
- `final/`: the best checkpoint as a plain `WhisperForConditionalGeneration` (LoRA merged
  into the base weights with `merge_and_unload`) plus the processor; generation is preset to
  Turkish transcription. This is what `evaluate_asr.py` and the demo load.
- `adapter/` (LoRA only): the adapter weights alone.
- `mlflow_run_id.txt`: lets a resumed run continue the same MLflow run.
- `train_summary.json`: copy of the summary below. Its presence marks the run as finished:
  the script then exits without training, because after training the Trainer deletes the
  last checkpoint when it is not the best one, and resuming would train again.

`output.results_dir` (`results/finetune/whisper-small-<method>/`, committed):

- `train_summary.json`: parameter counts, steps and epochs, best checkpoint, dev metrics of
  the best model, resources, the full log history (loss and dev WER per step), library
  versions, git commit and config.
- `metrics.json`, `predictions.csv`: written by `evaluate_asr.py` on the test split (same
  format as the baseline, see [baseline.md](baseline.md)).

## Resource accounting

`ResourceTracker` (`src/training/callbacks.py`) records, summed over all Colab sessions of
a run:

- `train_seconds`: wall time minus evaluation time (`eval_runtime`), so methods with a
  different generation speed are compared on training alone. Checkpoint saving is included.
- `wall_seconds`, `eval_seconds`, `sessions` (number of resumes + 1).
- `peak_gpu_memory_gib`: `torch.cuda.max_memory_allocated`, maximum over sessions,
  including evaluation.

The totals are stored in each checkpoint, so a resumed run continues from the values of
the checkpoint it resumes from; steps lost after the last checkpoint are redone and counted
once.

## Experiment tracking (MLflow)

- Backend: SQLite (`mlflow/mlflow.db`, git-ignored). MLflow 3.x refuses the old `./mlruns`
  file store unless explicitly allowed, and SQLite on the Google Drive mount is prone to
  locking errors, so the database stays on local disk.
- `--tracking-backup DIR` copies the database to `DIR` after every checkpoint (SQLite online
  backup) and at the end. A fresh Colab VM restores it from there before training.
- The `transformers` MLflow callback logs the training arguments and model config as params,
  and loss, learning rate, dev WER/CER per step as metrics. The script adds
  `summary/train_seconds`, `summary/peak_gpu_memory_gib`, `summary/trainable_params` etc.
- Viewing locally: download `MyDrive/turkish-asr/mlflow/mlflow.db` to `mlflow/` and run
  `mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db`.

## Colab free tier

- Sessions can end after a few hours, or earlier when the tab is idle; GPU access may be
  unavailable for a while after heavy use. Checkpoints every 500 steps on Drive and
  `--resume` make a run span several sessions.
- Google Drive space (estimates from the 241.7M parameters of `whisper-small`): a full-FT
  checkpoint holds fp32 weights (~0.97 GB) and Adam states (~1.9 GB), about 2.9 GB, and up
  to two are kept, plus `final/` (~0.97 GB). LoRA checkpoints are small; its `final/` is
  ~0.97 GB. With the data archives, keep roughly 10 GB free.
- Suggested order: LoRA first (smaller checkpoints), then full fine-tune.
- Colab preinstalls `torchao` 0.10; peft 0.21 raises *"Found an incompatible version of
  torchao ... only versions above 0.16.0 are supported"* when it creates LoRA layers. The
  project does not use torchao, so the notebook uninstalls it.

## Known limitations

- Hyperparameters are taken from the references, not searched.
- Dev WER during training is measured on 2,000 of 11,864 dev clips; test WER is always on
  the full test split.
- transformers prints *"The attention mask is not set..."* during evaluation; as in the
  baseline it is harmless (all inputs are padded to 30 s, same decoder prompt).
- Peak memory includes evaluation (generation), not only the training steps.
