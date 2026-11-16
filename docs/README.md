# Documentation

One Markdown file per component. Each file describes what the component does, how to run it,
its inputs/outputs, the config keys it reads and its known limitations. Metric values quoted
here must come from actual evaluation runs (see `results/`); unmeasured values are written as
**TBD**.

| Doc | Component | Roadmap stage | Status |
|-----|-----------|---------------|--------|
| [setup.md](setup.md) | Environment setup, `scripts/check_env.py` | 1 | done |
| [development.md](development.md) | Lint, format, tests, CI | 1 | done |
| [data.md](data.md) | Common Voice loading from local files, EDA | 2 | done |
| [normalization.md](normalization.md) | Turkish text normalization | 3 | done |
| baseline.md | Zero-shot `whisper-small` evaluation | 4 | planned |
| training.md | Full fine-tune and LoRA | 5 | planned |
| evaluation.md | WER/CER, comparison, error analysis | 6 | planned |
| summarization.md | Summarization model and chunking | 7 | planned |
| pipeline.md | Audio → transcript → summary | 8 | planned |
| demo.md | Gradio app and Hugging Face Spaces | 9 | planned |

A doc is added in the same commit as the component it describes.
