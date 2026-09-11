# USP at SHROOM-Visions 2026

Code for *Why Informative Proxy Visual Signals Do Not Localize Hallucinated Spans*,
our entry to the SHROOM-Visions 2026 shared task.

The task is to mark the hallucinated character spans in a vision–language model's
response, in English, French, Italian and Chinese. The generating model is not
released, so any model-derived signal is a **proxy**. Our system is a multilingual
XLM-R token tagger; the paper asks what three proxy visual signals add to it, and
why the answer is close to nothing.

Three findings drive the paper:

- Under the official metric, span **position** accounts for almost all attainable
  Cor. Category labels contribute nothing to it, and graded probabilities add
  0.018. An oracle sentence-level detector reaches 0.476 where gold spans reach
  0.970.
- Caption entailment carries real sentence-level information (AUROC 0.605 against
  a 0.533 surface baseline) and still adds 0.002 Cor — the mismatch the
  decomposition predicts.
- Representation shift is fine-grained and informative in all four languages
  (macro AUROC 0.652) yet adds 0.009 Cor, and neither granularity nor category
  mixture explains the gap.

## Layout

```
notebooks/  the runs that produced the published numbers, with their output
src/        the pipeline as scripts, numbered in run order
analysis/   figure data and the R script that draws the paper figures
tests/      end-to-end smoke test
```

## Which file produced which result

The published system is **XLM-R large**. `src/01_span_tagger.py` is the earlier
**base** configuration and does not reproduce the paper's numbers on its own.

| Table 1 row | produced by | lr | batch | epochs | embeddings |
|---|---|---|---|---|---|
| XLM-R base | `src/01_span_tagger.py` | 2e-5 | 16 | 3 | trained |
| XLM-R large | `notebooks/03_large_tagger_kaggle.ipynb` | 1e-5 | 8 | 5 | trained |
| + repr shift | `src/05_fused_tagger.py` over notebook 03 | 1e-5 | 8 | 5 | frozen |
| + caption NLI | `src/05_fused_tagger.py` over notebook 03 | 1e-5 | 8 | 5 | frozen |
| + BIO output | `notebooks/04_bio_tagger_colab.ipynb` | 1e-5 | 8 | 5 | frozen |

Notebooks 03 and 04 carry their full output, including per-language development
scores. `src/05_fused_tagger.py` replaces the model, dataset and training cells of
notebook 03 and inherits its configuration; it has no hyperparameters of its own.

Note that the response segment is truncated at 256 subwords, so characters past
that point are never scored.

## Pipeline

| Script | Purpose |
|---|---|
| `src/01_span_tagger.py` | Text-only XLM-R tagger. The baseline, and the fixed testbed for everything below. |
| `src/02_category_strategies.py` | Three ways of mapping token category scores onto span labels. |
| `src/03_output_likelihood.py` | Output-likelihood features from Qwen2-VL-2B with and without the image. Near chance; excluded from the final system. |
| `src/04_representation_shift.py` | The 38 representation-shift features. The submitted signal. |
| `src/05_fused_tagger.py` | XLM-R with those features concatenated to its final layer. Final submission. |
| `src/06_captioners.py`, `06b`, `06c` | Caption each unique image once. |
| `src/07_caption_nli.py`, `08_nli_features.py` | Verify response sentences against the caption with NLI. |
| `src/09_adjudication.py` | Zero-shot LVLM adjudication. Scores below an empty submission. |

`src/alignment.py` holds the character/token conversion, the one place where
information can silently leak away, and `src/scoring.py` wraps the organizers'
scorer unmodified.

Steps 3 and 4 need a GPU. The rest runs on CPU.

## Data and weights

Neither is included here. The dataset and participant kit belong to the task
organizers; the images are not redistributed. Extracted features are ~275 MB and
derived from that data. Point `DISTRIB` in `src/01_span_tagger.py` at your own copy.

No trained checkpoints are published — training ran on Kaggle and the weights were
not retained. `src/05_fused_tagger.py` regenerates one from the extracted features.

## Figures

```bash
python analysis/export_fig_data.py
Rscript  analysis/fig_probe.R
```

## Notebooks and scripts

`notebooks/` holds the two runs behind the published large-model numbers, kept
with their output as a record. Both are Kaggle/Colab exports with their
`execution_count` intact, so what you see is what ran.

The scripts in `src/` are the same pipeline in plain-Python form, which is what
the smoke test exercises. `src/01_span_tagger.py` is the earlier base-model
configuration and is superseded by notebook 03; it is kept because the later
stages import from it.

## Tests

```bash
python tests/smoke_test.py
```

Exercises the tagger end to end on CPU with a randomly initialised encoder: tensor
shapes, collation, token gather, both losses, inference, span construction, scoring
and submission format. It verifies that the plumbing holds, not that the model
learns.

## Citation

```bibtex
@inproceedings{fernandes2026usp,
  title     = {USP at SHROOM-Visions: Why Informative Proxy Visual Signals
               Do Not Localize Hallucinated Spans},
  author    = {Fernandes, Rafael Mac\'ario},
  booktitle = {Proceedings of SHROOM-Visions 2026},
  year      = {2026}
}
```

## License

MIT.
