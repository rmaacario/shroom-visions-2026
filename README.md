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

Every number in Table 1 is an official leaderboard score. This table says which
file contains the code behind each, and whether an executed copy of that run
survives on disk.

| Table 1 row | leaderboard id | code | executed record |
|---|---|---|---|
| XLM-R base | `xlmr-base-textonly-v1` | `src/01_span_tagger.py`, `notebooks/draft_base_tagger_unrun.ipynb` | none |
| XLM-R large | `xlmr-large-textonly-v2` | `notebooks/run_span_tagger_kaggle.ipynb` | yes, 13/13 cells |
| + repr shift | `xlmr-large-visual-v3` | `notebooks/master_kaggle_all_stages.ipynb`, cell "FUSED TAGGER" | none |
| + caption NLI | `xlmr-large-visnli-v4` | `notebooks/master_kaggle_all_stages.ipynb`, cell "FINAL TAGGER" | none |
| + BIO output | `xlmr-large-bio-v5` | `notebooks/run_bio_tagger_colab.ipynb` | yes, all outputs |

`+ repr shift` is the submitted system. Its code is in the master notebook; the
notebook was saved with outputs cleared, so no executed copy of that run exists
here.

### The notebooks

| file | what it is |
|---|---|
| `master_kaggle_all_stages.ipynb` | The full Kaggle notebook: tagger, output-likelihood and representation-shift probes, captioning, NLI features, both fused taggers, and the adjudication probe. Complete source for rows 1-4. Saved with outputs cleared. |
| `run_span_tagger_kaggle.ipynb` | The XLM-R large text-only run, executed, with training loss and per-language development scores. |
| `run_probes_kaggle.ipynb` | Executed. Output-level likelihood features (all near chance) and the representation-shift probe, including the token-split vs response-split comparison. |
| `run_bio_tagger_colab.ipynb` | The BIO run on Colab, executed, with development scores. |
| `draft_*_unrun.ipynb` | Early base-model drafts. `execution_count` is empty on every cell, so they were saved without being run. Kept for provenance; they are not a source of any number. |

Configuration differs between the base and large rows: base was trained at
lr 2e-5, batch 16, three epochs; every large row at lr 1e-5, batch 8, five
epochs, with embeddings frozen in the fused and BIO rows.

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
