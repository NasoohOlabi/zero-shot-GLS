# Evaluation

For BPW, PPL, KLD, and JSDs, check the scripts below.

- `ppl.py`: computes mean/median/stddev/min/max plus bootstrap confidence intervals from a CSV PPL column.
- `kld.py`: computes smoothed token-distribution KLD between stegotext and cover/plain baseline text.
- `metrics.sh`: legacy end-to-end metric runner.

For language evaluations, check `judge/judge.py` for details.

For steganalysis, check `steganalysis/` for details.
