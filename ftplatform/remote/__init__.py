"""
Runs a single customer's job somewhere with no access to this machine's
disk -- a Kaggle kernel today, potentially a rented GPU box later.

`packet.py` packages one customer + one job into a self-contained directory
and merges the result back; `kaggle_ops.py` drives the actual `kaggle` CLI
(dataset upload, kernel push, status poll, output download) around it. The
orchestration logic itself (candidates/runner.py, deploy.py, worker.py, ...)
is untouched -- this package only gets a job's inputs to wherever the GPU is
and its outputs back, same as a Colab cell did by hand.
"""
