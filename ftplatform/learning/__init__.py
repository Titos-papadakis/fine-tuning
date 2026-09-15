"""
Phase 8 -- cross-customer learning, deliberately narrow.

Only aggregate technique performance (workload, candidate_id, score) is
ever shared across customers -- never raw text, corpora, or model weights.
A candidate_id already *is* a technique descriptor by construction (see
candidates/generator.py: "stageA-qwen25-3b", "stageB-r16-a16",
"stageC-4bit-constrained" name the base model / LoRA config / quantization
directly), so recording it plus the same composite score the leaderboard
already computes is enough to answer "which technique tends to win for this
workload" without a customer's content ever entering this table.

Used for exactly one thing: reordering the default candidate lists in
candidates/generator.py so a new customer's sweep tries the
historically-strongest option first. Every option in the default set is
still tried for every customer -- their data may still favor something
different -- this only changes the order, which matters because each
candidate is a separate GPU-hour-consuming process. With no history yet (a
workload's first customer), ordering is left exactly as generator.py
already defines it -- purely additive, never a behavior change for the
first customer of any workload.
"""
