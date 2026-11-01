# Frozen cloud baseline snapshot — 2026-11-01

Entry points: `index.html` (English), `zh.html` (Chinese). Original cloud plots
and source attachments remain in their original language and bytes.
Pages publishes the repository `docs/` directory, so
the site URL is `baselines/` below the existing project URL. All site-local
links are relative and work under a GitHub Pages repository prefix.

This bundle includes exactly GI000–GI010 and GI215: 12 / 256 games, not a live
directory count. Every run uses one training seed, four evaluation seeds and
262,144 joint environment steps. Budget completion does not establish learning,
significance, task solution or generalization. GI009's negative comparison and
GI215's zero served orders are preserved. No LLM/UED method is validated here.

## Rebuild from the public repository bundle

From the repository root, run `python tools/build_baselines_snapshot.py`.
Python standard library only; no network, JAX, training or evaluation required.
Existing public input hashes are checked before rendering. The existing Pages
builder leaves this independent directory intact; no workflow edit is needed.

## Import an explicitly reviewed cloud bundle

Run `python tools/build_baselines_snapshot.py --review REVIEW.json --runs RUNS`.
Both arguments are local paths supplied by the operator; they are never written
to public output. The run list must equal the fixed 12-entry first-round set.
Every critical input file is checked against the independent review, including
the checkpoint bytes (never deserialized). Curves and the config, status,
provenance and evaluation JSONL are copied unchanged. Tracking is reduced to
public fields because its original logdir is private. No checkpoint is exported.

`snapshot_source.json` stores the frozen coverage and per-run public hashes.
`root_review.json` is the original independent review receipt, copied unchanged.
`hash_manifest.json` hashes every delivered file other than itself, plus the
builder. The manifest cannot contain its own hash. Exact floating-point values,
role vectors, paired episode differences and all evaluation points remain in
the original JSONL. Display numbers are rounded to six decimal places.

Original archive hashes in the review are provenance references, not a claim
that archives are published here. The four evaluation seeds are repeated
measurements of one learned policy, not four independent training seeds.
The config's PREPARED_NOT_RUN records preparation; execution status is in the
separate status receipt. Original plots are not regenerated or edited.
