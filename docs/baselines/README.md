# Frozen cloud baseline snapshot — 2026-11-01

Entry points: `index.html` (English), `zh.html` (Chinese). Original cloud plots
and source attachments remain in their original language and bytes.
Pages publishes the repository `docs/` directory, so
the site URL is `baselines/` below the existing project URL. All site-local
links are relative and work under a GitHub Pages repository prefix.

This bundle includes exactly 177 / 256 reviewed games, not a live
directory count. The explicit collection is in `snapshot_source.json` and must
match `root_review.json` exactly. Every run uses one training seed, four evaluation seeds and
262,144 joint environment steps. Budget completion does not establish learning,
significance, task solution or generalization. All non-positive comparisons are preserved on their own game cards.
GI215's zero served orders are preserved when that game is in the review. No LLM/UED method is validated here.

## Rebuild from the public repository bundle

From the repository root, run `python tools/build_baselines_snapshot.py`.
Python standard library only; no network, JAX, training or evaluation required.
Existing public input hashes are checked before rendering. The existing Pages
builder leaves this independent directory intact; no workflow edit is needed.
Run `python tools/build_baselines_snapshot.py --check` for a read-only integrity,
coverage, privacy and local-link check, also called by the release checker.

## Import an explicitly reviewed cloud bundle

Run `python tools/build_baselines_snapshot.py --review REVIEW.json --runs RUNS`.
Both arguments are local paths supplied by the operator; they are never written
to public output. The published run list must equal the explicit independently reviewed set.
Missing, extra or duplicate entries and unmatched hashes are rejected. Source
folders are never scanned to discover runs; later unreviewed returns are ignored.
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
