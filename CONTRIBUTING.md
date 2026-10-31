# Contributing

Preserve an existing game's mechanics, population, map, horizon and information
contract. Submit a versioned repair with the failing execution receipt and a
source diff. Keep the original source hash and historical failure evidence.
Changes to research claims, selection rules or frozen experiments require their
own research authorization and are outside this software repository's CI.

Each game needs a source module, manifest, mechanism description, and a scene
descriptor when spatial. All required assets must ship in the package. Avoid
absolute paths, global caches, credentials and parent-workspace dependencies.
Do not count alternate versions as new games.

Run `python tools/verify_all.py --games giNNN --retry` and
`python tools/verify_browser.py --games giNNN --retry`. Preserve failing receipts
in your PR when relevant. Engineering inputs are never human benchmark scores.
Measure floating differences and retain uncertainty rather than selecting an
arbitrary tolerance to declare numerical parity.

Build the wheel with `python -m build`, install it in a fresh virtual environment,
and run the documented quickstart. Rebuild the static site with
`python tools/build_docs.py`; check links with `python tools/check_release.py`.
