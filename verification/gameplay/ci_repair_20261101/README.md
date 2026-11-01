# Preserve actions when switching control mode

The Direct checkbox previously requested a snapshot and rebuilt the action
controls. Under a 500 ms response delay, an action selected before that
response could be replaced by Wait before it was queued. The reproduced
GI217 first action changed from 5 to 0. Switching the mode now changes only
the button labels, preserving the selection.

`regression_summary.json` records the controlled reproduction and the exact
22-step sequence after the repair. The maintained browser check also asserts
that an unqueued selection survives the mode change; serving the old app in
the browser made this new assertion fail.

All four original-art walkthroughs passed: GI214 (4 joint steps), GI215 (20),
GI216 (15), and GI217 (22). Their current runtime-bound receipts are the four
adjacent `gi*_live_ui.json` files. All 20 roles, 1x/2x zoom, and English UI were
checked. `protected_originals.json` verifies 30 unchanged game source,
manifest, scene and GIF files.

Detailed local browser recordings, full CI logs, and explicitly failed
diagnostic attempts are retained in `local-runs/ci-repair-20261101/`, excluded
from publication. The three diagnostic harness encoding failures are not
counted as successful app tests. The original remote CI run was
https://github.com/hl682/Crucibles_JAXMARL/actions/runs/41266677636 .

These are engineering checks, not trained-agent or human score benchmarks.
