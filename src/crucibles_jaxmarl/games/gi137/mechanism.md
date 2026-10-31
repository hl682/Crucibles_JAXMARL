# Mossline Pursuit

A=4, O=288, U=11, H=64. Roles 0,1 form the tracker team; roles 2,3 form the runner team. The 9 by 9 map has a movement-blocking fence at x=4,y=2..6 with a gap at (4,4), brush at (2,2),(2,6),(6,2),(6,6), and public extraction cells (8,1),(8,7). Trackers start at (5,3),(5,5); runners at (1,1),(1,7). Reset is deterministic; the key is accepted for the standard interface but is not consumed.

## Native information and sensing

Tracker 0 has Manhattan visual radius 2 and scent radius 3. Tracker 1 has visual radius 3 and scent radius 1. Runners each have visual radius 3 and scent radius 2. Walls/fences affect movement only; sensing uses taxicab distance without line-of-sight occlusion. This is an explicit rule, not an accidental renderer shortcut.

Each team shares its active teammates' coordinates and charges. Local scent fields and opponent detections are not automatically pooled. An opponent in brush or with the current hide flag is concealed unless its distance is at most one or the observer is currently scanning. A tracker scan expands that observer's visual radius by three, to 5 for role 0 or 6 for role 1, and ignores brush/hide concealment inside that radius. Scanning does not expand the scent radius. Hide and scan flags apply to the returned observation of the current transition, clearing on the next ordinary action.

The complete terrain map and each runner's active/captured/extracted status are public. A resolved runner's exact final coordinate does not become public through the status announcement. Invisible position slots are zero and paired with explicit visibility masks; zeros are not fabricated positions. Opponent charges are zero-masked in native observations. Scent is a single anonymous intensity field: neither real trails nor decoys reveal an emitter ID. The scene binds the same native role masks and never substitutes the debug/full-state view.

## Motion, scent manipulation and resource cost

All active roles may request compass moves. Boundary or fence attempts fail. Within a team, old occupied cells and multiply proposed destinations block movement; teammates cannot swap or follow into a vacated cell during one tick. Opponents may occupy the same cell or cross without an automatic tag. Legal masks do not test hidden opponent presence, so a tag's availability or a movement attempt cannot disclose a hidden location.

Trackers begin with eight charge each and runners with six. There is no recharge. Action 5 costs one: a tracker scans, a runner hides while stationary. Action 6 costs one for either role and erases scent from its own cell and the four adjacent cells. Runner erasure also suppresses that runner's fresh trace. For trackers action 7 is a one-charge tag sweep. For runners actions 7..10 throw decoys north/east/south/west and cost two. All these actions are stationary; spending charge sacrifices immediate movement as well as a later sensing/manipulation/tag opportunity.

The fixed 81-cell scent field updates in this order: decrease every strength by one, bounded below by zero; erase the union of all requested radius-one areas; deposit strength 4 at each ordinary active runner's old cell; deposit strength 6 at each requested decoy target. Fresh traces use max(existing,deposit), so simultaneous deposits do not sum beyond six. Waiting ordinarily emits scent too. Hide and erasure suppress that role's true fresh trace, while a decoy still leaves the ordinary genuine trace at its thrower's old cell. A decoy target is two cells in the selected direction, clamped to [0,8] in each coordinate; if that cell is a fence the target becomes the runner's current cell. No trail diffusion, wind or additional hidden randomness is implemented.

An erase can remove an ally's earlier trace or deny the other tracker's useful evidence. A new runner trace can refill an area erased earlier in the same transition; erasure is not an immunity field. These ordering rules are fixed and observable through the resulting native scent state.

## Capture, extraction and payoffs

After movement and trace updates, any active tracker requesting a tag captures every still-active runner at Manhattan distance at most one. A hiding or brush-covered runner can be physically swept; hide changes information, not collision immunity. A moving tracker does not tag on that step. After capture, surviving runners standing on either extraction cell leave play. Capture therefore has priority if a tag and arrival at an exit coincide.

Every captured runner gives each tracker +1 and each runner -1; every extracted runner gives each runner +1 and each tracker -1. Events for both runners can occur in one step, so per-role reward lies in [-2,2]. Team members share both successes and losses, including after one runner has already resolved. Inactive runners cannot act except wait. All dones remain joint: natural termination occurs when both runner statuses are nonzero, including one capture and one extraction; otherwise H=64 truncates. There is no additional score threshold or claimed success criterion.

## Partner-dependent decisions and causal hypotheses

A long-range nose can follow persistent traces while the other tracker uses its broader ordinary vision to cover a route. Spending a scan can reveal a concealed opponent but consumes charge that could have funded a tag, and it leaves the tracker stationary. Runners can split exits, erase each other's route history, lay a high-intensity false branch, or hurry without concealment. A diversion may help the partner while increasing the decoy thrower's own exposure. Opposite teams' best responses depend on whether partners pursue, wait, scan, erase or commit to an exit.

The main fixed script requests two decoys followed by the runners' top/bottom route to extraction while trackers wait. It is a legal-action hypothesis, not a successful trajectory. A second script brings a runner adjacent to tracker 0, then requests hide against a tag to expose the intended information-versus-physical-capture distinction.

The scent paired hypothesis keeps tracker actions and keys fixed while role 2 throws an eastward plume or hides at the same location. The intended difference is tracker 0's local scent observation over subsequent ticks, not a claim about an inferred belief or learned pursuit. The capture-versus-scan pair uses the same hiding runner and same one-charge tracker expenditure; the intended difference is removal/reward versus visual information without automatic capture. Both remain unexecuted.

Role sensing radii, fence gaps, exit locations, charge budgets or trail lifetimes could change useful team strategies. Those are future design axes, not authored extra proposals or evidence of transfer.

## State binding and intended invariants

Agent entities are visible only through agent_visibility and active masks. Their hide/scan attributes are shown only when the corresponding entity is natively visible. Separate kit entities use team-only visibility so opponent reserves cannot leak through the scene. Scent entities are fixed actual map cells, active only for nonzero scent, with per-role sensing masks; the access observation also identifies sensed empty cells. Extraction flags bind the real exit cells. Only public resolution events are declared for generic event overlays: private decoy, hide and erasure actions are deliberately absent.

Intended invariants: charge never becomes negative; scent stays in integer 0..6; positions stay within map bounds and off fences for active agents; active teammates have distinct positions, while opponent overlap is legal; runner status changes only from active to captured/extracted and never reverses; resolved_count is monotone and at most two; anonymous scent never contains emitter identity; hidden opponents do not affect legal masks. These are authored rules awaiting runtime checks, not measured noninterference or parity results.

## Literal source evidence

Evidence refers to the frozen candidate.py with SHA-256 8cc580b11c98d57a0a62af45f3d974279e80f7832ad065e6c0abd03c2e7e1374. Line numbers were obtained by parsing/reading source text without importing it.
- candidate.py:32-59: reset implementation.
- candidate.py:61-70: legal_actions implementation.
- candidate.py:72-139: raw_step implementation.
- candidate.py:141-158: observe implementation.
- candidate.py:12-30: refresh implementation.
- candidate.py:22: n["agent_visibility"] = (same_team | detectable) - Rendered role visibility follows actual team/local detection.
- candidate.py:24: n["kit_visibility"] = same_team - Opponent charge is hidden independently of visible body position.
- candidate.py:106: scent = jnp.where(scrubbed, 0, scent) - An agent can erase another agent's information-bearing trace.
- candidate.py:116: scent = scent.at[cell].set - A paid decoy changes actual spatial scent, not a visual overlay.
- candidate.py:120: captured = jnp.any(tag[:2, None] - Adjacent physical capture follows movement.
- candidate.py:122: escaped = at_exit & s["agent_active"][2:] & ~captured - Capture takes priority over extraction.
- candidate.py:153: jnp.where(own_team, s["charge"] / 8.0, 0.0) - Native observations do not reveal opponent resources.

## Boundary and source provenance

This package is an initial author artifact, status AUTHORED_NOT_EXECUTED. It has not been imported, reset, stepped, JIT-compiled, vmapped, trained, rendered, or subjected to action search. Source AST screening and the copied data-only manifest/scene validators are separate from R0 and numerical parity. All playthroughs and paired interventions remain UNEXECUTED_AUTHOR_HYPOTHESIS; a later executor must use their literal keys and record actual legality, causes and failures.

Nonterminal raw steps advance int32 t once. Natural termination wins at H; truncated=(t>=H)&~terminated and done is their union. Already-done inputs preserve all state and observation leaves, clock and causes, return zero reward and all dones true. No raw reset occurs. Invalid/masked actions become wait; movement attempts can be blocked. State uses fixed-shape int32/bool/float32 arrays; keys remain external.

A trusted JaxMARL dictionary adapter, real replay/animation, browser controls, sealed R0, numerical parity and scientific comparisons remain independent work. No acceptance or learnability claim is made.

The supplied JaxMARL README informed parallel-agent API integration, Pgx README explicit keys/masks, Craftax README persistent spatial state, and PureJaxRL README functional training integration. No upstream code, game rules, layouts, assets or weights were copied. Craftax_Baselines_ppo_rnn.py was hash-verified only. These references do not replace the frozen learner.

Only the assigned batch design/request, contracts/parsers and copied references were read. All 20 input hashes/byte sizes and the authorization SHA matched. No outcomes, candidate sources, replays, logs, rankings, heldout files, result_N or another author's directory were read. Source split is train; final_test_used=false. The coordinator retains dispatch identity. Potential variants are ungenerated hypotheses. Static screening cannot prove solvability, non-isomorphism, strategic richness, learning or transfer; failures and indeterminate results remain valid.
