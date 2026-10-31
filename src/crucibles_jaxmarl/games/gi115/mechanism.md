# Freight canal locks and shared water

Status: **AUTHORED_NOT_EXECUTED**. Candidate `game-v3-00115`, source attempt 0, source split train, final test unused.

## Spatial rules and cross-agent dependence

Three freight boats navigate two basins joined by two three-cell lock chambers. The western basin is at reference level zero and the eastern basin at reference level two. Each chamber has an actual low gate at x=4 and high gate at x=6 on its row, y=1 or y=3. Boats cross only open gates. The goals are indexed physical docks: (9,1), (9,3) and (1,2). A completed boat unloads, becomes inactive and ceases to occupy traffic space.

There are exactly two modelled service-water units shared across both chambers and the central reservoir. A closed chamber can be filled by one reservoir unit, drained by one unit back to the reservoir, or transfer one unit directly to the other closed chamber. Every operation subtracts where it adds. Opening a low gate requires chamber level zero; opening a high gate requires level two; opposite gates cannot be open simultaneously. Filling one chamber to the high side uses all two units and prevents the other chamber from independently reaching high until water is returned or transferred. Basins have fixed reference elevations; this is a discrete lock service-water model, not an assertion of realistic full fluid conservation.

A boat may command the lane at x=3 through 7 on that lane's row. It can therefore operate from an approach or inside the chamber. Commands are joint, but one command per lane is selected by rotating boat priority. The lanes then settle in alternating order, which makes simultaneous claims on the last reservoir unit deterministic and explicit. Movement also uses rotating priority against the current occupied cells: a boat may follow a cell vacated earlier in that transition but may not enter a currently occupied cell. Opposing boats have to yield enough approach space; direct swaps do not pass through each other.

A boat earns six units for its own delivery and one unit for every completed delivery. Every still-active boat pays 0.02 per tick; a selected pump, drain or transfer attempt costs its issuer 0.03 even if a prerequisite prevents flow. These mixed incentives permit mutually helpful operations but also give a boat a reason to prefer its own passage, avoid paying a partner's operation cost or occupy an approach. The source does not assume that reward sharing creates cooperation.

## Partner-dependent strategies and limits

A same-direction convoy can enter one chamber, share a fill and pass the high gate. An opposing boat can yield the high entrance, then reuse the high chamber and drain for its westbound trip. Alternatively, two lanes can alternate by transferring water directly between closed chambers, avoiding a return through the reservoir but still requiring appropriate gate closure and access. Filling both chambers once leaves each at an intermediate level and the reservoir empty; resolving that state needs a drain or transfer and an agreed lane order. A boat blocking an exit can make an otherwise correctly filled lock unusable.

The finite script proposes a two-boat eastbound convoy followed by one westbound return through the upper chamber. Its thirty-one commands are source-derived intentions, not a run receipt. The paired resource test instead uses a shorter fixed prefix to place two boats at separate controls and close both low gates. Branch A fills the upper lane twice before the lower boat asks for water; branch B waits on the upper lane. The later lower command is identical. Compare actual level and stock arrays, gate-opening opportunity and subsequent positions. There need not be any immediate delivery reward difference, and water hoarding can be individually useful or jointly wasteful.

All game information is public. Chamber bars, gate flags, boat cargo, reservoir quantity, net flow and the current priority are represented in the flat observation. Net flow is a per-transition signed difference, so cancelling transfers can produce zero even if internal commands were attempted; it is not a total pumping meter. Dynamic entity positions and gate states are renderer bindings, not invented transitions.

Different chamber capacity, reservoir stock, goal directions, lane lengths or operation costs could alter convoy and yielding strategies. Such variants are future hypotheses, not implemented randomized regimes or measured generalization. Actual invariant, timing, replay and strategic checks remain required.


## Literal source evidence

- Real gate crossing constraints: `candidate.py:64` contains `pass_gate = (~low_cross | s['gate_open'][2 * lock_index]) & (~high_cross | s['gate_open'][2 * lock_index + 1])`.
- Physical occupancy arbitration: `candidate.py:69` contains `occupied = jnp.any(n['boat_active'] & (jnp.arange(3) != i) & jnp.all(n['boat_pos'] == target[i], axis=1))`.
- Nearby lane control: `candidate.py:72` contains `operating = (n['boat_pos'][:, 0] >= 3) & (n['boat_pos'][:, 0] <= 7) & (`.
- Rotating command selection: `candidate.py:80` contains `winner = jnp.argmin(jnp.where(eligible, rank, 9)).astype(jnp.int32)`.
- Shared water prerequisite: `candidate.py:91` contains `fill = (cmd == 8) & closed & (n['level'][k] < 2) & (n['stock'][0] > 0)`.
- Direct inter-chamber transfer: `candidate.py:93` contains `transfer = (cmd == 10) & closed & ~n['gate_open'][2 * other] & ~n['gate_open'][2 * other + 1] & (`.
- Exact opposing reservoir update: `candidate.py:97` contains `n['stock'] = n['stock'] - fill.astype(jnp.int32) + drain.astype(jnp.int32)`.
- Gate interlock: `candidate.py:98` contains `open_low = (cmd == 5) & (n['level'][k] == 0) & ~hi`.
- Actual delivered cargo: `candidate.py:107` contains `n['cargo'] = s['cargo'] - delivered_now.astype(jnp.int32)`.
- Recorded conservation quantity: `candidate.py:112` contains `n['water_total'] = jnp.array([jnp.sum(n['level']) + n['stock'][0]], jnp.int32)`.
- Mixed rewards: `candidate.py:119` contains `reward = (6.0 * delivered_now + jnp.sum(delivered_now.astype(jnp.float32)) -`.
- Absorbing raw state: `candidate.py:121` contains `n = {k: jnp.where(s['done'], s[k], n[k]) for k in s}`.

Method ranges: reset at candidate.py:12-37; observe at candidate.py:39-49; legal_actions at candidate.py:51-52; raw_step at candidate.py:54-124.

## Literal interface and execution boundary

The author supplied three-agent flat float32 observations, int32 joint actions, fixed-shape dictionary state and homogeneous discrete command attempts. The exact dimensions, every leaf, every observation column, reward bounds, actions and spatial bindings are in manifest.json. Unmet command prerequisites produce no effect; out-of-range integers are converted to wait. Each live raw transition advances t once. Natural completion takes precedence over an equal-tick timeout. Already-terminal raw calls preserve every state leaf, observation and clock, return zero reward and preserve actual terminal causes. There is no raw autoreset.

No candidate was imported or called. No JAX/NumPy import, compilation, numerical game call, training, model API, installation, SSH or remote work occurred during authoring. Standard-library AST and supplied data validators may establish only their literal static checks. They do not establish interface execution, numerical parity, conservation over actual traces, strategic richness, solvability, policy learning or transfer. AUTHORED_NOT_EXECUTED is the package status. The JaxMARL-compatible public dictionary adapter, real replay, RGB animation, interactive playback and role-view audit belong to the separately authorized implementation/validation stage and are not delivered or claimed here.

The playthrough JSON uses literal seeds and joint integer vectors. A trusted runner should use jax.random.PRNGKey(seed) for reset, then jax.random.fold_in(reset_key, step_index+1) for raw-step keys; these are prospective instructions and were not executed. All four kernels use step keys only as interface inputs, so their transition stochasticity is reset-derived. Scripts stop at joint done. They are author hypotheses, not successful trajectories or acceptance gates. Paired tests freeze the shared prefix and the non-intervened actions; they must retain zero, negative, mixed, failure and indeterminate outcomes.

The source uses int32 for small integer categories, owners and counters and bool for masks/terminal causes. The uint32 PRNG key remains trusted trajectory provenance rather than a duplicated state leaf. Render data is declarative. Map and entity positions bind real state. An omniscient debug state, where provided later, must be labelled separately and never substituted for a role observation.

## Official ecosystem inspiration and source use

Only this batch's physically supplied official README excerpts were read for broad ecosystem context: JaxMARL separates MARL environments and algorithms and exposes multi-agent interfaces; Pgx exemplifies JAX-native board states, legal action masks and explicit step randomness; Craftax illustrates persistent spatial game state, inventories and separate reset/render interfaces. These are conceptual influences. No upstream source code, environment layout, asset, baseline or algorithm was copied into this original kernel.

The supplied JaxMARL README identifies Apache-2.0, the Pgx README Apache-2.0, and the Craftax README MIT. Those are descriptions of upstream snapshots, not a new license grant for this package; an original-package release license remains the coordinator's decision. PureJaxRL describes algorithm implementations rather than an environment library. Craftax_Baselines_ppo_rnn.py was supplied and hash-accounted for but was not opened or reused. No baseline replaces the frozen experimental learner. Exact source snapshot URLs and hashes are in generation_record.json; no batch-external source or candidate feedback was consulted.

Generating or replaying these rules would not establish large-scale meta-RL adaptation. Rule/role/map variations discussed above are future research hypotheses requiring their own frozen source, cost and information controls, and genuine train/develop/confirm/final-transfer evidence. They are not implemented variants or measured generalization.
