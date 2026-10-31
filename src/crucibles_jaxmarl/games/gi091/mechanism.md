# Ember Weir: Shared Firebreaks

Status: **AUTHORED_NOT_EXECUTED**. Candidate `game-v3-00091`, attempt 00; train-only prospective source. A/O/U/H = 3/293/10/96. Source SHA256: `ecdc0a48686f6ef2675b5c36074a44339a2882bfef85950b50df836fd7e09d40`.

Original cooperative containment world. The central stone spine leaves two traversable fire corridors, so a barrier can protect the community while cutting a worker's route. Finite shared stock, handoffs, fuel exhaustion and timed erosion generate persistent cross-agent effects.

## Literal rules and transition order

1. Initial flames occupy (8,2) and (8,4); their own fuel is already consumed. Terrain stones at x=4,y=0,1,3,5,6 leave corridors y=2 and y=4. Assets at (1,1),(1,5),(3,3) each have three integrity; the nonflammable depot at (0,3) holds eighteen units, in addition to three carried by each worker.

2. Actions 1..4 both attempt movement and set facing, including when blocked. Firebreaks and stones block movement. Another role's previous cell and a duplicated target block movement. All in-range IDs are legal attempts; unavailable interactions do nothing. Invalid IDs and any actions while stunned become wait. Only wait is legal after joint termination.

3. Interactions are processed in role order rotated by the current clock. Spray spends one water and extinguishes current/front cells. Construction spends two and requires empty grass, no worker, no flame and no existing barrier in the front cell. It produces strength two.

4. Refill draws one unit when within Manhattan distance one of the depot. Give moves one unit to an adjacent other role with capacity, using rotating priority for multiple recipients. Dismantling a front firebreak removes its entire remaining strength and refunds at most one water; it can reopen a teammate's route but reopen a fire corridor as well.

5. After all interventions, cardinal neighbors of remaining fire are exposed. On every third step exposed unused fuel ignites unless a firebreak is present. Existing fire loses one lifetime each step; new ignition starts at three and consumes fuel permanently. An exposed barrier loses one strength on each spread tick after blocking that tick.

6. Assets in the resulting fire lose one integrity. A worker in the resulting fire loses one carried water and receives stun two. A non-burning worker's stun decays. Workers remain physical obstacles and can later act again; there is no individual early termination.

7. All-fire extinction with a surviving asset is natural success. All assets destroyed is natural failure. Damage incurs a shared penalty of 1/16 per asset on that step; success gives +1 and failure -1, so a failure/damage step can be below -1.

## Cross-agent consequences and untested paired interventions

- A worker who builds a front firebreak changes the next spread into that physical cell and a partner's ability to walk through it. Dismantling reverses both effects with delayed fire consequences.

- Drawing the last depot unit deprives a later teammate of refill; giving water can let that teammate build a barrier they otherwise could not afford. These are transfers of the same finite state resource.

- Suppressing a source before a third-step spread removes ignition opportunities for every adjacent cell, potentially preserving a partner's route or preventing shared asset loss several steps later.

These are consequences derived from source inspection and proposed intervention traces. No paired-action execution or strategic-richness acceptance has been performed by the author.

## Partner-dependent strategies and possible variation

Unmeasured alternatives include early coordinated corridor sealing, direct forward suppression with a supply runner, or sacrificing an outlying asset to conserve water for the remaining two. Barriers trade immediate protection against teammate routing; spray and construction compete for the same water. Action order and fire phase matter.

Separately authorized future map/fuel arrangements, barrier endurance, depot stock or observation restrictions could change which corridors and handoffs matter. This package implements only the stated fully observable rules; no measured learning or transfer is claimed.

## Information, rendering and invariants

The entire physical world and resource state are public. All groups have all-true role visibility and the static terrain is public. Observation includes the complete fire, fuel and barrier arrays, all worker resources and all protected assets.

The scene contains no executable rendering code. Its world, entity groups, attribute labels, event semantics, actions and observation labels are copied from this package's own manifest declarations. Visible positions bind directly to state arrays used in transitions. Inactive delivered objects are hidden as declared; the privileged debug state must be labelled separately from each role view. Events are last-transition flags, not cumulative event counts. The absorbing terminal keeps the final flags frozen; a repeated terminal call is not a newly occurring game event.

Worker xy lies in x=0..8,y=0..6 with unique positions; water 0..3, facing 1..4, stun 0..2, stock 0..18, fire 0..3, fuel 0..1, barriers 0..2, asset integrity 0..3. Reward lies in [-1.1875,1], enclosed by manifest [-2,1]. All observation values are in [0,1], enclosed by [-1,1].

All integer state fields use int32, all logical leaves bool, and returned observations/rewards float32. No key is duplicated in state. This particular source is deterministic and does not use the supplied stochastic key. Normalization uses powers of two. These are design and source properties, not an eager/JIT parity receipt.

## Terminal and integration contract

A live input increments scalar int32 t exactly once. Natural completion takes priority on the horizon: truncated is (next_t >= H) AND NOT terminated. episode_done equals terminated OR truncated. A terminal input returns the identical state/observation/clock, zero rewards, all dones true and the preserved natural/truncated cause. There is no reset inside raw_step. A trusted outer wrapper owns reset. Native learning receives only observe(state) and legal_actions(state).

The kernel is the homogeneous array boundary. A community JaxMARL dictionary adapter with per-agent dictionaries and __all__ is a separate coordinator-owned integration layer; its existence or correctness is not asserted by this source package. Completed individual deliveries do not remove a role from the joint action interface.

## Source line evidence

`candidate.py`, lines 12-16 (original_world):

```python
        # EVIDENCE: original_world
        cells = jnp.stack((jnp.mod(jnp.arange(63), 9), jnp.arange(63) // 9), axis=1).astype(jnp.int32)
        terrain = jnp.zeros((7, 9), dtype=jnp.int32)
        terrain = terrain.at[jnp.array([0, 1, 3, 5, 6]), 4].set(1)
        terrain = terrain.at[3, 0].set(2)
```

`candidate.py`, lines 57-61 (public_observation):

```python
        # EVIDENCE: public_observation
        public = jnp.concatenate((
            state["agent_pos"].reshape(-1).astype(jnp.float32) / 16.0,
            state["water"].astype(jnp.float32) / 4.0,
            state["facing"].astype(jnp.float32) / 4.0,
```

`candidate.py`, lines 87-91 (barriers_change_partner_routes):

```python
        # EVIDENCE: barriers_change_partner_routes
        blocked = (state["map"][target[:, 1], target[:, 0]] == 1) | (state["barrier"][index] > 0)
        occupied = jnp.any(jnp.all(target[:, None, :] == old[None, :, :], axis=2) & (ids[:, None] != ids[None, :]), axis=1)
        duplicate = jnp.sum(jnp.all(target[:, None, :] == target[None, :, :], axis=2), axis=1) > 1
        pos = jnp.where((is_move & ~blocked & ~occupied & ~duplicate)[:, None], target, old)
```

`candidate.py`, lines 104-108 (scarce_shared_resources_and_transfers):

```python
        # EVIDENCE: scarce_shared_resources_and_transfers
        for k in range(3):
            i = jnp.mod(k + state["t"], 3)
            spray = (a[i] == 5) & (water[i] > 0)
            sprayed_cells = (cell_ids == current_index[i]) | ((cell_ids == front_index[i]) & inside[i])
```

`candidate.py`, lines 131-135 (persistent_spreading_hazard):

```python
        # EVIDENCE: persistent_spreading_hazard
        adjacent = jnp.sum(jnp.abs(state["cell_pos"][:, None, :] - state["cell_pos"][None, :, :]), axis=2) == 1
        exposed = jnp.any((fire > 0)[:, None] & adjacent, axis=0)
        spread_tick = jnp.mod(state["t"] + 1, 3) == 0
        ignition = spread_tick & exposed & (state["fuel"] > 0) & (barrier == 0)
```

`candidate.py`, lines 153-157 (natural_priority_absorbing_terminal):

```python
        # EVIDENCE: natural_priority_absorbing_terminal
        n = dict(state)
        n["t"] = state["t"] + 1
        n["agent_pos"] = pos
        n["water"] = water
```

## Supplied community references and originality

- [JaxMARL_README.md](https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md) — Read supplied Quick Start/API text: parallel agent-keyed dictionaries and __all__ are an external adapter target; no upstream environment source or layout copied. Supplied-file SHA256: `6649d04f5ed05d620e78e8416a00d4db841cf2633ae60fdb3c9729e037a71c15`.

- [Pgx_README.md](https://github.com/sotetsuk/pgx/blob/main/README.md) — Read supplied Usage/API v2 text: JAX-native state transitions, explicit legal masks and caller-provided stochastic keys. No board rules, implementation or rendering copied. Supplied-file SHA256: `153f44999e55d5df3e51e13dee6ef418cc9704ddfa26206f1dc5789b9cbbf8df`.

- [Craftax_README.md](https://github.com/MichaelTMatthews/Craftax/blob/main/README.md) — Read supplied description, Basic Usage and reset discussion for spatial resource consequences and environment/algorithm separation. No game mechanics implementation, assets or layout copied. Supplied-file SHA256: `7f9757a2047143cf183f7e2e84851a740f386af80dbee050c89b316d66a3d3ff`.

- [PureJaxRL_README.md](https://github.com/luchris429/purejaxrl/blob/main/README.md) — Read supplied Code Philosophy and overview: single-file end-to-end JAX learning implementations are algorithm references, not a new game library. No learning code imported, copied or run. Supplied-file SHA256: `d5f40182c924a42cec820931e61a7d5070e48c4c8413c7d16455abc6ea0ef6b0`.

Only the supplied static README copies were consulted. The supplied Craftax PPO recurrent baseline was not read or used. Craftax is the game/mechanics reference; Pgx is a discrete-game API reference; PureJaxRL and the unconsulted PPO baseline are algorithm/integration references. No upstream code, benchmark layout, heldout implementation, project result, prior candidate, model call or network request was used. Common interface and terminal code here implements the supplied author contract, not an imported environment.

## Playthrough limits

`playthrough.json` is a fixed, in-range joint-action script with a reset seed. Every in-range attempt is legal while live. A replay driver must stop on done_all; only wait is legal afterward. Objectives and events in that file are intended hypotheses, not observations. The script is not a policy, solvability proof or success certificate. Candidate imports, reset/step calls, validators, JIT/vmap, training and rendering were not executed by this author. Root independently accepts or rejects the entire sealed batch after delivery.

