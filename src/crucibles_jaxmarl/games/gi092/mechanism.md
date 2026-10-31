# Lumen Locks: Cross-Wired Signals

Status: **AUTHORED_NOT_EXECUTED**. Candidate `game-v3-00092`, attempt 00; train-only prospective source. A/O/U/H = 3/154/10/128. Source SHA256: `6c3f3acbcd8f41c155ff02f5e78f4fa050fcb20678cb50867b5cba45e22b5155`.

Original three-role signal-routing and delivery world. A west surveyor and middle keeper hold different local clues but operate each other's keyed remote gate. A courier must physically carry two objects through the east corridor. No single role can reach all jobs.

## Literal rules and transition order

1. The static 13 by 7 map has a west room x=0..3, a middle room x=5..7 and an east courier corridor along x=10, with a loading bay across x=9..12 at y=6 and a small alcove at (11,3). Solid walls separate the roles.

2. Role 0's private sensor is (1,1), its panel (2,4). Role 1's sensor is (6,5), its panel (6,2). Sensor values are independent binary draws from the supplied reset key. Only the corresponding operator within Manhattan distance one receives the value; other roles and distant owners receive -1 with visibility false.

3. Every in-range action is a legal attempt while live. Wrong-role actions, unavailable panel/cargo actions and blocked moves do nothing. Invalid IDs become wait. After joint completion only wait is legal. All three agents remain active until joint completion.

4. Directional movement checks the previous state's gate_open flag, static walls and other roles' old occupancy. A correct pulse therefore opens entry for later raw calls, not retroactively on the current simultaneous move. An occupant may exit a gate cell after closure.

5. Signals are public one-bit observations with identity preserved by sender slot. Sending requires pre-step cooldown zero, sets lifetime four and cooldown three, and uses the action that otherwise could move or operate. Lifetimes decay each step. Expired values are -1 in observations, and signal entities become inactive.

6. At panel i, operator i can pulse bit zero or one if it has energy and the old jam timer is zero. The required bit is code_value[1-i]. Correct pulses set open timer six; wrong pulses set timer zero and jam two. Every accepted pulse costs one energy. Waiting exactly on one's own sensor restores one energy up to four.

7. Only role 2 can pick a colocated undelivered cargo with action 9. Holding cargo follows its real position. The same action releases it; release at (10,0) marks it delivered, elsewhere it becomes a ground object. Each delivery draws new private bits from the supplied step key, clears message lifetimes and closes gates. Draws on non-delivery steps are discarded.

8. Public correctness feedback permits a costly probing strategy; privacy is not a claim that brute force is impossible. Gate/cargo positioning, energy replenishment travel, clue changes and short opening windows make timely communication a plausible alternative. No omniscient role can do all physical jobs.

## Cross-agent consequences and untested paired interventions

- Changing a nearby operator from wait to broadcast changes the other roles' input for four frames while consuming the sender's movement turn. Its teammate can use that information at the cross-wired panel.

- Holding the courier action fixed, changing a correctly keyed panel pulse changes whether the courier can enter its gate on subsequent calls. A wrong pulse removes that opportunity and introduces a two-step lockout.

- A courier delivery changes both operators' private tasks by redrawing clues and closing gates, so repeating a memorized initial pulse is not a guaranteed second delivery strategy.

These are consequences derived from source inspection and proposed intervention traces. No paired-action execution or strategic-richness acceptance has been performed by the author.

## Partner-dependent strategies and possible variation

Unmeasured possibilities include exchanging clues before departing sensors, storing received bits while moving to panels, staggered gate opening synchronized to courier progress, or resource-costly panel probing. Courier progress signals compete with carrying movement; stale remembered messages are hazardous after a delivery.

Future separately authorized variants could alter code alphabets, message lifetime, corridor length, pulse duration or cross-wiring. They would test different information and timing requirements, but this package claims neither learned communication nor transfer.

## Information, rendering and invariants

Map, role positions, panel attempts, cargo, gate timers, pulse energy, cooldowns and live broadcasts are public. Sensor values are masked to -1 unless the querying role is that sensor's owner and within Manhattan distance one. Role 2 never directly observes either bit. The sensor entity visibility uses exactly the same stored code_visible mask. Expired signal entities are inactive and their observation values are -1.

The scene contains no executable rendering code. Its world, entity groups, attribute labels, event semantics, actions and observation labels are copied from this package's own manifest declarations. Visible positions bind directly to state arrays used in transitions. Inactive delivered objects are hidden as declared; the privileged debug state must be labelled separately from each role view. Events are last-transition flags, not cumulative event counts. The absorbing terminal keeps the final flags frozen; a repeated terminal call is not a newly occurring game event.

All positions are within x=0..12,y=0..6; roles have unique positions. Energy 0..4, cooldown 0..3, gate timer 0..6, jam 0..2, bit values 0..1, signal lifetime 0..4; hidden observation sentinel -1. Cargo owner is -1 or 2. Per-role reward is 0, 0.5 or 1; observations stay in [-1,1].

All integer state fields use int32, all logical leaves bool, and returned observations/rewards float32. No key is duplicated in state. Only reset and successful-delivery clue redraws use the supplied PRNG keys; non-delivery draws are discarded. Normalization uses powers of two. These are design and source properties, not an eager/JIT parity receipt.

## Terminal and integration contract

A live input increments scalar int32 t exactly once. Natural completion takes priority on the horizon: truncated is (next_t >= H) AND NOT terminated. episode_done equals terminated OR truncated. A terminal input returns the identical state/observation/clock, zero rewards, all dones true and the preserved natural/truncated cause. There is no reset inside raw_step. A trusted outer wrapper owns reset. Native learning receives only observe(state) and legal_actions(state).

The kernel is the homogeneous array boundary. A community JaxMARL dictionary adapter with per-agent dictionaries and __all__ is a separate coordinator-owned integration layer; its existence or correctness is not asserted by this source package. Completed individual deliveries do not remove a role from the joint action interface.

## Source line evidence

`candidate.py`, lines 17-21 (separated_roles_and_private_clues):

```python
        # EVIDENCE: separated_roles_and_private_clues
        terrain = jnp.ones((7, 13), dtype=jnp.int32)
        terrain = terrain.at[:, 0:4].set(0)
        terrain = terrain.at[:, 5:8].set(0)
        terrain = terrain.at[:, 10].set(0)
```

`candidate.py`, lines 74-78 (role_observation_and_signal_lifetime):

```python
        # EVIDENCE: role_observation_and_signal_lifetime
        private_codes = jnp.where(state["code_visible"], state["code_value"][None, :], -1).astype(jnp.float32)
        public = jnp.concatenate((
            state["agent_pos"].reshape(-1).astype(jnp.float32) / 16.0,
            state["energy"].astype(jnp.float32) / 4.0,
```

`candidate.py`, lines 113-117 (limited_causally_effective_messages):

```python
        # EVIDENCE: limited_causally_effective_messages
        signal_ttl = jnp.maximum(state["signal_ttl"] - 1, 0)
        cooldown = jnp.maximum(state["signal_cooldown"] - 1, 0)
        send = ((a == 7) | (a == 8)) & (state["signal_cooldown"] == 0)
        signal_value = jnp.where(send, a - 7, state["signal_value"])
```

`candidate.py`, lines 121-125 (cross_wired_remote_gates):

```python
        # EVIDENCE: cross_wired_remote_gates
        gate_timer = jnp.maximum(state["gate_timer"] - 1, 0)
        jam = jnp.maximum(state["jam"] - 1, 0)
        energy = state["energy"]
        at_sensor = jnp.concatenate((jnp.all(pos[:2] == state["code_pos"], axis=1), jnp.array([False])))
```

`candidate.py`, lines 143-147 (physical_cargo_and_round_change):

```python
        # EVIDENCE: physical_cargo_and_round_change
        owners = state["cargo_owner"]
        carried = owners == 2
        had_cargo = jnp.any(carried)
        matching = jnp.all(state["cargo_pos"] == pos[2], axis=1) & ~state["cargo_delivered"] & (owners < 0)
```

`candidate.py`, lines 166-170 (natural_priority_absorbing_terminal):

```python
        # EVIDENCE: natural_priority_absorbing_terminal
        n = dict(state)
        n["t"] = state["t"] + 1
        n["agent_pos"] = pos
        n["energy"] = energy
```

## Supplied community references and originality

- [JaxMARL_README.md](https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md) — Read supplied Quick Start/API text: parallel agent-keyed dictionaries and __all__ are an external adapter target; no upstream environment source or layout copied. Supplied-file SHA256: `6649d04f5ed05d620e78e8416a00d4db841cf2633ae60fdb3c9729e037a71c15`.

- [Pgx_README.md](https://github.com/sotetsuk/pgx/blob/main/README.md) — Read supplied Usage/API v2 text: JAX-native state transitions, explicit legal masks and caller-provided stochastic keys. No board rules, implementation or rendering copied. Supplied-file SHA256: `153f44999e55d5df3e51e13dee6ef418cc9704ddfa26206f1dc5789b9cbbf8df`.

- [Craftax_README.md](https://github.com/MichaelTMatthews/Craftax/blob/main/README.md) — Read supplied description, Basic Usage and reset discussion for spatial resource consequences and environment/algorithm separation. No game mechanics implementation, assets or layout copied. Supplied-file SHA256: `7f9757a2047143cf183f7e2e84851a740f386af80dbee050c89b316d66a3d3ff`.

- [PureJaxRL_README.md](https://github.com/luchris429/purejaxrl/blob/main/README.md) — Read supplied Code Philosophy and overview: single-file end-to-end JAX learning implementations are algorithm references, not a new game library. No learning code imported, copied or run. Supplied-file SHA256: `d5f40182c924a42cec820931e61a7d5070e48c4c8413c7d16455abc6ea0ef6b0`.

Only the supplied static README copies were consulted. The supplied Craftax PPO recurrent baseline was not read or used. Craftax is the game/mechanics reference; Pgx is a discrete-game API reference; PureJaxRL and the unconsulted PPO baseline are algorithm/integration references. No upstream code, benchmark layout, heldout implementation, project result, prior candidate, model call or network request was used. Common interface and terminal code here implements the supplied author contract, not an imported environment.

## Playthrough limits

`playthrough.json` is a fixed, in-range joint-action script with a reset seed. Every in-range attempt is legal while live. A replay driver must stop on done_all; only wait is legal afterward. Objectives and events in that file are intended hypotheses, not observations. The script is not a policy, solvability proof or success certificate. Candidate imports, reset/step calls, validators, JIT/vmap, training and rendering were not executed by this author. Root independently accepts or rejects the entire sealed batch after delivery.

