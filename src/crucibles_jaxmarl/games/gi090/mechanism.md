# Relayline: The Four-Tick Ball

Status: **AUTHORED_NOT_EXECUTED**. Candidate `game-v3-00090`, attempt 00; train-only prospective source. A/O/U/H = 4/126/8/128. Source SHA256: `af94258741acd76dc181234a66519f6f40e7961555b369e484b0b98ba30cf989`.

Original two-versus-two court with a physical one-cell projectile, possession-dependent relay eligibility and a short possession clock. No upstream game rules, layouts, implementation or heldout evidence were copied.

## Literal rules and transition order

1. Roles 0/1 attack the goal at (10,3); roles 2/3 attack (0,3). The midfield stones at x=5, y=0,1,5,6 make the three central rows the only crossing band. All positions determine movement, tackling, catches and scoring.

2. Each action is an attempt. Out-of-range action IDs become wait. Every in-range action is legal while live; unmet spatial/resource conditions cause no effect. At joint termination only action 0 is legal. All roles remain active together.

3. Movement proposals are clipped to the court. Stones, another role's previous position, or a duplicated proposed destination block movement. Swaps and same-step following into vacated cells are intentionally blocked.

4. After movement, existing free-ball flight advances one cardinal cell. Stones or a boundary stop flight. A runner on the resulting ball cell catches it. A same-team catch by a role other than the sender grants relay_ready; an opponent catch or successful tackle removes it.

5. Tackling is resolved after catches, and spends two stamina even when it fails. The nearest adjacency test uses actual post-move positions. Multiple eligible roles are ordered by the clock-rotated role rank. Bracing blocks tackles only with positive pre-step stamina.

6. The holder can launch toward its unique teammate only if cardinally aligned and one to four cells away. Passing spends one stamina, leaves the ball at the launch cell for this frame, and starts travel on the next step. The target direction is frozen; a receiver who moves away can miss the ball.

7. Hold starts at zero on a new owner and rises while that owner retains possession. At four it spills unless a pass launches on that step. Brace can buy tactical protection but cannot hold forever. A spilled ball remains at its real last holder position and loses relay eligibility.

## Cross-agent consequences and untested paired interventions

- Holding role 1 fixed on the pass endpoint, changing role 0 from wait to pass changes role 1's later possession and scoring eligibility. Moving role 1 off that endpoint instead changes the future ball location/owner.

- An opponent who waits versus tackles next to an unguarded holder changes possession, both teams' scoring opportunity and the eventual zero-sum terminal payoff; a simultaneous brace can reverse that opportunity.

- An off-ball runner occupying a corridor cell prevents another role from entering it, including a teammate. Waiting restores a future movement/pass budget but advances the holder's spill clock.

These are consequences derived from source inspection and proposed intervention traces. No paired-action execution or strategic-richness acceptance has been performed by the author.

## Partner-dependent strategies and possible variation

Plausible unmeasured strategies include leapfrog relays with receivers moving ahead before the launch, safe short passes around a defender, screening by position, and timed tackle/brace exchanges. Always waiting spills possession; repeated solo rushing cannot create relay eligibility. Passing early reduces spill risk but exposes a predictable flight line.

Future separately authorized variations could alter the court band, stamina budget, hold limit, pass range or team orientation, changing spacing and defense. None of those variants is implemented here, and no non-isomorphism, learning or generalization result is claimed.

## Information, rendering and invariants

Every actual game variable affecting transition is public or a fixed known constant. All entities and the entire map are visible to every role. Observations contain the full public physical state and own role identity; no renderer-only policy information.

The scene contains no executable rendering code. Its world, entity groups, attribute labels, event semantics, actions and observation labels are copied from this package's own manifest declarations. Visible positions bind directly to state arrays used in transitions. Inactive delivered objects are hidden as declared; the privileged debug state must be labelled separately from each role view. Events are last-transition flags, not cumulative event counts. The absorbing terminal keeps the final flags frozen; a repeated terminal call is not a newly occurring game event.

Positions stay within x=0..10,y=0..6; stamina 0..4; owner/source -1..3; team -1..1; flight 0..4; hold 0..3 in returned live states. Agent positions are unique. Rewards are -1,0,+1. Observation bounds are [-1,1].

All integer state fields use int32, all logical leaves bool, and returned observations/rewards float32. No key is duplicated in state. This particular source is deterministic and does not use the supplied stochastic key. Normalization uses powers of two. These are design and source properties, not an eager/JIT parity receipt.

## Terminal and integration contract

A live input increments scalar int32 t exactly once. Natural completion takes priority on the horizon: truncated is (next_t >= H) AND NOT terminated. episode_done equals terminated OR truncated. A terminal input returns the identical state/observation/clock, zero rewards, all dones true and the preserved natural/truncated cause. There is no reset inside raw_step. A trusted outer wrapper owns reset. Native learning receives only observe(state) and legal_actions(state).

The kernel is the homogeneous array boundary. A community JaxMARL dictionary adapter with per-agent dictionaries and __all__ is a separate coordinator-owned integration layer; its existence or correctness is not asserted by this source package. Completed individual deliveries do not remove a role from the joint action interface.

## Source line evidence

`candidate.py`, lines 12-16 (original_world):

```python
        # EVIDENCE: original_world
        board = jnp.zeros((7, 11), dtype=jnp.int32)
        board = board.at[jnp.array([0, 1, 5, 6]), 5].set(1)
        state = {
            "t": jnp.array(0, dtype=jnp.int32),
```

`candidate.py`, lines 46-50 (action_attempts):

```python
        # EVIDENCE: action_attempts
        live = jnp.ones((4, 8), dtype=jnp.bool_)
        resting = jnp.broadcast_to(jnp.arange(8)[None, :] == 0, (4, 8))
        return jnp.where(state["episode_done"], resting, live)

```

`candidate.py`, lines 52-56 (public_observation):

```python
        # EVIDENCE: public_observation
        public = jnp.concatenate((
            state["agent_pos"].reshape(-1).astype(jnp.float32) / 16.0,
            state["stamina"].astype(jnp.float32) / 4.0,
            state["team"].astype(jnp.float32),
```

`candidate.py`, lines 81-85 (movement_and_fatigue):

```python
        # EVIDENCE: movement_and_fatigue
        target = jnp.clip(old + delta[a], jnp.array([0, 0]), jnp.array([10, 6]))
        moving = (a >= 1) & (a <= 4) & (state["stamina"] > 0)
        target = jnp.where(moving[:, None], target, old)
        wall = state["map"][target[:, 1], target[:, 0]] == 1
```

`candidate.py`, lines 94-98 (real_ball_flight_and_interception):

```python
        # EVIDENCE: real_ball_flight_and_interception
        owner = state["ball_owner"][0]
        ball = state["ball_pos"][0]
        velocity = state["ball_velocity"][0]
        left = state["flight_left"][0]
```

`candidate.py`, lines 131-135 (teammate_pass_and_hold_limit):

```python
        # EVIDENCE: teammate_pass_and_hold_limit
        holder = jnp.clip(owner, 0, 3)
        teammate = jnp.array([1, 0, 3, 2], dtype=jnp.int32)[holder]
        difference = pos[teammate] - pos[holder]
        distance = jnp.sum(jnp.abs(difference))
```

`candidate.py`, lines 157-161 (natural_priority_absorbing_terminal):

```python
        # EVIDENCE: natural_priority_absorbing_terminal
        n = dict(state)
        n["t"] = state["t"] + 1
        n["agent_pos"] = pos
        n["stamina"] = stamina
```

## Supplied community references and originality

- [JaxMARL_README.md](https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md) — Read supplied Quick Start/API text: parallel agent-keyed dictionaries and __all__ are an external adapter target; no upstream environment source or layout copied. Supplied-file SHA256: `6649d04f5ed05d620e78e8416a00d4db841cf2633ae60fdb3c9729e037a71c15`.

- [Pgx_README.md](https://github.com/sotetsuk/pgx/blob/main/README.md) — Read supplied Usage/API v2 text: JAX-native state transitions, explicit legal masks and caller-provided stochastic keys. No board rules, implementation or rendering copied. Supplied-file SHA256: `153f44999e55d5df3e51e13dee6ef418cc9704ddfa26206f1dc5789b9cbbf8df`.

- [Craftax_README.md](https://github.com/MichaelTMatthews/Craftax/blob/main/README.md) — Read supplied description, Basic Usage and reset discussion for spatial resource consequences and environment/algorithm separation. No game mechanics implementation, assets or layout copied. Supplied-file SHA256: `7f9757a2047143cf183f7e2e84851a740f386af80dbee050c89b316d66a3d3ff`.

- [PureJaxRL_README.md](https://github.com/luchris429/purejaxrl/blob/main/README.md) — Read supplied Code Philosophy and overview: single-file end-to-end JAX learning implementations are algorithm references, not a new game library. No learning code imported, copied or run. Supplied-file SHA256: `d5f40182c924a42cec820931e61a7d5070e48c4c8413c7d16455abc6ea0ef6b0`.

Only the supplied static README copies were consulted. The supplied Craftax PPO recurrent baseline was not read or used. Craftax is the game/mechanics reference; Pgx is a discrete-game API reference; PureJaxRL and the unconsulted PPO baseline are algorithm/integration references. No upstream code, benchmark layout, heldout implementation, project result, prior candidate, model call or network request was used. Common interface and terminal code here implements the supplied author contract, not an imported environment.

## Playthrough limits

`playthrough.json` is a fixed, in-range joint-action script with a reset seed. Every in-range attempt is legal while live. A replay driver must stop on done_all; only wait is legal afterward. Objectives and events in that file are intended hypotheses, not observations. The script is not a policy, solvability proof or success certificate. Candidate imports, reset/step calls, validators, JIT/vmap, training and rendering were not executed by this author. Root independently accepts or rejects the entire sealed batch after delivery.

