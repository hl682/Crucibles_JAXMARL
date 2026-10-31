# Lockstep Gantry

**Status: AUTHORED_NOT_EXECUTED.**

A cooperative workshop game in which a five-cell-long carrier/beam assembly must negotiate a narrow shutter or a longer northern bypass. The tender is a spatial worker with the same potential jobs, not an out-of-world gate switch.

## Rules and real cross-agent dependencies

The beam is three cells long. Its two handles are at center plus/minus two cells along its current axis. TOGGLE GRIP attaches a free worker standing on a handle, or releases an attached worker. Exactly two distinct handle holders must issue the same cardinal command to translate the beam, or both issue QUARTER TURN to rotate it. A disagreement or a single remaining holder leaves it in place. The carrier positions move with the accepted beam geometry; they are not decorative positions.

PUMP requires a free worker exactly at [5,2]. It adds three shared pressure units, capped at ten, and resets the shutter timer to four before the proposed cargo motion is checked. Other turns decrement the timer by one. A translation spends one pressure unit, and a rotation spends two. Rotations require a clear full 5x5 swept square, including all walls and noncarrier workers, so a pole cannot simply pivot through a wall. A move can consume charge a partner needed later or put its handle across another worker's route.

The wall at x=6 has the shutter at y=4 and an open northern bypass at y=1. Only a horizontal assembly fits through the central aperture. The bypass trades extra charge and travel for avoiding shutter expiry. Role exchange is possible by releasing and walking to an available handle, but the released assembly does not move under one remaining carrier. The tender can intentionally or accidentally block a destination or rotation sweep. These mechanisms create joint-command, geometry, timing, job-allocation and common-resource dependencies within the transitions.

A closing shutter on any worker or beam cell ends the episode as a jam, including a carrier whose handle is still passing through. Arrival means horizontal center [10,4] and no jam. The dock target is a game objective, not an experimental acceptance threshold.

## Kernel, information and rewards

A=3, O=40, U=8, H=80. Coordinates are integer [x,y], with y increasing downward. All state leaves are fixed-shape int32, float32 or bool arrays; all observations and rewards are float32.

Reset is deterministic and starts at horizontal center [3,4] with six pressure units, ungripped workers [1,4], [5,4], [5,2], and a closed shutter. All dynamic state needed to interpret the task is public. Static map layout is explicitly public in the manifest; the map is not inferred from an unobserved leaf.

Each live step handles grip changes, pumping and timer decrement, agreement/clearance and cargo motion, then free-worker movement and terminal assessment. Ordinary walking cannot enter another worker's old cell, including a cell the other worker will vacate, and all contenders for the same free destination are blocked. Thus there are no swaps or overlaps. A noncarrier's departure does not retroactively clear the cargo's earlier geometry test. This conservative simultaneity rule is part of the game.

Rewards are -0.01 per role per live step, an additional -0.02 for the pumping role, +8 to every role on delivery, and -5 to every role on a jam. The declared [-6,9] envelope covers all possible one-step rewards. Coordinates, identity, grip, pressure, timer, clock, terminal causes and event bits are encoded into exactly 40 float32 columns in [0,1].

The binding addendum is implemented: a live input advances the scalar int32 clock by one; natural completion takes priority over the horizon. Already-terminal raw calls preserve the complete state, observation, clock, terminal causes and final event flags, return zero rewards and all-true dones. They are not extra gameplay transitions. The only legal terminal action is WAIT. Autoreset belongs to an external wrapper. No individual is removed from the world before joint completion. Invalid or out-of-range actions become WAIT; blocked cardinal directions remain legal attempted movements and may produce no motion.

Each observation slice, entity position, active mask, visibility mask and displayed attribute is declared in manifest.json. scene.json repeats exactly the hash-bound map and declared channels. Its motion comes only from real state positions. A debug full-state view must remain distinct from role views. Public event channels are explicitly present in each role observation; local object/agent attributes are zero-masked together with a visible bit. A hidden entity is not a position hint.

## Literal source evidence

- Physical length and coordinate construction: candidate.py:L19 — `return center + jnp.arange(-1, 2, dtype=jnp.int32)[:, None] * unit`
- Two distinct cooperating ends: candidate.py:L125 — `ready = ready & jnp.any(grip == 0) & jnp.any(grip == 1)`
- Equal commands are required: candidate.py:L127 — `agreement = jnp.all(~crew | (a == command))`
- Third-party supply changes the shared resource: candidate.py:L120 — `pressure = jnp.minimum(10, s["pressure"][0] + 3 * jnp.sum(pump.astype(jnp.int32)))`
- Third-party pump refreshes the actual timer: candidate.py:L121 — `timer = jnp.where(jnp.any(pump), 4, jnp.maximum(0, s["gate_timer"][0] - 1))`
- Physical swept-clearance constraint: candidate.py:L145 — `rotation_clear = jnp.all(self.clear(s, sweep, gate_open)) & ~sweep_block`
- Common resource and agreement jointly gate motion: candidate.py:L147 — `operated = ready & agreement & (translate | rotate) & (pressure >= cost)`
- Terminal consequence of a missed pump: candidate.py:L167 — `jammed = (~gate_open) & in_gate`
- Actual task completion: candidate.py:L168 — `success = jnp.all(center == s["dock_pos"][0]) & (axis == 0) & ~jammed`
- Natural-priority terminal bound: candidate.py:L171 — `truncated = (t >= self.max_steps) & ~terminated`

Method extents in this exact source:

- `direction`: candidate.py:L12-L15.
- `cells`: candidate.py:L17-L19.
- `handles`: candidate.py:L21-L23.
- `clear`: candidate.py:L25-L31.
- `reset`: candidate.py:L33-L72.
- `observe`: candidate.py:L74-L94.
- `legal_actions`: candidate.py:L96-L106.
- `advance`: candidate.py:L108-L194.
- `absorb`: candidate.py:L196-L199.
- `raw_step`: candidate.py:L201-L202.

## Fixed witnesses and intended intervention differences

The main fixed script is an eight-transition whole-task attempt. The missed-refresh pair differs in one tender action, then uses the same carrier commands; it is intended to expose a delayed failure rather than merely a different immediate reward. The disagreement script is local and stops before delivery.

No scripted outcome is an execution receipt. A successful-looking path in prose is only a prediction from the rules. Scripts that exercise local mechanisms intentionally do not assert whole-task completion.

## Invariants, variation and limits

Declared invariants include unique worker positions and nonnegative clock, hydraulic charge, shutter timer and translation count. Construction additionally keeps charge at most 10, timer at most 4, the three cargo cells contiguous, the two attached handle roles distinct, and every occupied cell inside the workshop and off solid walls. A closed gate may coincide with the final jam state by design. Altering aperture locations, dock orientation, pressure capacity or pump lifetime would change routing and job-allocation decisions, but no such variants or transfer results are implemented or measured here.

Integer sign-bearing fields (for example no-owner=-1) remain exact int32 values. There is no design-specified int8 distribution to adapt. The external uint32 PRNG key is not duplicated as a state leaf. This particular world is deterministic; reset and step accept keys for the fixed pure interface without consuming them.

## Provenance and reference use

This is an original authored kernel for initial window0, attempt00, in the train source stream. No prior candidate, project execution result, heldout layout, evaluation score, ranking or final test was read. No candidate or JAX was imported or run. The action lists are fixed hypotheses, not recorded trajectories. Interface/R0 checks, numerical parity, actual legality along the scripts, intervention replays, animation, training, solvability, optimality, strategic richness and meta-learning effects remain unestablished.

Only supplied official references were used. JaxMARL's README informed the intended simultaneous multi-agent adapter boundary (agent-keyed dictionaries and __all__ belong in the later integration layer). Pgx's README informed the separation of explicit transition keys, legal masks and raw terminal state. Craftax's README informed the distinction between a world kernel, state-based rendering and an external reset wrapper. PureJaxRL is an algorithm implementation reference, not a game library; its performance statements are not evidence about this game. Craftax_Baselines/ppo_rnn.py was read only at its environment/reset wrapper interfaces; no PPO/network code or benchmark mechanics were copied. No upstream implementation or map is incorporated. These references are attributed in generation_record.json with exact supplied file hashes and upstream URLs.

The copied community_delivery_requirements.json still lists IDs 82-85. The exact assigned design/request IDs 86-89 control this batch; the general interaction and presentation requirements are applied without rewriting that input. Actual dispatch identity, backend and usage evidence are retained by the coordinator, not invented here.
