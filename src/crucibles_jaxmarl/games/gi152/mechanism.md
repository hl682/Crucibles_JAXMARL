# Lane Relay Gridball

**AUTHOR_ONLY_NOT_EXECUTED.** This is initial parentless proposal 152, source attempt 0, in the frozen train-only window. Candidate runtime execution, numerical R0, replay, rendered animation, learning and scientific acceptance have not occurred in this authoring task. Python AST/syntax and declarative data-shape/schema screens are authoring checks only; they do not establish any game outcome.

Native contract: A=4, O=116, U=9, H=160. Observations are float32 `[A,O]`, actions int32 `[A]`, rewards float32 `[A]`, masks bool `[A,U]`, and state leaves are fixed-shape int32/bool arrays. Source SHA-256: `5ad8900a38e4b890b8ff1ebb06a3288622bab398814e035089f9f440a9c15dc8`.

## Task and rules

Two teammates create passing and shooting lanes against two opponents; score three goals before the opposing team while managing stamina and possession.

The one ball has a single position, owner, velocity and remaining flight count. A pass is not an instant reward exchange: it launches toward a teammate who must be aligned within five cells after simultaneous movement. The ball then moves one cell each raw step; the receiver may move, and a defender may occupy or actively guard a lane. Shooting is restricted to the center row, so teammates can draw pressure away from that row or offer a safer pass.

Movement resolves first. Adjacent unguarded carriers can then be tackled; a newly stealing player cannot launch the ball on the same tick. Successful pass/shot launches follow. For the next ball cell, exact-cell catches outrank adjacent braced catches, then rotating priority resolves ties. Catching is checked before goal entry, so a bracing defender can save a shot from an adjacent cell. A wall or obstacle stops flight at the previous cell. An exhausted player may remain a body blocker but cannot brace, tackle or move until resting.

Dribble-and-screen, give-and-go, guarded possession and pressing for contact turnovers are distinct plausible strategies. Their usefulness depends on teammate alignment, opponent guarding and stamina. Bracing denies a lane but costs stamina and a movement opportunity. Passing to a stationary teammate may be safe against a resting opponent and immediately intercepted when that opponent braces. None of these prospective statements establishes learnability, optimal play or strategic richness by measurement.

After a goal, the actual scoring frame remains in state. A nonterminal goal enables only action 0 for one subsequent kickoff tick. That tick restores the fixed formation and stamina, gives the conceding team player 0 or 2 possession, clears transient event flags, preserves score and advances the match clock. It never calls reset and does not hide an episode reset. A natural final goal or a horizon boundary preserves the terminal frame under the absorbing raw contract.

There is always exactly one active ball. A possessed ball position equals its carrier's position; otherwise its stored position is the actual loose/flight coordinate. Stamina is in 0..5, player positions remain unique, and at most one goal is credited per tick. These are declared source properties for later numerical checks. Future variants could change lane obstacles, recovery, permitted pass range or team roles; no such variant benchmark has run. Reset is deterministic and ignores the supplied key.

## Actions and simultaneous timing

- `0`: Rest and recover two stamina.
- `1`: Move north, spending one stamina.
- `2`: Move east, spending one stamina.
- `3`: Move south, spending one stamina.
- `4`: Move west, spending one stamina.
- `5`: Pass toward aligned teammate within five cells, spending one stamina.
- `6`: Shoot toward opposing goal from center row, spending two stamina.
- `7`: Brace: protect possession or catch within one cell, spending one stamina.
- `8`: Attempt adjacent opponent tackle, spending two stamina.

Movement rejects another agent's start-of-tick cell even if that agent moves; swaps are blocked. Among proposals to the same empty cell, the smallest (agent_id - t) mod A wins. Terrain or joint contention may make a legal movement attempt stationary.

Out-of-range or masked actions deterministically become action 0. Legal interaction attempts may still do nothing when their resolution-phase prerequisites are absent; the action labels state their intended operation. All roles remain active until joint termination, so no eliminated role is silently deleted. Terminal masks retain action 0.

## Observation and renderer contract

Fully observable playing field, all entity attributes, map, clock, score, restart state and event flags. Each observation adds own player id. No separate privileged role view is needed; all declared entity visibility is true.

- Columns `0:1` (self): Own immutable player id.
- Columns `1:12` (public): Clock,score amber,score indigo,restart pending,goal amber,goal indigo,pass event,catch event,intercept event,tackle event,joint done.
- Columns `12:32` (public): Four players in id order: x,y,stamina,team,guard.
- Columns `32:39` (public): Ball x,y,carrier id (-1 free),velocity x,velocity y,remaining flight,last-touch team.
- Columns `39:116` (public): Static 7 by 11 public tile map, row-major yx.

All coordinates are integer `[x,y]`; the tile map uses `[y,x]` indexing and a downward y axis. `scene.json` binds these actual map, position, active, visibility, attribute and event paths. It contains no invented motion, external image, executable markup or policy-only information. Public event channels are also present in observations. Counters and persistent levels are labeled separately from per-step events. Full-state inspection must be distinguished from role-visible playback. A real trajectory and its animation remain to be produced by the trusted implementation process.

## Termination, rewards and conservation

First team reaching three goals ends the match. A tied or unfinished match at clock 160 truncates without inventing a winner. Goal-round kickoffs are explicit in-match transitions, not episode resets.

A nonterminal call advances the int32 clock once. Natural completion takes precedence over the horizon. Terminal raw calls preserve every leaf, observation and clock, return zero rewards and all dones true, and preserve terminated/truncated. No episode reset occurs in raw_step.

Declared finite observation bounds are -1..160; reward bounds are -1..1. These are game data bounds, not outcome thresholds. Rule-specific reward terms and conservation claims above are literal source design claims; they remain untested numerically. `manifest.json` declares every state leaf and safe structural invariants; source-dependent accounting identities require independent execution review.

## Source-line evidence

- Ball, teams, obstacles and kickoff state: `candidate.py:12` — `def reset(self, key):`
- Stamina, possession and forced-wait masks: `candidate.py:41` — `def legal_actions(self, s):`
- Movement and action costs: `candidate.py:72` — `s["agent_pos"] = self.move(state,actions)`
- Contact before launch: `candidate.py:78` — `# Contact steals precede launches; guarding protects the carrier for this tick.`
- Physical flight and active interception: `candidate.py:106` — `# Every tick of travel is interceptable. Bracing extends reach by one cell.`
- Scoring and preserved scoring frame: `candidate.py:127` — `s["goal_event"] = goals`
- Explicit in-match kickoff: `candidate.py:131` — `# Scoring frame is retained. A separate forced-wait tick performs an in-match kickoff.`
- Absorbing match completion: `candidate.py:150` — `s["terminated"] = jnp.any(s["score"] >= 3)`
- Fully observed state layout: `candidate.py:158` — `def observe(self, s):`

## Prospective playthroughs and single-action interventions

`playthrough.json` contains literal source-derived scripts and paired branches that change exactly one role action at one decision. Each script is explicitly `UNEXECUTED_AUTHOR_HYPOTHESIS`; its action dimensions and ranges were checked as data, while runtime legality, actual trajectories, numerical parity and success were not checked. No script result, score, learning gain or browser acceptance is invented. The paired hypotheses identify state paths for later comparison and do not substitute for measured causal traces.

## Community interface and provenance

The supplied official JaxMARL README informed the separation between the homogeneous array kernel and a later agent-keyed parallel API with `__all__`; no upstream environment was copied. Pgx informed explicit legality, raw terminal state and explicit-key provenance, rather than importing a board-game implementation. Craftax supplied a reference for spatial objects, inventories and persistent consequences; this game is an original multiplayer design. PureJaxRL is an algorithm implementation reference, not a game library. The supplied Craftax recurrent PPO baseline credits PureJaxRL; its training/network/wrapper code was not copied, run or substituted for the frozen learner.

Reference snapshots are hash-frozen in `author_input_freeze.json` and the assigned request: [JaxMARL README](https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md), [Pgx README](https://github.com/sotetsuk/pgx/blob/main/README.md), [Craftax README](https://github.com/MichaelTMatthews/Craftax/blob/main/README.md), [PureJaxRL README](https://github.com/luchris429/purejaxrl/blob/main/README.md), and [Craftax recurrent PPO baseline](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py). API/design ideas only; copied upstream code: none.

The public JaxMARL adapter belongs to the trusted integration layer: it should map fixed ids `agent_0` through `agent_3` to the existing array boundary, expose masks and set done `__all__` from `done_all`. No agent dictionary adapter or new learner is smuggled into this restricted one-class kernel. Source integer categories use int32, flags use bool, and the supplied uint32 reset key is external provenance rather than a hidden mutable state object.

The source/manifest/scene hashes are bound in `generation_record.json`; `author_input_freeze.json` binds the verified frozen author inputs. No prior candidate, result, replay, rank, training feedback or final-test material was read. No numeric candidate execution, CSD3 operation, model download or background task was performed. Initial-window accounting remains four parentless proposals 150..153, 154 used or reserved of the existing 512 cap, with later windows unchanged. Authoring and visualization support alone do not establish learnability, non-isomorphism, generalization or meta-learning.
