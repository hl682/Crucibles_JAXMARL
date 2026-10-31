# Whisper Corridors

Status: AUTHORED_NOT_EXECUTED. A=4, O=145, U=9, H=80. Roles 0/1 are the scout-and-courier team; roles 2/3 are the scout-and-interceptor team. The kernel and rule structure are original.

## Competing routes and asymmetric roles

A public 9 by 7 map has a dividing wall column at x=4, with gate sites at y=1,3,5. Two public delivery exits lie at (8,1) and (8,5). The courier begins at (0,3), carrying one actual parcel, while the interceptor starts at (8,3). Scouts start at (2,1) and (6,5). Reset draws one uniformly selected closed gate from the supplied key; the other two start open. The rest of reset is fixed and steps contain no random draws.

Scouts have radar-like Manhattan sensing radius three; main roles have radius one. Sensing does not use optical wall occlusion. Only actual locally sensed agents and gate states are accessible. Both scouts can traverse a closed gate using a maintenance pass, but still cannot cross wall cells. Courier and interceptor require an open gate. They block one another's destinations, including attempted swaps or moves into the same cell; scouts may collocate and do not physically block main roles. The separate top and bottom exits make defending one location a route commitment rather than an automatic guard of the only goal.

A scout adjacent to a ready gate may spend one of three switch-energy units to toggle it. Waiting recovers one unit. A used switch has a four-state cooldown. Movement uses the input gate configuration; toggles affect subsequent movement. If both scouts operate the same ready gate on a tick, two toggles cancel in the gate boolean, but both spend energy and the cooldown is set. This gives conflict a real future consequence without an artificial communication reward.

## Private notebooks, explicit relay and interception

Each scout records the opposing main role's last locally seen position and the last observed status of each gate. Ages reset only on a fresh local observation. An unrefreshed private notebook entry becomes unknown at age nine. Neither main role receives its scout's private notebook automatically.

Action 5 sends the scout's current notebook as a snapshot. Its teammate receives it regardless of distance. Every role within Manhattan distance two of the transmitter at that moment also receives that same message, including opponents. Reception is fixed at transmission time; entering that radius later does not retroactively reveal an old message. A fresh report lives for four states, and its ages continue to increase while it is available. Origin coordinates are the actual transmission location, not the sender's current position after subsequent movement. Age nine means unknown or saturated age; an already transmitted snapshot may retain old coordinates until its message expires, explicitly accompanied by that age.

This creates a causal information tradeoff. Relaying lets a distant courier choose a route based on a partner's observations, while speaking near the interception team reveals those same observations. A scout may wait, move away before reporting, or act on a gate before informing a partner; those delays can make an opponent position stale. The opposing scout can scout the courier and relay to the interceptor, or approach a transmitter to overhear. None of these possibilities has been measured as an effective strategy.

## Public lures and truthful signals

Each team owns two single-use public unverified signal charges. Either member may emit a five-state marker at their actual current coordinate; a new marker replaces that team's previous marker. For the courier team it claims courier-like activity; for the other team it claims interceptor-like activity. Because either the scout or main role can emit it, the marker is not evidence that the named main role is there. If both teammates emit on one tick, the lower role id supplies the position and only one charge is spent. The actual emitter id is not a marker attribute. The marker cannot teleport an agent, fabricate a true sensor contact or itself produce reward.

An opponent may investigate a signal, ignore it, compare it with a fresh scout report, or use its age to infer whether a route has been abandoned. The scene and observation explicitly label this channel as unverified. True local contacts, private cached facts, accessible radio snapshots and public claims occupy separate observation slices and entity groups.

## Payoffs, timing and natural termination

Each step attempts movement, applies physical switches, refreshes sensors/notebooks, ages or transmits reports, ages or emits public lures, then checks interception and delivery. Interceptor action 7 captures if its final position is within Manhattan distance one of the courier. Courier action 7 delivers only at either exit and only if capture did not trigger on the same tick. Capture has explicit priority. Delivery gives roles 0/1 +1 and roles 2/3 -1; capture reverses those payoffs. Every other step pays zero. The finite 80-step horizon yields zero-payoff unresolved truncation. Reports, signals and gate switches have no reward by themselves.

A prospective courier strategy sends its scout ahead through a closed maintenance gate, receives current defender/gate information, and chooses a different exit from the defender's commitment. Another opens a direct corridor and relies on timing rather than a long detour, accepting the risk of report interception. The defensive team may keep its scout near the courier or instead eavesdrop and send the interceptor to a likely exit. Changing a partner's willingness to relay, gate operation or movement changes the useful response through actual information and access. These are hypotheses for later measured paired interventions, not evidence of learned coordination or deception.

## Observation, scene and invariants

The 145-column observation includes own role/position/energy; locally visible role coordinates and visibility bits; locally visible gate states/cooldowns; local parcel position/status; own scout-only notebook; all messages accessible through team delivery or interception; public unverified signals and stock; clock, both exits, terminal outcome and public topology. Unknown information is encoded as -1, with explicit visibility bits. The fixed topology's gate-site tile never exposes its hidden open value. The legal mask is uniform over all action attempts before termination and therefore cannot leak hidden gate or opponent information.

Agent, gate and parcel entity visibility bind actual local sensor matrices. Report entities bind actual report_listeners and positive TTL, including genuine eavesdroppers. Lures and exit sites are public. Scene attributes never bind another role's private notebook or energy. The raw relay_event/toggle_event leaves are audit-only state fields, not declared public scene events; exposing them globally would leak private activity. Only public lure emission and terminal outcome are declared event channels. A debug/full-state renderer must be labelled separately from role views.

Declared bounds include energy 0..3, switch cooldown 0..4, notebook/report ages 0..9, report TTL 0..4, lure TTL 0..5, stock 0..2 and outcome 0..2. Objects and agents remain within world bounds. All action ids are legal attempts; non-scout relay/toggle, scout delivery/capture, unavailable energy or inappropriate locations produce no corresponding skill effect. Invalid ids become wait. True terminal states and causes are absorbing, with zero reward on later raw calls.

## Prospective variation

Sensor radii, gate topology, interception radius, report lifetime, listening radius and lure stock would change the value of reconnaissance and communication. Such variations require new source identities and a frozen protocol. This single authored environment does not establish learned information use, optimal play, transfer or scalable meta-RL.

## Literal source-line evidence

- candidate.py:15: radius = jnp.array([3, 1, 3, 1]. Scouts and main roles have different actual local sensing radii.
- candidate.py:22: age = jnp.where(seen. Information age depends on new local sightings rather than a fabricated message reward.
- candidate.py:106: notes = jnp.full. Only scout rows receive the corresponding private notebook.
- candidate.py:140: passable =. Scouts have a maintenance-pass capability while main-role movement depends on actual gate state.
- candidate.py:152: gate_open = jnp.where(toggles % 2 == 1. Physical switch actions change corridor access, with simultaneous opposite toggles cancelling.
- candidate.py:163: broadcast = a[scouts] == 5. Cached information becomes a message only after an explicit scout action.
- candidate.py:173: listeners = jnp.where. Team delivery and nearby interception determine actual per-role report access.
- candidate.py:174: report_visible = listeners. Expired reports cease being visible even to previous listeners.
- candidate.py:180: emitters = (a == 8). Either teammate can create a finite public claim marker at their real position.
- candidate.py:187: caught =. Interception has a physical proximity requirement and explicit action.
- candidate.py:188: delivered =. Delivery requires courier presence at one of two actual exit sites.
- candidate.py:215: advantage =. Only actual terminal outcomes create team-opposed rewards.

## Public reference provenance

- references\JaxMARL_README.md: SHA-256 6649d04f5ed05d620e78e8416a00d4db841cf2633ae60fdb3c9729e037a71c15; https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md. API/design reference only; no upstream implementation copied.
- references\Pgx_README.md: SHA-256 153f44999e55d5df3e51e13dee6ef418cc9704ddfa26206f1dc5789b9cbbf8df; https://github.com/sotetsuk/pgx/blob/main/README.md. API/design reference only; no upstream implementation copied.
- references\PureJaxRL_README.md: SHA-256 d5f40182c924a42cec820931e61a7d5070e48c4c8413c7d16455abc6ea0ef6b0; https://github.com/luchris429/purejaxrl/blob/main/README.md. API/design reference only; no upstream implementation copied.
- references\Craftax_Baselines_ppo_rnn.py: SHA-256 5f93e6a7c3a1e0c551831d75be311735ba495dfdecae51787a1c03a27b1cd011; https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py. API/design reference only; no upstream implementation copied.
- references\Craftax_README.md: SHA-256 7f9757a2047143cf183f7e2e84851a740f386af80dbee050c89b316d66a3d3ff; https://github.com/MichaelTMatthews/Craftax/blob/main/README.md. API/design reference only; no upstream implementation copied.

JaxMARL supplies the parallel agent-keyed API idea; Pgx supplies the distinction between game state, legal masks and terminal flags. Craftax supplies resource-world design context. PureJaxRL and Craftax_Baselines are algorithm-integration references, not substitutes for the frozen learner. No upstream implementation was copied. Source author self-report is gpt-6-astra/max; the coordinator retains dispatch evidence.

## Status and limits

AUTHORED_NOT_EXECUTED. Only stdlib AST/JSON/hashes and the supplied static code, manifest and scene validators were used. Candidate code was never imported or executed; no JAX/NumPy, numerical emulation, model API, network, SSH, training or replay ran. Playthroughs and paired interventions are UNEXECUTED_AUTHOR_HYPOTHESIS. Static acceptance establishes neither numerical correctness nor solvability, interaction quality, learning, meta-learning, transfer, browser acceptance or execution authorization. Runtime R0/M0, adapter integration, actual per-role replay and scientific evaluation remain separate. Integer quantities use int32, boolean flags use bool, observations/rewards float32. The external reset key is not duplicated in state. Already-terminal raw inputs preserve all state, clock and terminal causes, with zero reward; natural completion has priority over horizon truncation.
