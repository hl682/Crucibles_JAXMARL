# Ballast Run — source package game-v3-00076 / attempt_00

Status: **AUTHORED_NOT_EXECUTED**. This package authors the complete 3-agent, 1939-column, 14-action, H192 design. It has not been imported, simulated, rendered or trained. It is one source attempt for design 76, not a new design proposal.

Source request SHA-256: 18f3e7720bd8ef5563c5fa5a838c0ccb461673f034c90c4d25167eaf48549559  
Design JSON SHA-256: 5bab379abd034a1ad5fb35f182596ee77e1a2e0c3a9a5e1a61228c80f531acd5  
candidate.py SHA-256: 3aefd945deb38d64614ec4dfaf55ddc131d192b81e6b155f2814bb88f56d37ea

## Rules and causal mechanism

The 23×15 river and fixed-orientation 5×3 vessel are actual transition geometry. Three crew move among a helm, upper scoop, lower dispatch station and four corner tanks. Reef/boundary collision checks the complete hull footprint, including both orthogonal intermediate footprints on diagonal travel.

Persistent velocity has integer components in [-1,1]. Thrust consumes one fuel and adds its unit vector; a periodic current adds a public horizontal impulse using the old boat centre and old tick. Brake consumes one fuel, forces zero velocity and suppresses that impulse. Waiting retains momentum. Initial fuel is 80 and hull condition 12.

Crew mass is 1, each ballast unit mass 1 and aboard capsule mass 2. Torque is the integer sum of mass times offset from deck centre [2,1]. Loose capsules propose one simultaneous cell of gravity motion on each axis at |tx|>=4 and |ty|>=3. They do not repeatedly slide to equilibrium. Pinned/blocked capsules remain; capsule positions remain distinct. Leaving the deck loses the real capsule. Post-slide |tx|>=10 or |ty|>=6 causes one additional hull damage.

Two reusable clips prevent sliding while retaining cargo mass. Releases precede attach requests. If all requested new clips do not fit, every new attach request fails. Walking crew, altered ballast, newly loaded samples and sliding cargo all contribute to future torque.

The three public source buoys are [7,2], [18,3], [18,12]. Loading uses actual scoop proximity, stationary post-motion velocity, an empty designated tray and safe hypothetical post-load torque. Banking requires the exact [3,12] berth, stationary velocity and safe torque; all aboard capsules then become permanently banked and free their clips.

## Actions and simultaneous resolution

Actions 0..13 are wait, N/E/S/W walk, N/E/S/W thrust, brake, fill tank, drain tank, station work, toggle clip. All crew share the vocabulary.

Walking uses a simultaneous intent graph: same-target requests all fail, blocked dependencies propagate for three fixed passes, and surviving cycles/swaps are allowed. Cargo sliding uses an independent three-pass graph and does not treat crew as barriers.

The active transition order is crew movement; ballast/clip operations; helm/current/inertia; full-footprint navigation; station work; one cargo-slide pass; post-slide strain; end-state selection. Event order records these physical stages. Event payloads marked force or collision attempt are not accepted object trajectories.

Masks enable context-appropriate attempts. Shared preconditions such as a partner braking this same transition, another worker releasing a clip, or a simultaneous swap are resolved by raw_step. The mask does not promise an effect.

## Observations and access

Every physical variable is public. observe returns float32[3,1939] with terrain 1725, deck 165, capsule rows 21, buoy coordinates 6 and global/self 22 columns. Each exact slice is declared in manifest.json. No O16 projection or action/horizon reduction was introduced.

The deck channels are the three crew occupancies, tank fill, capsule mass/2, loose and pinned counts, three station flags and tank-site flag. Global fields include true velocity, fuel/hull, clock, tide, next-pulse indicator, torque, remaining clips and end state. Animation does not add policy observations.

## Reward, endings and invariants

Each crew receives newly banked capsules + 2 for a newly selected natural success - 1 for natural failure - 0.02 per hull unit lost - 0.001 per active transition.

End priority is hull exhausted, capsule lost, all capsules banked, then H192. Natural outcomes take priority at H. raw_step retains an ended input state and observation exactly, returns zero reward and preserves its cause flags. It never resets internally.

Obligations include distinct crew cells and distinct aboard capsule cells; navigable complete hull footprint; at most two pins; pin implies aboard; no resurrection/repeated banking; bounded tank/fuel/hull values; t increments exactly once for nonterminal input; and mutually exclusive terminated/truncated. These remain untested source obligations. The manifest's machine-readable invariants use only the supplied unambiguous nonnegative/unique-position forms; the remaining full obligations are retained here and in declared_invariants.

## Actual scene and event bindings

manifest.json declares 60 actual state leaves. Entity groups bind crew world coordinates, vessel centre/velocity/torque, moving corner tanks, moving station cells, capsule source/aboard/banked positions, source buoys and berth. Active/visibility arrays are concrete bool state leaves.

The world descriptor carries the 15 actual hull offsets and a deck coordinate frame. The scene presents a recognizable river board and deck detail, with real tank fill, pins, cargo, current arrows and hull damage. Generic allowed glyphs are used rather than arbitrary SVG/HTML. A later renderer may style them but must preserve these state bindings.

event_kind plus its active/tick/actor/object/from/to/space/value arrays bind all 14 nonzero event kinds. Event identity is candidate/tick/slot; repeated absorbing snapshots cannot replay them. Each visual lifetime is bounded by one recorded transition interval. No invented path, idle drift or successful ending is supplied.

## Interface adaptations and implementation evidence

- Numeric state is int32/bool throughout; observations and rewards are float32. All declared shapes are static.
- Deck xy coordinates are converted losslessly to world xy using boat_xy + deck_xy - [2,1]; tile indexing remains yx.
- Scalar vessel data is reshaped into length-one entity attributes. Presentation leaves are deterministic projections of game state, not extra random gameplay.
- The design's fixed berth display slots are materialized at home_xy + [-1,1], [0,1], [1,1]. This fixes an unspecified presentation offset only; no navigation, reward or interaction changes.
- The deterministic reset ignores its supplied key without fabricating RNG state. Any actual trajectory key/seed belongs to the trusted outer receipt.
- The active helper writes new t early only to stamp event identities. Current phase explicitly reads the old input t. The raw interface still advances once or absorbs.
- The raw terminal policy is the binding v2 addendum, not the superseded v1 horizon sentence.

- `candidate.py:121` — `def reset(self, key):`: constructs the declared map, initial positions, resource levels and source capsules
- `candidate.py:53` — `def resolveMoves(self, positions, targets, moving, valid, active):`: implements simultaneous conflicts, dependencies and cycles
- `candidate.py:64` — `def torque(self, state):`: computes crew, ballast and cargo mass moments
- `candidate.py:73` — `def footprint(self, terrain, centre):`: checks each actual hull cell against terrain
- `candidate.py:215` — `def activeStep(self, state, actions):`: implements all station, inertia, clip, gravity, damage and natural-ending rules
- `candidate.py:172` — `def observe(self, state):`: assembles all 1939 columns
- `candidate.py:80` — `def present(self, state):`: binds real world positions, active masks and public visibility
- `candidate.py:334` — `def raw_step(self, key, state, actions):`: selects the absorbing old state and returns zero terminal-call rewards

## Unexecuted playthrough and remaining limits

playthrough.json is a fixed, seed-declared hypothesis covering crew travel, thrust/brake and paired tank changes from the exact reset state. It does not claim a sampled rollout, capsule delivery, solution, learning result or verified legality. It does not change reset.

Only stdlib AST text parsing, document construction and hashes were used. No source validator was called and no PASS is claimed. JAX import/execution, eager/JIT/vmap comparisons, invariant tests, masking checks, runtime/storage feasibility and real animation remain unperformed. The 1939-column policy requires matching new policy metadata/weights; old O16/U5 weights are inapplicable. A 3-agent game also does not establish support for the older two-role payoff/rank path.

Source execution 0; environment calls 0; rollout steps 0; training updates 0. backend_revision and token_usage remain null.
