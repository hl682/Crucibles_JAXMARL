# Magnet Frontier

AUTHORED_NOT_EXECUTED. Current initial attempt 00. Requested author gpt-6-astra / max; provider measured model, effort, tokens, elapsed time and cost are unknown.

This original 24x24 block frontier has a long broken-floor seam, a short starting freight rail and connected perimeter routes. Reset uses only the supplied key to vary resource stocks and three remote ore-node x coordinates. Six fixed reusable tool objects travel with their crews and determine specialist action masks. Resources live in bounded per-crew inventories and spatial resource nodes; transferring a selected unit requires an adjacent receiver with room.

Two construction chains are central: two ore make a coil, then coil plus timber makes a movable/dismantlable magnet mount; two timber make a nonmagnetic brace, which can bridge a gap, immobilize cargo or recover a jam. A third chain turns ore plus timber into a power pack. All mounts consume a common finite charge budget. Field vectors sum before a dominant-axis cargo move, so activating or reversing one mount affects other crews' loads, including canceling movement. Rails, bridges, optional manual lifting and persistent anchors provide different timing/resource responses. Cargo collisions create a visible jam and lose integrity; a brace repairs it. Pickup releases an anchor. Mounts can be dismantled for their coil, losing their timber investment.

Optional type-0/type-1 deliveries pay different values at different physical sites. The same fixed freight bodies deactivate for a cooldown and then consume ore stock at a mine to reappear; this is a renewable production cycle, not a reward reset. Ore slowly replenishes and rainy timber/clear-weather solar cycles alter future supply. Exploring a node region records a per-crew first visit. Achievement and infrastructure choices can coexist throughout the bounded interaction window.

## Native roles, controls and observations

A/O/U/H = (6, 938, 28, 384). U=28 includes four independent aim commands, resource selection and adjacent transfer, physical construction, power/polarity and recovery. Roles are complementary capabilities rather than duplicated independent actors. Human and AI controllers use exactly the same scalar int32 joint-action interface and native state transition. The shared platform can assign any subset of roles to humans. No separate browser dynamics are provided.

Fully observable native world: every role sees the same flattened map, crews and inventories, reusable tools, node resources, magnet settings/actual supply/radius, cargo holders/jams/anchors/integrity, facilities, shared charge/weather and reward-component state. Inactive capacity slots remain explicit in observations and are not rendered as active objects.

0: Wait.
1: Walk north and face north.
2: Walk east and face east.
3: Walk south and face south.
4: Walk west and face west.
5: Gather ore with miner kit or timber with saw kit beside matching node.
6: Lift lowest-index adjacent loose cargo; miner or hauler only.
7: Put held cargo at current tile.
8: Craft a coil from two ore at workbench; engineer.
9: Craft a brace from two timber at workbench; builder.
10: Craft a power pack from one ore and one timber at workbench; electrician.
11: Build a magnet on front tile using one coil and one timber; engineer.
12: Lay front rail using one ore and one timber; engineer.
13: Bridge a front gap with one brace; builder.
14: Toggle nearest adjacent mounted magnet.
15: Reverse nearest adjacent magnet polarity.
16: Dismantle front magnet and recover one coil; engineer or builder.
17: Load one power pack into bus at workbench for eight charge; electrician.
18: Deliver held cargo at its matching destination; starts its mine replenishment cooldown.
19: Select next carried resource: ore, timber, coil, brace, pack.
20: Give one selected resource to lowest-index adjacent peer with room.
21: Repair or unjam adjacent cargo with a brace; hauler or builder.
22: Hand-crank one charge at the workbench.
23: Anchor adjacent loose cargo with a nonmagnetic brace; builder.
24: Aim north without moving.
25: Aim east without moving.
26: Aim south without moving.
27: Aim west without moving.

## Resolution, reward and time

Out-of-range or mask-false commands become wait. Movement is simultaneous: duplicate targets and destinations occupied at tick start block the involved moves; swaps wait. Interactions rotate priority by (t+k) mod 6. Eligible pickup/transfer/recovery ties use lowest native index, while magnet toggles use nearest eligible mount. Cargo field moves reject occupied or contested cargo destinations and out-of-map destinations; jamming never changes array shape. Crews and freight may share a tile while handling loads.

Native components per crew are: 0.02 per first visit to a resource region (at most 8), 2 or 4 per own matching delivery, 0.15 per successful construction by any crew, -0.05 per consumed crafting/build/repair/charge material unit plus -0.01 per shared bus unit used, and -0.25 per newly jammed cargo last lifted by that crew. Each step admits at most one own action, at most six construction events, eight powered mounts and eight new cargo jams. These conservative event counts bound the returned reward within [-3,6]; the actual legal role masks tighten several counts. All five components are native float32 state.

No natural success/failure terminal is imposed in this persistent world. Native terminated remains false; tick 384 is the truncation window. Delivering freight or building a route never resets the world. Natural causes take priority over H, and already-terminal input is absorbing: unchanged state/observation/clock, zero reward and preserved terminal causes. Raw transitions never reset. Inventories are capped at six per resource; the sixth column is an explicit reserved zero. Eight ore/timber nodes cap at twelve units, eight magnet capacities and eight freight identities are fixed. Mine regeneration, solar input, contract replenishment and material sinks are explicit. Built cells and mount state persist until changed. The world is finite and does not persist across process resets.

## Literal source evidence

candidate.py:L40: variation=jax.random.randint(key,(8,),0,3,dtype=jnp.int32)

candidate.py:L91: coil=(a==8)&(inv[0]>=2)&(inv[2]<6)

candidate.py:L92: brace=(a==9)&(inv[1]>=2)&(inv[3]<6)

candidate.py:L73: newbuild=jnp.array(0,jnp.int32)
candidate.py:L98: build=(a==11)&inside&(tile!=1)&~occupied&jnp.any(free)&(inv[2]>0)&(inv[1]>0)
candidate.py:L107: newbuild=newbuild+(build|rail|bridge).astype(jnp.int32)

candidate.py:L100: bridge=(a==13)&inside&(tile==1)&(inv[3]>0)

candidate.py:L129: give=(a==20)&jnp.any(peers)&(inv[res]>0)&(n['inventory'][to,res]<6)

candidate.py:L133: anchor=(a==23)&jnp.any(loose)&(inv[3]>0)&~n['anchored'][c]

candidate.py:L143: powered=demand&(n['charge']>=draw);used=jnp.where(n['charge']>=draw,draw,0)

candidate.py:L148: force=jnp.sum(jnp.sign(diff)*n['polarity'][None,:,None]*reach[:,:,None],axis=1,dtype=jnp.int32)

candidate.py:L157: jammed=movable&hit

candidate.py:L167: node=c%4;spawn=~n['cargo_active'][c]&(n['cooldown'][c]==0)&(n['stock'][node]>0)

candidate.py:L173: n['reward_parts']=n['reward_parts'].at[:,2].set(0.15*newbuild.astype(jnp.float32)).at[:,3].add(-0.01*used.astype(jnp.float32))

candidate.py:L179: n['terminated']=jnp.array(False);n['truncated']=n['t']>=self.max_steps;n['done']=n['truncated']

## Presentation and evidence boundary

Original muted stone/grass blocks, brass tool kits, violet field mounts and iron freight. Scene tiles are the actual floor/gap/bridge/rail state; powered flags and polarity expose causes. All scene geometry, entity attributes and visibility refer to declared native leaves. Public scene events are restricted to actually public information. Debug/full-state view must be separate from role view.

Two literal scripts and a single actor/action paired hypothesis are supplied, with fixed integer seed and source-derived suffix lengths. All remain UNEXECUTED_AUTHOR_HYPOTHESIS; neither legality of the full trajectory nor any effect, solvability, strategic richness, learning or transfer is certified. Prospective variation could examine resource timing, tool assignments, field geometry or collaborator power/routing habits under separately frozen future source. No such variation or candidate has been run.

Current author checks use stdlib AST/literal shape arithmetic, byte hashes and the supplied trusted manifest/scene/whitelist validators only. No candidate/JAX/NumPy import, native method call, numerical rollout, rendering, training, network/model call or seed search occurred. Four official README excerpts supplied API context only: JaxMARL parallel dictionaries, Pgx explicit-key and terminal interfaces, Craftax spatial game framing, PureJaxRL single-file implementation philosophy. No upstream implementation, layout or asset was copied. The supplied PPO implementation was not read. Exact current input paths, bytes, hashes and excerpt extents are recorded in author_input_freeze.json. Current diversity addendum controls A/H and the scope of open worlds.
