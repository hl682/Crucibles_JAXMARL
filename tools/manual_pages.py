"""Bilingual per-game manuals generated from source and authored descriptions."""
from manual_content import ROOT,SRC,THEMES,read_json,action_help
GH='https://github.com/hl682/Crucibles_JAXMARL'
def choose(en,zh,lang):return zh if lang=='zh' else en
def example(key):return f'''import jax
from crucibles_jaxmarl import make

env = make("{key}")
key = jax.random.PRNGKey(0)
key, rk, ak, sk = jax.random.split(key, 4)
obs, state = jax.jit(env.reset)(rk)
actions = env.sample_actions(ak, state)
obs, state, rewards, dones, info = jax.jit(env.step_env)(sk, state, actions)
done = dones["__all__"]'''

def manual(row,p,m,sc,lang):
    key=row['key'];dim=row['dimensions'];en=lang=='en';loc=p[lang];t=lambda a,b:choose(a,b,lang);sections=[]
    def add(title,*paras):sections.append('## '+title+'\n\n'+'\n\n'.join(paras))
    intro='# '+key.upper()+' · '+loc['title']+'\n\n'+loc['brief']
    add(t('Identity and original contract','身份与原始接口'),t(
        f"{dim['num_agents']} independently actionable roles; {dim['num_actions']} discrete commands per role; {dim['observation_size']} native observation columns per role; original horizon {dim['max_steps']} steps. Declared per-step reward range: {m['reward_low']} to {m['reward_high']}. Observation regime: `{m['observability']}`.",
        f"共有 {dim['num_agents']} 个独立可操作角色；每人 {dim['num_actions']} 个离散动作；每人的原生观察宽度 {dim['observation_size']}；原始时限 {dim['max_steps']} 步。声明的单步奖励范围为 {m['reward_low']} 至 {m['reward_high']}。观察制度标识：`{m['observability']}`。"),
        t('Source SHA-256:','源码校验值：')+' `'+row['source_sha256']+'`',
        t('The objective, rewards, population, map and horizon remain original. A preview is not a success claim.','目标、奖励、人口、地图和时限均保留原值。动图不表示已完成任务。'))
    gameplay=[loc['brief']]
    if en:
        if m.get('task',{}).get('interaction'):gameplay.append(m['task']['interaction'])
        gameplay.extend('- '+r for r in sc.get('rules',[]))
        if m.get('task',{}).get('natural_termination'):gameplay.append('Ending: '+m['task']['natural_termination'])
        if m.get('action_semantics'):gameplay.append('Action timing: '+m['action_semantics'])
    else:
        gameplay.append('围绕上述目标安排每位角色的合法动作，以实际资源、位置、工序和信息为约束。提交的是同一步的联合行动，不是把角色按界面顺序逐个推进。动作编号与按钮名保持游戏中的英文，见下方逐项操作表。')
        gameplay.append(f"自然完成、失败和截断由原始内核决定；最长运行 {dim['max_steps']} 步，不能把到达时限当作成功。先看结束标记，再检查目标、损失和实际回报。空间游戏保留原始吸收终止；抽象游戏保留历史终止行为。")
    if key=='gi134':gameplay.append(t('First complete six auction steps (input t=0–5). Physical movement starts at t=6; the keyboard cannot bypass this phase.','先完成六步拍卖，输入时点零至五只允许合法竞价或等待；第六步之后进入实体移动，键盘不能越过阶段。'))
    if key=='gi196':gameplay.append(t('Advance the council phase before expecting march controls; actions depend on the current phase.','先推进议会阶段，再使用行军指令；议会、行军、解算按原始阶段轮转。'))
    add(t('Objective and gameplay','目标与玩法'),*gameplay)
    add(t('What this game explores','这款游戏体现的理念'),*[THEMES[v['code']][0 if en else 1]+': '+THEMES[v['code']][3 if en else 4] for v in p['themes']],t('These are questions supported by rule structure, not observed learning effects.','这些是规则结构支持的研究问题，不是已观察到的学习效果。'))
    add(t('Play it: operating walkthrough','实际游玩操作指南'),
        '```sh\ncrucibles play --game '+key+'\n```',
        t('Keep Python running and open its printed address. Select this game, set a seed and press `Create session`. The game itself is always English.','保持终端运行并打开打印的网址，选择本款、设种子，点击 `Create session` 建局。游戏本体始终为英文。'),
        t('Assign a role range to Host with `Assign / take over`, or `Assign all roles`. Invite a second participant with `Create player invitation`, then assign a disjoint subset. Every original role is available.','用 `Assign / take over` 把角色编号或范围交给主持人，或用 `Assign all roles` 接管全员。用 `Create player invitation` 邀请另一参与者，再分配不重叠角色；全部原角色均可操作。'),
        t('For every owned role choose a legal command and press `Queue`. Arrow keys / WASD queue a declared directional action for the current role camera when available. Space or `Advance one joint step` advances everyone together. Non-directional games use the selectors.','为每个接管角色选合法指令，点击 `Queue` 排队。方向键或字母方向键在本款存在对应动作时，为当前角色视角排队；空格或 `Advance one joint step` 同时推进全员。无方向动作时使用下拉框。'),
        t('Unqueued roles use the labelled untrained uniform-legal policy. Take over or release roles before the next step without resetting. Full-state debug is privileged and permanently logged; role cameras preserve permissions. Close the session to retain logs. Server restart does not restore sessions.','未排队角色由标注的未经训练、合法动作均匀采样策略操作。下一步前可接管或释放，不重置世界。完整状态调试视角暴露特权信息并永久记录；角色视角保留权限。关闭会话可保留日志，服务重启不恢复会话。'))
    add(t('Complete command reference','全部动作操作表'),t('IDs and English labels exactly match the manifest. Availability depends on role, resources, location and phase. An available action can still have no productive effect.','编号和英文按钮逐项对应清单。中文提示说明操作类别；具体目标、阈值、阶段和顺序按本款规则与清单执行。合法不保证有产出，前提不足可能没有效果。'),*['- `'+str(a['id'])+'` · `'+a['label']+'`'+(' — '+action_help(a['label']) if not en else '') for a in m['actions']])
    add(t('Roles and capability identifiers','角色与能力标识'),t('These original role names describe capabilities, not interchangeable skins. Check each owned role\'s legal selector before planning handoffs.','以下保留英文游戏能力标识，便于在界面找到角色，并非可任意互换的皮肤。协调交接前先查看对应角色的合法动作。'),*['- `'+a['id']+'` · `'+a['role']+'`' for a in m['agents']])
    access=m.get('task',{}).get('observation_access')
    add(t('Information and observation permissions','信息与观察权限'),access if en and access else t('Each actor receives its original native observation row. Public state, private fields and delayed messages remain separate.','每个行动者仅接收自己的原生观察行；公共状态、私人字段和延迟消息须分别处理，不能从全图推断角色已经知道。'),t('The GIF is full-state debug and may reveal hidden entities or private state. Train on `obs[agent]`, not the picture, state tree or another role\'s private row. Field slices and visibility are in the original manifest.','动图是完整状态调试视角，可能揭示隐藏实体或私人状态。策略输入使用 `obs[agent]`，不能直接使用图片、完整状态树或伙伴私人观察。字段切片和可见性见原始清单。'))
    add(t('Reinforcement-learning integration','强化学习接入'),'```python\n'+example(key)+'\n```',
        t(f"Observations are float32 vectors of width {dim['observation_size']}; actions are scalar int32; masks have {dim['num_actions']} entries. Supply all {dim['num_agents']} roles each transition. Masked sampling is an engineering reference, not a trained learner.",f"观察为宽度 {dim['observation_size']} 的浮点向量；动作是单个整型编号；每个合法掩码有 {dim['num_actions']} 项。每次转移提供全部 {dim['num_agents']} 人动作。掩码采样只是工程参照，不是训练后的学习器。"),
        t('`step_env` retains terminal data without reset. `step` performs outer autoreset and keeps terminal data in `info["raw_state"]` and `info["raw_observation"]`. Use `dones["__all__"]` for episode boundaries; separate completion, failure and time limit in analysis.','`step_env` 保留终止数据，不自动重置；`step` 提供外层自动重置，在信息字典保留原始终止状态与观察。用 `dones["__all__"]` 划分回合，分析时区分完成、失败和时限。'),
        t('JIT or vmap equal identities and shapes. Split independent reset, policy and transition keys. Group different shapes explicitly; do not shrink the population or horizon. Host rendering stays outside training.','对同身份同形状编译或批处理，重置、策略和转移使用独立密钥。不同形状明确分组，不能缩小人口或时限。主机渲染留在训练图外。'))
    add(t('Training questions and fair evaluation','训练问题与公平评测'),
        t('Define valid completion from the objective before fitting. Preserve original rewards and also report real deliveries/services, losses, resource state, unresolved episodes and per-role returns. Animation is not a reward signal.','拟合前按本款目标定义有效完成指标。保留原奖励，另报真实交付或服务、损失、资源状态、未决局和逐角色回报。动画不是奖励信号。'),
        *[THEMES[v['code']][3 if en else 4] for v in p['themes']],
        t('Compare memory, communication or centralized critics under matched actor information. Declare critic access; keep actor inputs role-filtered. Match partner/opponent distributions, masks, compute and checkpoint selection. Final tests never enter fitting or selection.','比较记忆、通信或集中价值网络时须匹配策略信息，声明价值网络权限并保持角色输入过滤。伙伴或对手分布、掩码、算力和检查点选择规则须匹配，最终测试不能参与拟合或选择。'),
        t('Independent training runs or source-pairing blocks are inferential units; episodes from one policy are not independent training replications. Save hashes, keys, versions, device, time and cost. Negative, zero, failed and indeterminate outcomes are valid. No trained checkpoint or target score is supplied.','独立训练运行或来源配对块才是推断单位，同一策略的多局不是独立训练复现。保存哈希、种子、版本、设备、耗时和成本。负向、零、失败和无法判定都有效，不提供训练检查点或承诺分数。'))
    labels=[]
    for v in p['themes']:
        labels.append('- '+THEMES[v['code']][0 if en else 1])
        for ev in v['evidence']:
            labels.append('- '+('Evidence `'+ev['field']+'`: '+ev['text'] if en else '依据字段：`'+ev['field']+'`。该字段规则支持此阅读标签，可在下方原始资料逐字核对。'))
    add(t('Classification and evidence','分类与判断依据'),t('Overlapping editorial labels are reading aids, not proof of novelty, difficulty, strategic equivalence or learning effects.','可重叠编辑标签是阅读辅助，不证明新颖性、难度、策略等价性或学习效果。'),*labels)
    source=GH+'/tree/main/src/crucibles_jaxmarl/games/'+key
    add(t('Source, receipts and preview limits','来源、回执与预览边界'),t('The additional generated state GIF replays saved untrained legal actions through the original kernel. At most 25 actual endpoints include initial and terminal state; no invented interpolation. It is full-state debug, not the role observation. Where an original scene preview exists, it is displayed first with its own recorded information scope. The two recordings may use different trajectories and views.','补充生成的状态动图把保存的未训练合法动作交给原内核重放，最多抽取二十五个真实端点，含初始与终止，无虚构插值。它是完整状态调试，不是角色观察。已有原版场景预览优先显示，并标注自身记录的信息范围。两份录像的轨迹与视角可能不同。'),
        '['+t('Original source, observation slices, manifest and mechanism','原始源码、观察切片、清单和机制')+']('+source+')',
        '['+t('Native receipt','原生回执')+']('+GH+'/blob/main/verification/'+key+'.json) · ['+t('Presentation receipt','展示回执')+']('+GH+'/blob/main/verification/presentation/'+key+'.json)',
        t('Engineering passes do not establish learning. Nonzero floating differences remain recorded and undetermined; no new tolerance was introduced. Human scores remain unverified.','工程通过不证明学习。非零浮点差异保留并记为未定，不增加容差。人类成绩尚未核验。'))
    if key in ['gi226','gi227','gi229']:
        add(t('Complete original saved replays','完整原始保存回放'),'['+t('Download original endpoints and standalone players','下载完整端点和独立播放器')+']('+GH+'/releases/download/v0.1.0/crucibles-original-saved-replays-v0.1.0.zip)',t('All eleven GI226/227 tracks and six GI229 tracks retain failures, zero effects and numerical uncertainty. Older instructional panels are archival content.','整包保留大型运河与森林十一条轨迹及厨房六条轨迹，包括失败、零效应和数值不确定性。旧说明面板作为档案内容保留。'))
    return intro+'\n\n'+'\n\n'.join(sections)+'\n'
