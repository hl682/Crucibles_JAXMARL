"""Source-grounded editorial metadata; never changes game manifests or kernels."""
from pathlib import Path
import json,re

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src/crucibles_jaxmarl'
CJK=re.compile(r'[\u3400-\u9fff]')
# Multi-label reading aids, not inferred strategy or a proof of equivalence.
THEMES={
 'commons':('Resource commons & externalities','资源公地与外部性',r'pollut|toxin|water|irrigat|pasture|orchard|ecolog|coolant|heat|thermal|soil|meadow|graz|forest|seed|pollen|pollinat|compost|aquifer|plant|nutrient',
  'An action changes a persistent resource or burden that another role also encounters. Compare extraction with maintenance and record the downstream resource trajectory.',
  '动作会改变其他角色也要面对的持久资源或负担。比较抽取与维护时，应记录下游资源轨迹，不能只看当前得分。'),
 'information':('Information, sensing & communication','信息、探测与通信',r'private|hidden|signal|report|broadcast|message|sens(?:e|ing|or)|scan|decod|reveal|observ|evidence|cipher|radar|survey|echo|fog|whisper',
  'Who knows a fact, when it was measured, and whether it was communicated are separate state variables. Test memory or communication using only the same role-visible evidence.',
  '谁知道事实、何时测得、是否传到对方是不同状态。检验记忆或通信作用时，各对照必须遵守相同角色信息权限。'),
 'commitment':('Commitment, exchange & governance','承诺、交换与治理',r'escrow|bond|collateral|credit|debt|loan|insur|underwrit|signature|consent|authoriz|contract|offer|quote|barter|auction|bid|toll|permit|lease|treaty|vote|voting|charter|seal|commit|reservation|reputation',
  'Promises consume time, liquidity, access or material. Separate an announced agreement from a binding commitment and from its eventual delivery; log cancellation and expiry too.',
  '承诺会占用时间、资金、通行或材料。区分口头约定、已生效承诺与最终交付，也记录取消和到期。'),
 'production':('Complementary production & service','互补生产与服务',r'fabricat|finish|manufactur|assembl|workpiece|kiln|forge|foundry|furnace|kitchen|recipe|cook|meal|soup|dish|tray|bowl|ferment|bread|loaf|loaves|ingredient|catalyst|cataly|reaction|repair|service|order|cloth|loom',
  'A useful outcome requires ordered stages, scarce equipment or another role\'s capability. Measure real completed products and resource circulation rather than counting intermediate actions as success.',
  '有效产出依赖有序工序、稀缺设备或另一角色能力。应统计真正完成的产品和资源回流，不能把中间动作数当作成功。'),
 'transport':('Joint transport & shared access','联合运输与共享通行',r'carry|carried|cargo|parcel|deliver|freight|transport|route|gate|bridge|canal|lock|tug|tether|beam|pallet|haul|convoy|courier|ropeway|lift|clutch|carrier|rail|ferry|skiff|boat|barge|conduit|packet|pipeline',
  'Positions, load geometry, occupancy and persistent access constrain whose action can succeed next. Distinguish physical handoff, shared controls and simultaneous force from a cosmetic cooperation label.',
  '位置、载物几何、占位与持久通行条件约束下一步谁能成功。区分实体交接、共用控制和同步合力，不能用合作标签替代机制。'),
 'adversarial':('Team opposition & interception','团队对抗与拦截',r'opponent|enemy|rival|two teams|team 0|team 1|wardens|runners|hunters|evader|pursuit|capture|flag|pennant|\bball\b|tagging|intercept|contest',
  'Cooperation within a team can coexist with opposition between teams. Preserve team membership, capture and possession rules, and report each team\'s outcomes instead of assuming a single shared return.',
  '队内合作可与队间对抗并存。保留分队、捕获与控球规则，分别报告各队结果，不能默认所有角色共享收益。'),
 'hazard':('Survival, protection & rescue','生存、防护与救援',r'fire|rescue|evac|casualt|surviv|oxygen|breath|warm|shelter|collapse|injur|hazard|protection|shield|damage|heat exposure|flood|stranded|allergen',
  'Protection and repair alter another role\'s future feasible actions. Record losses, exposure, rescue and infrastructure as well as delivery; survival is determined by the actual kernel.',
  '防护与维修会改变其他角色未来可行动范围。除交付外也记录损失、暴露、救援和设施状态；生存条件以真实内核为准。'),
}

def read_json(path):return json.loads(path.read_bytes())

def english_title(row,scene):
    title=scene.get('title') or row['title']
    title=re.split(r'\s*/\s*(?=[\u3400-\u9fff])',title)[0]
    title=re.sub(r'^(?:Candidate\s+)?astra-d2-\d+\s*[:—–-]?\s*','',title,flags=re.I)
    title=re.sub(r'\s*[;|]\s*(?:ROLE VIEW|full state).*','',title,flags=re.I)
    title=title.replace(' — ROLE VIEW','').strip(' /;')
    assert not CJK.search(title),(row['key'],title)
    return title[:1].upper()+title[1:]

def mechanism_intro(text):
    text='\n'.join(line for line in text.splitlines() if not line.startswith(('#','Status:','Parentless proposal;')))
    paragraphs=re.split(r'\n\s*\n',text.replace('\r',''))
    for para in paragraphs:
        para=' '.join(para.split())
        if para.startswith(('Two agents','Each agent','Each role','Both agents','Agents share','Each participant','Two independently','The world','Each player')):
            return para
    # Select an actual causal paragraph; exclude historical proposal status.
    for para in paragraphs:
        para=' '.join(para.split())
        if not para.startswith(('#','Status:','This ','Parentless','Candidate:')) and re.search(r'\b(action|Actions|Action)\b',para):return para
    if len(paragraphs)==1:return ' '.join(paragraphs[0].split())
    raise ValueError('No causal mechanism paragraph')

def profiles():
    authored={}
    curated=dict(line.split('\t') for line in (ROOT/'manuals/classification.tsv').read_text(encoding='utf-8').splitlines())
    codes={'C':'commons','I':'information','K':'commitment','P':'production','T':'transport','A':'adversarial','H':'hazard'}
    for line in (ROOT/'manuals/authoring.tsv').read_text(encoding='utf-8').splitlines():
        key,title,brief=line.split('\t');authored[key]=(title,brief)
    rows=read_json(SRC/'catalog.json')['games'];assert len(rows)==len(authored)==256
    assert set(curated)==set(authored)
    assert all(v and set(v)<=set(codes) and len(v)==len(set(v)) for v in curated.values())
    result={}
    for row in rows:
        key=row['key'];g=SRC/'games'/key;m=read_json(g/'manifest.json')
        sc=read_json(g/'scene.json') if (g/'scene.json').exists() else {}
        mech=(g/'mechanism.md').read_text(encoding='utf-8-sig')
        intro=mechanism_intro(mech) if row['family']=='legacy_array' else m['task']['objective']
        # Prefer explicit task/action/rule evidence to titles or visual styling.
        evidence=[('manifest.task.objective',m['task']['objective'])]+[('manifest.task.'+field,m['task'][field]) for field in ['interaction','observation_access'] if field in m['task']]+[('manifest.actions.'+str(a['id']),a['label']) for a in m['actions']]+[('scene.rules.'+str(i),v) for i,v in enumerate(sc.get('rules',[])) if not re.search(r'animate|render|diagnostic|recorded|unexecuted|authored source|show the|display|debug|masquerade',v,re.I)]
        if row['family']=='legacy_array':evidence.append(('mechanism.md',intro))
        tags=[]
        for letter in curated[key]:
            code=codes[letter];pattern=THEMES[code][2]
            matches=[dict(field=p,text=t) for p,t in evidence if re.search(r'\b(?:'+pattern+')',t,re.I)]
            if not matches:matches=[dict(field='mechanism.md' if row['family']=='legacy_array' else 'manifest.task.objective',text=intro)]
            tags.append(dict(code=code,assignment='CURATED_RULE_READING',evidence=matches[:3]))
        assert tags,key
        result[key]=dict(key=key,en=dict(title=english_title(row,sc),brief=intro),zh=dict(title=authored[key][0],brief=authored[key][1]),themes=tags)
    return result

ACTION_HELP=[
 (r'wait|idle|rest|coast|cool|save|recharge|replenish','等待、恢复或补充；是否产生被动变化取决于本款规则。'),
 (r'move|walk|north|south|east|west|climb|swim|thrust|steer|throttle','按按钮方向移动或施力；距离、占位、载物和阶段仍须满足本款限制。'),
 (r'cancel|withdraw|recall|revoke|timeout|retract','撤回、取消或超时处理已有安排；已锁定成本、保证金或物品按本款规则处理。'),
 (r'inspect|scan|sense|survey|probe|audit|measure','执行检查或测量；核对成本、信息归属和测量时点，结果不自动对所有人公开。'),
 (r'report|broadcast|transmit.*(?:bit|symbol|glyph|survey|register|message)|message|signal|disclose|reveal|send.*(?:bit|symbol|glyph)','发送或公开所列信息；接收延迟、有效期和信道竞争按本款规则处理。'),
 (r'accept|receive|decode|catch|clear incoming|process inbox','接收、确认或处理已经到达的对象；先确认目标、容量和当前阶段。'),
 (r'offer|escrow|pledge|bond|deposit|sign|authoriz|consent|vote|bid|quote|insure|underwrite|lend|repay|credit|seal|commit|reserve|claim|witness','提交所列交易、权利或承诺操作；注意对方同意、余额、有效期与既有承诺。'),
 (r'open|close|toggle|gate|switch|valve|latch|clamp|engage|disengage|dial|register|rotate','改变所列开关、锁存或配置；持续多久以及何时影响伙伴，以当前规则为准。'),
 (r'clean|wash|scrub|filter|purge|clear|repair|maintain|service|reinforce|support','处理清理、维修或支撑；检查耗材、距离及对公共设施的影响。'),
 (r'plant|sow|water|irrigat|pollen|nutrient|cultivat|grow|feed','进行所列种植或照料；资源和生长阶段限制仍然有效，收益可能延迟。'),
 (r'harvest|collect|bank|deliver|ship|sell|serve|finish|consume|eat|extract','进行所列采收、交付或兑现；地点、对象和加工状态必须匹配，不能假定按钮必然成功。'),
 (r'pick|drop|grab|release|attach|detach|carry|hand|transfer|transmit.*energy|load|unload|route|forward|launch|pump|pack|tow|pull|push','操作实体物料、能量、连接或运输；核对库存、携带容量、目标和伙伴动作。'),
 (r'work|produce|fabricat|make|manufactur|process|stage|cook|heat|forge|temper|cut|chop|assembl|build|mine|dig|drill|fire|charge','执行所列生产、施工或设备操作；互补工序、热度、工具和材料前提仍须满足。'),
 (r'tag|tackle|shoot|block|guard|shield|rescue|aid|cover|decoy|hide|erase|jam|capture|sprint','执行所列对抗、防护、隐蔽或救援操作；目标距离、能量和冷却以合法提示为准。'),
]
def action_help(label):
    exact={'Set dial 0':'把自己的拨盘设为零。','Set dial 1':'把自己的拨盘设为一。','Send message 0':'把零比特放入发往伙伴的延迟消息。','Send message 1':'把一比特放入发往伙伴的延迟消息。'}
    if label in exact:return exact[label]
    for pattern,text in ACTION_HELP:
        if re.search(pattern,label.replace('_',' '),re.I):return text
    return '使用所列专用指令；动作编号与英文游戏按钮一一对应，须结合上面的玩法和当前合法提示执行。'

if __name__=='__main__':
    p=profiles();out={k:dict(title=v['en']['title'],summary=v['en']['brief']) for k,v in p.items()}
    (SRC/'presentation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (ROOT/'manuals/profiles.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Prepared',len(p),'source-grounded bilingual profiles')
