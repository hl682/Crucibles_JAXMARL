'use strict';
const $=id=>document.getElementById(id);let session=null,token=null,current=null,timer=null,busy=false,queued={},queueWrites=Promise.resolve();
const params=new URLSearchParams(location.hash.slice(1));session=params.get('session');token=params.get('token');
async function api(body){const r=await fetch('/api',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session,token,...body})});const v=await r.json();if(!r.ok||v.error)throw Error(v.error||r.status);return v}
function fail(e){$('error').textContent=e.message||String(e);$('progress').textContent='Request failed. The original state and logs remain available.'}
function action(fn){return async()=>{try{$('error').textContent='';await fn()}catch(e){fail(e)}}}
function parseRoles(){const out=new Set();for(const part of $('roles').value.split(',')){const m=part.trim().match(/^(\d+)(?:-(\d+))?$/);if(!m)throw Error('Use roles such as 0,2,5-8');const a=Number(m[1]),b=m[2]?Number(m[2]):a;if(b<a||b-a>10000)throw Error('Invalid role range');for(let i=a;i<=b;i++)out.add(i)}return [...out]}
function option(value,text){const o=document.createElement('option');o.value=value;o.textContent=text;return o}
function color(s){let h=0;for(const c of String(s))h=(h*31+c.charCodeAt(0))>>>0;return `hsl(${h%360} 43% 49%)`}
function directionOf(label){const s=label.toLowerCase().replaceAll('_',' ');if(!/^(north|east|south|west)\b|\b(move|walk|step|throttle|climb|swim|attempt|steer|pull|thrust|scout|request|travel|face)\b/.test(s))return null;const m=s.match(/\b(northward|north|n|eastward|east|e|southward|south|s|westward|west|w)\b/);return m?m[1][0].toUpperCase():null}
function queueAction(role,id){const target={session,token};queueWrites=queueWrites.catch(()=>{}).then(()=>api({...target,op:'actions',actions:{[role]:id}}));return queueWrites}
function sprite(c,e,x,y,r){
  const g=String(e.glyph).toLowerCase(),a=e.attributes||{};c.save();c.translate(x,y);c.fillStyle=e.color||color(e.group);c.strokeStyle='#15262a';c.lineWidth=Math.max(1,r*.13);
  const rect=(x,y,w,h)=>{c.fillRect(x,y,w,h);c.strokeRect(x,y,w,h)};
  const ellipse=(x,y,rx,ry)=>{c.beginPath();c.ellipse(x,y,rx,ry,0,0,Math.PI*2);c.fill();c.stroke()};
  if(e.kind==='agent'||/worker|person|chef|citizen|crew|farmer/.test(g)){
    c.fillStyle=e.color||'#dfae83';ellipse(0,-r*.6,r*.45,r*.45);c.fillStyle=color('team'+Object.values(a)[0]);rect(-r*.55,-r*.15,r*1.1,r*1.05);c.strokeStyle='#d3ded8';c.beginPath();c.moveTo(-r*.35,r*.9);c.lineTo(-r*.35,r*1.25);c.moveTo(r*.35,r*.9);c.lineTo(r*.35,r*1.25);c.stroke();
  }else if(/plant|crop|tree|seed|flower/.test(g)||/plot|crop|garden/.test(e.group)){
    c.fillStyle='#685141';rect(-r,-r*.1,r*2,r);const growth=Object.entries(a).find(([k])=>/crop|growth|stage|maturity/.test(k));const n=growth?Math.max(0,Number(growth[1])):2;c.strokeStyle='#a7cf79';c.lineWidth=Math.max(1,r*.16);c.beginPath();c.moveTo(0,r*.5);c.lineTo(0,-r*(.25+Math.min(5,n)*.15));c.stroke();c.fillStyle=n>=3?'#d5d876':'#69ad6f';ellipse(-r*.4,-r*.2,r*.45,r*.22);ellipse(r*.4,-r*.5,r*.45,r*.22);
  }else if(/water|reservoir|pond|canal|trough/.test(g+' '+e.group)){
    c.fillStyle='#365a6d';rect(-r,-r*.65,r*2,r*1.3);c.fillStyle='#73c3ce';rect(-r*.8,-r*.4,r*1.6,r*.7);c.strokeStyle='#b8e5e8';c.beginPath();c.moveTo(-r*.6,0);c.lineTo(r*.6,0);c.stroke();
  }else if(/gate|door|bridge/.test(g+' '+e.group)){
    c.fillStyle=e.color||'#b6956d';rect(-r,-r,r*.35,r*2);rect(r*.65,-r,r*.35,r*2);const open=Object.entries(a).find(([k])=>/open/.test(k));if(!open||!open[1])rect(-r*.65,-r*.35,r*1.3,r*.65);else{c.strokeStyle='#a9d493';c.beginPath();c.moveTo(-r*.7,0);c.lineTo(r*.6,-r);c.stroke()}
  }else if(/boat|barge|ship|vessel/.test(g+' '+e.group)){
    c.beginPath();c.moveTo(-r,-r*.6);c.lineTo(r,-r*.6);c.lineTo(r*.65,r*.6);c.lineTo(-r*.65,r*.6);c.closePath();c.fill();c.stroke();c.fillStyle='#d8c39d';rect(-r*.4,-r*.6,r*.8,r*.7);
  }else if(/fire|flame/.test(g+' '+e.group)){
    c.fillStyle='#dc7d4e';c.beginPath();c.moveTo(-r,r);c.lineTo(-r*.6,-r*.3);c.lineTo(0,-r);c.lineTo(r,r);c.closePath();c.fill();c.fillStyle='#f4d575';ellipse(0,r*.3,r*.4,r*.6);
  }else if(/herd|animal|grazer|cow|sheep/.test(e.group)){
    c.fillStyle=e.color||'#dbd7b7';ellipse(0,0,r*.8,r*.55);ellipse(r*.75,-r*.4,r*.4,r*.35);c.beginPath();for(const dx of [-.5,.4]){c.moveTo(dx*r,r*.4);c.lineTo(dx*r,r)}c.stroke();
  }else if(/station|oven|forge|kitchen|stove|workshop/.test(g+' '+e.group)){
    c.fillStyle=e.color||'#9aa8a4';rect(-r,-r,r*2,r*2);c.fillStyle='#24353a';ellipse(0,-r*.25,r*.5,r*.4);c.fillStyle='#e1ac6d';rect(-r*.5,r*.35,r,r*.35);
  }else{rect(-r,-r,r*2,r*2);c.strokeStyle='#ecdcaa';c.beginPath();c.moveTo(-r*.75,-r*.75);c.lineTo(r*.75,r*.75);c.moveTo(r*.75,-r*.75);c.lineTo(-r*.75,r*.75);c.stroke()}
  c.restore();
}
let paintSerial=0;const paintedSteps=new Set();
async function render(frame){const canvas=$('world'),c=canvas.getContext('2d'),w=canvas.width,h=canvas.height;c.fillStyle='#101e29';c.fillRect(0,0,w,h);if(!frame){c.fillStyle='#a4b4b5';c.font='22px sans-serif';c.fillText('Assign roles or enable the labelled full-state debug camera.',40,90);return}if(frame.original_animation){const a=frame.original_animation,serial=++paintSerial,stamp=session+':'+frame.role+':'+frame.clock;const images=paintedSteps.has(stamp)?a.frames.slice(-1):a.frames;paintedSteps.add(stamp);for(let i=0;i<images.length;i++){const im=new Image();await new Promise((ok,bad)=>{im.onload=ok;im.onerror=()=>bad(Error('Scene image failed to load'));im.src=images[i]});if(serial!==paintSerial)return;const scale=Math.min(w/a.width,h/a.height),x=(w-a.width*scale)/2,y=(h-a.height*scale)/2;c.fillStyle='#101e29';c.fillRect(0,0,w,h);c.drawImage(im,x,y,a.width*scale,a.height*scale);if(i+1<images.length)await new Promise(ok=>setTimeout(ok,a.duration_ms))}canvas.onclick=event=>action(()=>(current.game.key==='gi214'?magneticClick:bistroClick)(event,frame))();canvas.onmousemove=null;return}canvas.onclick=null;c.fillStyle='#e8eee5';c.font='16px sans-serif';c.fillText(`${frame.game.toUpperCase()} · ${frame.information} · t=${frame.clock}`,24,30);
if(frame.legacy){c.fillText('Original abstract resource / communication game — native observations',24,65);frame.legacy_observation.forEach((a,i)=>{const y=120+i*190;a.forEach((v,j)=>{const x=30+j*58,bar=Math.min(120,Math.abs(v)*100);c.fillStyle=color(j);c.fillRect(x,y+120-bar,40,bar);c.fillStyle='#eee';c.fillText(`${j}: ${v.toFixed(2)}`,x,y+145)})});return}
const [x0,y0,x1,y1]=frame.bounds,scale=Math.min((w-70)/(x1-x0+1),(h-110)/(y1-y0+1));const pt=p=>[35+(p[0]-x0+.5)*scale,65+(p[1]-y0+.5)*scale];
if(frame.map){const [ox,oy]=frame.map_origin,[sx,sy]=frame.cell_size,palette=Object.fromEntries((frame.tile_legend||[]).map(t=>[t.value,t.color]));frame.map.forEach((row,y)=>row.forEach((v,x)=>{const [px,py]=pt([ox+x*sx,oy+y*sy]);c.fillStyle=v===null?'#0b151c':palette[v]||color('tile'+v);c.fillRect(px-scale*.49,py-scale*.49,scale*.98,scale*.98);if(v!==null){c.strokeStyle='#ffffff0a';c.strokeRect(px-scale*.49,py-scale*.49,scale*.98,scale*.98)}}))}
for(const e of [...frame.entities].sort((a,b)=>(a.kind==='agent')-(b.kind==='agent'))){const [x,y]=pt(e.xy),r=Math.max(3,Math.min(14,scale*.32));sprite(c,e,x,y,r);if(e.kind==='agent'&&scale>=17){c.fillStyle='#fff';c.font=`${Math.max(9,scale*.33)}px sans-serif`;c.fillText(String(e.index),x+r+2,y+3)}}
canvas.onmousemove=event=>{const rect=canvas.getBoundingClientRect(),x=(event.clientX-rect.left)*w/rect.width,y=(event.clientY-rect.top)*h/rect.height;const e=frame.entities.find(e=>{const p=pt(e.xy);return Math.hypot(p[0]-x,p[1]-y)<Math.max(10,scale*.5)});canvas.title=e?`${e.label||e.group} ${e.index} @ ${e.xy.join(', ')}\n${Object.entries(e.attributes).map(([k,v])=>k+': '+JSON.stringify(v)).join('\n')}`:'Native world state'};
$('legend').replaceChildren(...[...new Set(frame.entities.map(e=>e.group))].map(s=>{const t=document.createElement('span');t.className='tag';t.style.borderLeft=`4px solid ${color(s)}`;t.textContent=s;return t}));
}
async function refresh(){if(!session)return;const v=await api({op:'snapshot',debug:$('debug').checked});current=v;$('session').hidden=false;$('title').textContent=v.game.title;$('clock').textContent=`t=${v.step} / ${v.game.dimensions.max_steps}`;$('scope').textContent=`${v.information} · ${v.game.dimensions.num_agents} independently actionable agents · ${v.ai_policy} · ${v.terminal?'EPISODE ENDED':'active'} · human scores: unverified / null`;
const old=$('camera').value;$('camera').replaceChildren(...v.roles.map(r=>option(r,`Role ${r}`)));if(v.views['-1'])$('camera').append(option(-1,'Full-state debug'));if([...$('camera').options].some(o=>o.value===old))$('camera').value=old;await render(v.views[$('camera').value]);
const target=$('player').value;$('player').replaceChildren(...v.players.map(p=>option(p.name,`${p.name} (${p.roles.length} roles)`)),option('AI','AI / release control'));if([...$('player').options].some(o=>o.value===target))$('player').value=target;
$('controls').replaceChildren(...v.roles.map(r=>{const d=document.createElement('div');d.className='role';const label=document.createElement('span');label.textContent=`Role ${r}`;const select=document.createElement('select');select.id=`action-${r}`;v.actions.filter(a=>v.legal[r][a.id]).forEach(a=>select.append(option(a.id,a.label)));if(queued[r]!==undefined&&[...select.options].some(o=>Number(o.value)===queued[r]))select.value=queued[r];const button=document.createElement('button');button.textContent=$('direct').checked&&v.host?'Do action':'Queue';button.onclick=action(async()=>{if($('direct').checked&&v.host)return directAction(r,Number(select.value));queued[r]=Number(select.value);await queueAction(r,queued[r]);button.textContent='Queued'});d.append(label,select,button);return d}));
const paragraphs=[v.task.objective,v.task.natural_termination,`Native per-step reward bounds: ${v.reward_bounds.join(' … ')}`,...(v.rules||[])].filter(Boolean).map(text=>{const p=document.createElement('p');p.textContent=text;return p});const roles=document.createElement('details'),summary=document.createElement('summary');summary.textContent=`All ${v.agent_roles.length} roles and abilities`;roles.append(summary,...v.agent_roles.map(a=>{const p=document.createElement('p');p.textContent=`${a.id}: ${a.role}`;return p}));$('rules').replaceChildren(...paragraphs,roles);for(const id of ['assign','all','invite','step','auto','debug','close'])$(id).disabled=!v.host||v.terminal&&['step','auto'].includes(id);if(v.terminal)stop();if(v.bistro){const b=v.bistro;$('feedback').textContent=`Orders served: ${b.served}/4 · Clean water: ${b.tank}/12 · Pan: ${b.pan_dirty?'CONTAMINATED':'clean'} · ${v.views[$('camera').value]?.original_animation?.events.join(' · ')||'Click a worker to select; WASD moves. Click adjacent items or stations to use them.'}`}else if(v.game.key==='gi214'){const a=v.views[$('camera').value]?.original_animation;$('feedback').textContent=a?`Shared power: ${a.shared.charge}/32 · Builds: ${a.shared.builds} · Inventory (${a.inventory_labels.join(', ')}): ${(a.inventory||[]).join(', ')} · ${a.events.join(' · ')}`:''}else $('feedback').textContent='';$('progress').textContent=`Session ${v.id.slice(0,8)} ready. Logs are saved by Python.`}
async function step(){if(busy)return;busy=true;try{await queueWrites;await api({op:'step'});queued={};await refresh()}finally{busy=false}}
function stop(){if(timer)clearInterval(timer);timer=null;$('auto').textContent='Start AI / queued-action clock'}
async function createSession(){stop();$('create').disabled=true;$('progress').textContent='Compiling original JAX reset… first compilation can take a minute.';try{const v=await api({op:'create',game:$('game').value,seed:Number($('seed').value)});session=v.session;token=v.token;location.hash=new URLSearchParams({session,token}).toString();$('debug').checked=false;queued={};await refresh()}finally{$('create').disabled=false}}
$('create').onclick=action(createSession);

async function directAction(role,id){if(busy){$('feedback').textContent='Finishing the current action. Try again when the animation ends.';return}if(!current.legal[role]?.[id])throw Error('That action is not available from this position');if(['gi214','gi215'].includes(current.game.key))for(const r of current.roles)if(r!==role)await queueAction(r,0);queued[role]=id;await queueAction(role,id);await step()}
async function objectAction(role,id){
  $('object-actions').hidden=true;
  if(current.host&&$('direct').checked)await directAction(role,id);
  else{queued[role]=id;await queueAction(role,id);const button=$('action-'+role)?.parentElement.querySelector('button');if(button)button.textContent='Queued';$('feedback').textContent='Action queued; the host advances the clock.'}
}
function showObjectActions(label,role,choices){
  const panel=$('object-actions');panel.replaceChildren();const heading=document.createElement('strong');heading.textContent=label+' · ';panel.append(heading);
  for(const [id,text]of choices){if(!current.legal[role]?.[id])continue;const b=document.createElement('button');b.textContent=text;b.onclick=action(()=>objectAction(role,id));panel.append(b)}
  panel.hidden=false;
}
async function magneticClick(event,frame){
  $('object-actions').hidden=true;
  const c=$('world'),a=frame.original_animation,rect=c.getBoundingClientRect(),scale=Math.min(c.width/a.width,c.height/a.height);
  const gx=(((event.clientX-rect.left)*c.width/rect.width-(c.width-a.width*scale)/2)/scale-a.geometry.ox)/a.geometry.cell-.5;
  const gy=(((event.clientY-rect.top)*c.height/rect.height-(c.height-a.height*scale)/2)/scale-a.geometry.oy)/a.geometry.cell-.5;
  const hit=frame.entities.filter(e=>Math.hypot(e.xy[0]-gx,e.xy[1]-gy)<.48).sort((x,y)=>(y.kind==='agent')-(x.kind==='agent'))[0];
  if(hit?.kind==='agent'&&current.roles.includes(hit.index)){$('camera').value=hit.index;await render(current.views[hit.index]);return}
  const role=Number($('camera').value),actor=frame.entities.find(e=>e.kind==='agent'&&e.index===role);if(!actor)return;
  const pos=hit?.xy||[Math.round(gx),Math.round(gy)],distance=Math.abs(pos[0]-actor.xy[0])+Math.abs(pos[1]-actor.xy[1]);
  if(distance>1){$('feedback').textContent='Move beside this object with WASD first.';return}
  if(hit?.group==='magnets'){showObjectActions('Magnet',role,[[14,'Switch power'],[15,'Reverse polarity']]);return}
  if(hit?.group==='deposits'){showObjectActions('Resource deposit',role,[[5,'Gather resource']]);return}
  if(hit?.group==='freight'){showObjectActions('Freight',role,[[6,'Pick up freight'],[7,'Put down freight'],[21,'Repair or unjam'],[23,'Anchor freight']]);return}
  if(hit?.group==='sites'){showObjectActions('Workstation',role,[[8,'Craft coil'],[9,'Craft brace'],[10,'Craft power pack'],[17,'Supply shared power'],[18,'Deliver freight']]);return}
  if(distance===1){const dx=pos[0]-actor.xy[0],dy=pos[1]-actor.xy[1];await objectAction(role,dy===-1?1:dx===1?2:dy===1?3:4)}
}
async function bistroClick(event,frame){
  $('object-actions').hidden=true;
  const canvas=$('world'),a=frame.original_animation,rect=canvas.getBoundingClientRect(),scale=Math.min(canvas.width/a.width,canvas.height/a.height);
  const px=((event.clientX-rect.left)*canvas.width/rect.width-(canvas.width-a.width*scale)/2)/scale,py=((event.clientY-rect.top)*canvas.height/rect.height-(canvas.height-a.height*scale)/2)/scale;
  const gx=(px-a.geometry.ox)/a.geometry.cell-.5,gy=(py-a.geometry.oy)/a.geometry.cell-.5;
  const hit=frame.entities.filter(e=>Math.hypot(e.xy[0]-gx,e.xy[1]-gy)<.48).sort((x,y)=>(y.kind==='agent')-(x.kind==='agent'))[0];
  if(hit?.kind==='agent'&&current.roles.includes(hit.index)){$('camera').value=hit.index;await render(current.views[hit.index]);return}
  const r=Number($('camera').value),actor=frame.entities.find(e=>e.kind==='agent'&&e.index===r);if(!actor)return;
  const pos=hit?.xy||[Math.round(gx),Math.round(gy)],distance=Math.abs(pos[0]-actor.xy[0])+Math.abs(pos[1]-actor.xy[1]);
  if(distance>1){$('feedback').textContent='Move beside the item or station with WASD first.';return}
  let id=null;const held=frame.entities.find(e=>e.group==='kitchen_items'&&e.attributes.holder===r);
  if(hit?.group==='kitchen_items'){
    const first=frame.entities.filter(e=>e.group==='kitchen_items'&&e.attributes.holder===-1&&Math.abs(e.xy[0]-actor.xy[0])+Math.abs(e.xy[1]-actor.xy[1])<=1).sort((x,y)=>x.index-y.index)[0];
    if(!held&&first?.index!==hit.index){$('feedback').textContent='Another item is within reach first. Move closer to the selected item.';return}
    id=held?6:5;
  }else if(hit?.group==='stations'){
    const cook=frame.entities.some(e=>e.group==='utensils'&&e.index===0&&e.attributes.tool_holder===r),b=current.bistro;
    const choices=[[[7,cook?'Prepare ingredient':'Scrub pan'],[12,'Assemble soup']],[[7,'Wash bowl'],[9,b.draining?'Close drain':'Open drain']],[[8,held?.attributes.kind===2?'Clean filter':'Pump clean water']],[[13,'Serve order']],[[10,b.divert?'Send water toward pan':'Divert water to basin'],[11,'Install held filter']],[[14,'Discard held item']],[]];
    showObjectActions(['Pan','Sink','Pump','Serving counter','Valve','Bin','Waste basin'][hit.index],r,choices[hit.index]);return;
  }else if(distance===1){const dx=pos[0]-actor.xy[0],dy=pos[1]-actor.xy[1];id=dy===-1?1:dx===1?2:dy===1?3:4}
  if(id!==null)await objectAction(r,id);
}
$('quickplay').onclick=action(async()=>{await createSession();await api({op:'assign',player:'Host',roles:['gi214','gi215'].includes(current.game.key)?Array.from({length:current.game.dimensions.num_agents},(_,i)=>i):[0]});$('direct').checked=true;paintedSteps.clear();await refresh()});
$('direct').onchange=action(refresh);
$('assign').onclick=action(async()=>{await api({op:'assign',player:$('player').value,roles:parseRoles()});await refresh()});
$('all').onclick=action(async()=>{await api({op:'assign',player:$('player').value,roles:Array.from({length:current.game.dimensions.num_agents},(_,i)=>i)});await refresh()});
$('invite').onclick=action(async()=>{const name=`Player ${current.players.length}`;const v=await api({op:'invite',name});$('invitation').hidden=false;$('invitation').value=location.origin+'/#'+new URLSearchParams({session,token:v.invite});await refresh()});
$('step').onclick=action(step);$('refresh').onclick=action(refresh);$('debug').onchange=action(refresh);$('camera').onchange=()=>render(current.views[$('camera').value]);
$('close').onclick=action(async()=>{stop();await api({op:'close'});session=null;token=null;current=null;queued={};location.hash='';$('session').hidden=true;$('controls').replaceChildren();$('progress').textContent='Session closed. Logs preserved on disk.';render(null)});
$('auto').onclick=()=>{if(timer){stop();return}timer=setInterval(()=>step().catch(e=>{stop();fail(e)}),Math.max(250,Number($('interval').value)||800));$('auto').textContent='Pause clock'};
document.addEventListener('keydown',e=>{if(/INPUT|SELECT|TEXTAREA/.test(e.target.tagName)||!current)return;if(e.code==='Space'){e.preventDefault();action(step)();return}const dir={ArrowUp:'N',KeyW:'N',ArrowRight:'E',KeyD:'E',ArrowDown:'S',KeyS:'S',ArrowLeft:'W',KeyA:'W'}[e.code];if(!dir)return;const r=Number($('camera').value);if(r<0||!current.roles.includes(r))return;const a=current.actions.find(a=>directionOf(a.label)===dir&&current.legal[r][a.id]);if(a){e.preventDefault();$('action-'+r).value=a.id;if($('direct').checked&&current.host){action(()=>directAction(r,a.id))();return}queued[r]=a.id;queueAction(r,a.id).then(()=>{$('progress').textContent=`Queued ${a.label} for role ${r}`}).catch(fail)}});
(async()=>{try{const health=await(await fetch('/health')).json();$('health').textContent=`Python ready • ${health.games} source identities`;const data=await(await fetch('/api/catalog')).json();$('game').replaceChildren(...data.games.map(g=>option(g.key,`${g.key.toUpperCase()} · ${g.title} · A${g.dimensions.num_agents}`)));const game=new URLSearchParams(location.search).get('game')||'225';$('game').value=/^(?:gi)?\d+$/i.test(game)?`gi${Number(game.replace(/^gi/i,'')).toString().padStart(3,'0')}`:game;if(session&&token)await refresh()}catch(e){fail(e)}})();
