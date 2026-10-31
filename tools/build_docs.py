"""Build self-contained static game documentation from installed source metadata."""
from pathlib import Path
import html,json,re,shutil

R=Path(__file__).resolve().parents[1];S=R/'src/crucibles_jaxmarl';D=R/'docs'
def esc(x):return html.escape(str(x))
def details(title,obj):
    return '<details><summary>'+esc(title)+'</summary><pre>'+esc(json.dumps(obj,ensure_ascii=False,indent=2))+'</pre></details>'
def shell(title,body,prefix=''):
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+esc(title)+' · Crucibles JAXMARL</title><link rel="stylesheet" href="'+prefix+'style.css"></head><body><header><a href="'+prefix+'index.html">CRUCIBLES <small>JAXMARL</small></a><nav><a href="'+prefix+'guide.html">Install & play</a><a href="'+prefix+'api.html">API</a><a href="'+prefix+'verification.html">Verification</a><a href="https://github.com/hl682/Crucibles_JAXMARL">GitHub ↗</a></nav></header>'+body+'<footer>Original JAX transitions. Explicit evidence. Human scores remain unverified / null. Apache 2.0.</footer></body></html>'
def main():
    rows=json.loads((S/'catalog.json').read_bytes())['games'];(D/'games').mkdir(parents=True,exist_ok=True)
    for row in rows:
        key=row['key'];g=S/'games'/key;m=json.loads((g/'manifest.json').read_bytes())
        scene=json.loads((g/'scene.json').read_bytes()) if (g/'scene.json').exists() else {}
        report=R/'verification'/(key+'.json');v=json.loads(report.read_bytes()) if report.exists() else dict(status='PENDING')
        bp=R/'verification/browser'/(key+'.json');bv=json.loads(bp.read_bytes()) if bp.exists() else dict(status='PENDING')
        row['release_validation']=v['status'];row['browser_validation']=bv['status']
        mechanism=(g/'mechanism.md').read_text(encoding='utf-8-sig') if (g/'mechanism.md').exists() else ''
        actions=''.join('<div class="action"><b>'+str(a['id'])+'</b><span>'+esc(a['label'])+'</span></div>' for a in m['actions'])
        roles=''.join('<p><b>'+esc(a['id'])+'</b> '+esc(a['role'])+'</p>' for a in m['agents'])
        dim=row['dimensions'];body='<main><p class="eyebrow">'+key.upper()+' / '+row['family'].upper()+'</p><h1>'+esc(row['title'])+'</h1><p class="lead">'+esc(row['summary'])+'</p><div class="stats">'+f"<span>{dim['num_agents']} agents</span><span>{dim['num_actions']} actions</span><span>{dim['observation_size']} observation columns</span><span>{dim['max_steps']} original horizon</span><span>Runtime: {v['status']}</span>"+'</div>'
        if (D/'previews'/(key+'.gif')).exists():body+='<section><h2>Original saved demonstration</h2><p>Recorded native state, often authored scripts or interventions; information scope and failures are preserved in the original provenance. This is a saved demonstration, not live browser play.</p><img class="demo" src="../previews/'+key+'.gif" alt="Original saved demonstration '+key+'"></section>'
        if (D/'demos'/(key+'.gif')).exists():body+='<section><h2>Release verification episode</h2><p>Untrained uniform legal actions. Actual JAX state sampled along the full original episode; full-state debug camera. No human performance or learning claim.</p><img class="demo" src="../demos/'+key+'.gif" alt="Actual verification episode '+key+'"></section>'
        body+='<section><h2>Run this game</h2><pre>crucibles play --game '+str(row['id'])+'</pre><p>Start Python locally, then open the address printed by the terminal. <a href="../guide.html">Installation, multiplayer, takeover and keyboard controls →</a></p></section>'
        body+='<section><h2>Objective, rewards & ending</h2>'+details('Exact task contract',m.get('task',dict(termination='Original kernel t >= 32; see mechanism and source')))+'<p>Declared per-step reward bounds: '+esc(m['reward_low'])+' to '+esc(m['reward_high'])+'. Observability: '+esc(m['observability'])+'.</p></section>'
        body+='<section><h2>Rules</h2><ol>'+''.join('<li>'+esc(s)+'</li>' for s in scene.get('rules',[]))+'</ol><details><summary>Original mechanism document (historical status wording retained)</summary><pre class="prose">'+esc(mechanism)+'</pre></details></section>'
        body+='<section><h2>Commands</h2>'+actions+'</section><section><h2>Roles & abilities</h2><div class="scroll">'+roles+'</div></section>'
        body+='<section><h2>Observation permissions</h2><p>Training receives only the original observation rows. Role cameras use declared visibility; debug cameras are explicitly marked. Local benchmark records log debug exposure and role takeover history.</p>'+details('Exact observation layout',m.get('observation_fields',[]))+details('Entity visibility & attributes',m.get('entities',[]))+'</section>'
        body+='<section><h2>Actual verification & provenance</h2>'+details('Per-game execution receipt',v)+details('Chromium control verification',bv)+details('Original identity and file hashes',row)+'<p><a href="https://github.com/hl682/Crucibles_JAXMARL/tree/main/src/crucibles_jaxmarl/games/'+key+'">Complete source, manifest and mechanism →</a></p></section></main>'
        (D/'games'/(key+'.html')).write_text(shell(row['title'],body,'../'),encoding='utf-8')
    cards=''.join('<a class="card" data-family="'+r['family']+'" data-search="'+esc(r['key']+' '+r['title']+' '+r['summary'])+'" href="games/'+r['key']+'.html"><span class="eyebrow">'+r['key'].upper()+' · '+r['family'].replace('_',' ')+'</span><h3>'+esc(r['title'])+'</h3><p>'+esc(r['summary'])+'</p><div class="meta">A'+str(r['dimensions']['num_agents'])+' · H'+str(r['dimensions']['max_steps'])+' · '+r['release_validation']+'</div></a>' for r in rows)
    body='<main><div class="hero"><p class="eyebrow">MULTI-AGENT WORLDS, NATIVE IN JAX</p><h1>Coordination has<br>consequences.</h1><p class="lead">Crucibles brings existing worlds of shared resources, private information, complementary skills and delayed commitments into one installable library.</p><div class="stats"><span>180 spatial source identities</span><span>72 legacy abstract source identities</span><span>252 distinct source hashes</span></div><p><a class="button" href="guide.html">Install & run locally →</a> <a href="api.html">Explore the JAX API</a></p></div><section><h2>Browse the complete source collection</h2><p>Counts describe existing unique source identities. Each card carries its actual verification status. Saved GIFs are demonstrations; live actions require the local Python/JAX server.</p><div class="filters"><input id="search" placeholder="Search games, mechanics or ID" aria-label="Search games"><select id="family"><option value="">All families</option><option value="spatial">Spatial worlds</option><option value="legacy_array">Legacy abstract games</option></select></div><p id="count">252 identities</p><div class="grid">'+cards+'</div></section></main><script src="catalog.js"></script>'
    (D/'index.html').write_text(shell('Worlds',body),encoding='utf-8')
    (D/'catalog.json').write_text(json.dumps(dict(games=rows),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    checks='<main><p class="eyebrow">EVIDENCE / PER GAME</p><h1>What actually ran.</h1><p class="lead">Every environment has a separate execution receipt. Source import, numerical parity, original-horizon execution, render filtering, browser control and human performance remain separate claims.</p>'
    for r in rows:
        p=R/'verification'/(r['key']+'.json');v=json.loads(p.read_bytes()) if p.exists() else dict(status='PENDING')
        checks+='<details><summary>'+r['key'].upper()+' — '+v['status']+' — '+esc(r['title'])+'</summary><pre>'+esc(json.dumps(v,ensure_ascii=False,indent=2))+'</pre></details>'
    checks+='<section><h2>Historical evidence limits</h2><p>GI226/227 original reset floating differences remain numerically undetermined. Their original failed and zero-effect trajectories are retained in the research archive. GI229 original comparison remains undetermined. New software verification does not revise those research conclusions. No old training matrix was resumed.</p></section></main>'
    (D/'verification.html').write_text(shell('Verification',checks),encoding='utf-8')
    for page in ['guide','api']:
        (D/(page+'.html')).write_text(shell(page.title(),(D/(page+'-body.html')).read_text(encoding='utf-8')),encoding='utf-8')
    print('Built',len(rows),'game pages')

if __name__=='__main__':main()
