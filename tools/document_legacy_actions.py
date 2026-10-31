"""Source-reviewed action names for every original abstract game, no rule edits."""
from pathlib import Path
import ast,hashlib,json

R=Path(__file__).resolve().parents[1];S=R/'src/crucibles_jaxmarl'
LABELS=[
['Idle','Charge','Transmit energy','Harvest','Intercept partner energy'],
['Cool','Fabricate','Forward goods','Finish partner goods','Purge goods'],
['Idle','Gentle pump','Intensive pump','Plant','Filter'],
['Wait','Move clockwise','Move anticlockwise','Toggle owned gate','Deliver'],
['Wait','Open irrigation gate further','Close irrigation gate further','Request extra water','Harvest crop'],
['Produce stock','Offer escrow','Pledge bond','Withdraw escrow and forfeit bond','Consume stock'],
['Produce','Clean pollution','Maintain filter','Close shutter','Open shutter'],
['Set dial 0','Set dial 1','Send message 0','Send message 1','Charge lock'],
['Wait','Launch parcel type 0','Launch parcel type 1','Set receiver gate 0','Set receiver gate 1'],
['Wait','Draw public water','Consume private reserve','Plant nursery','Clean toxin'],
['Wait','Request credit','Lend to requesting partner','Repay debt','Enforce aged partner debt'],
['Wait','Inspect private signal','Publish claim 0','Publish claim 1','Launch settlement'],
['Wait','Make goods','Post escrow','Ship against partner escrow','Cancel escrow'],
['Wait','Extract water','Clean contamination','Restore filter','Toggle gate'],
['Wait','Sign project 1','Sign project 2','Work on active project','Withdraw signature or active project'],
['Recharge','Scan private sample','Request broadcast','Launch claim 0','Launch claim 1'],
['Wait','Deposit bond','Work on committed project','Sign completed work','Withdraw and forfeit half bond'],
['Wait','Rotate emitter','Send current glyph','Rotate decoder','Accept and decode mail'],
['Rest','Harvest local patch','Sow seeds','Clean local toxin','Move to other patch'],
['Wait','Load cargo','Commit route','Vote switch 0','Vote switch 1'],
['Wait','Open channel 1 escrow','Open channel 2 escrow','Authorize release','Revoke escrow'],
['Rest','Inject owned work','Process local queue','Route work to partner','Maintain station'],
['Wait','Harvest biomass','Feed nutrients','Toggle connection valve','Filter toxin'],
['Wait','Send symbol 0','Send symbol 1','Toggle decoder','Accept pending symbol'],
['Produce','Pledge inventory','Authorize consent','Sell shipment','Inspect grade'],
['Wait','Feed chamber','Catalyse reaction','Cool chamber','Harvest product'],
['Sample field','Relay signal','Harvest negative direction','Harvest positive direction','Restore resource'],
['Flip first register','Flip second register','Rotate register routing','Shield incoming pulse','Start stability audit'],
['Save','Issue escrow','Guarantee partner escrow','Cancel active escrow','Expedite active escrow'],
['Admit fluid','Process fluid','Forward fluid','Filter dirt','Regenerate filter'],
['Coast and recharge','Queue positive impulse','Queue negative impulse','Brake','Flip gear'],
['Sample private evidence','Announce 0','Announce 1','Commit 0','Commit 1'],
['Refine stock','Dispatch lot','Inspect pending lot','Accept pending lot','Arm warranty'],
['Monitor','Extract','Flush sediment','Filter contamination','Build gate'],
['Wait','Renew service ticket','Set setup latch 0','Set setup latch 1','Serve matching ticket'],
['Publish negative report','Publish positive report','Audit partner report','Invest negative','Invest positive'],
['Recharge','Harvest stock','Send catalyst type 0','Process with available catalyst','Send catalyst type 1'],
['Save','Deposit escrow','Endorse paired escrow','Cancel deposit','Settle approved escrow'],
['Cool','Route job type 0','Route job type 1','Restore shield','Vent heat'],
['Recharge','Propose 0','Propose 1','Inspect private clue','Authorize vote'],
['Produce stock','Make honest delivery','Make shortcut delivery','Inspect partner delivery','Buy insurance shield'],
['Bank water','Pump','Flush','Dredge','Release reservoir water'],
['Recharge','Send ordinary version','Read cached version','Write content','Send fencing lease'],
['Rest','Work','Screen infection','Vaccinate','Treat'],
['Wait','Prepare cargo','Inspect','Ship cargo','Recharge'],
['Rest','Apply positive impulse','Apply negative impulse','Switch polarity','Probe oscillator'],
['Rest','Inject packet','Switch loop','Receive packet','Flush loop'],
['Wait','Sow type 0','Sow type 1','Harvest','Inspect'],
['Wait','Launch insurance project','Contribute capital','Audit peer','Cancel project'],
['Wait','Stage assembly 0','Stage assembly 1','Disclose demand','Lock irreversible stage'],
['Rest and replenish','Load batch','Release culture','Harvest batch','Scrub chamber'],
['Recharge','Launch packet 0','Launch packet 1','Toggle shared switch','Clamp switch'],
['Save','Place parcel in escrow','Witness partner parcel','Release sealed parcel','Recall parcel'],
['Rest and replenish cistern','Pump','Send liquid','Sell liquid','Service equipment'],
['Rest','Work choice 0','Work choice 1','Probe partner target','Transmit survey'],
['Rest','Plant seeds','Irrigate','Emit pollen','Harvest fruit'],
['Wait','Pack cargo','Launch bonded cargo','Clear incoming cargo','Maintain'],
['Rest','Process backlog','Extract heat','Open exchange valve','Close exchange valve'],
['Rest','Move negative direction','Move positive direction','Plant','Harvest'],
['Harvest stock','Vote allocation 0','Vote allocation 1','Repair','Survey'],
['Replenish inventory','Deposit escrow','Sign authorization','Withdraw escrow','Inspect'],
['Idle','Request low-intensity work','Request high-intensity work','Service equipment','Replenish'],
['Wait','Increment register','Decrement register','Transmit register','Execute instruction'],
['Wait','Harvest fruit','Plant seeds','Water','Clear competing vegetation'],
['Wait','Grow biomass','Harvest','Clean toxin','Release catalyst'],
['Wait','Launch pending transaction','Underwrite partner transaction','Claim available credit','Recall bond'],
['Coast','Pump','Collect','Switch flow','Reinforce'],
['Wait','Report 0','Report 1','Audit partner report','Trade on report'],
['Save','Start project','Insure partner project','Liquidate project','Audit partner project'],
['Wait','Send ordinary packet','Send protected packet','Process inbox','Timeout transmission'],
['Replenish nutrients','Plant variety 0','Plant variety 1','Irrigate','Release pollen'],
['Coast','Engage clutch','Disengage clutch','Shift disengaged gear','Apply engaged pulse'],
]

def main():
    assert len(LABELS)==72
    catalog=json.loads((S/'catalog.json').read_bytes())
    for row in catalog['games'][:72]:
        p=S/'games'/row['key'];mp=p/'manifest.json';m=json.loads(mp.read_bytes())
        history=R/'source-history'/row['key']/'release-metadata-v0.json'
        if not history.exists():history.parent.mkdir(parents=True,exist_ok=True);history.write_bytes(mp.read_bytes())
        source=(p/'candidate.py').read_text(encoding='utf-8-sig');tree=ast.parse(source)
        observe=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='observe')
        step=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='raw_step')
        m['actions']=[dict(id=i,label=label) for i,label in enumerate(LABELS[row['id']])]
        m['observation_fields']=[dict(slice=[0,m['observation_size']],visibility='exact original observe() below',source_code=ast.get_source_segment(source,observe))]
        m['task']=dict(objective='Accumulate original per-agent rewards under the interaction rules in mechanism.md.',
            natural_termination='No early terminal; original t >= 32 horizon.',
            exact_transition_and_reward_source=ast.get_source_segment(source,step))
        m['release_metadata_revision']=1
        mp.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        row['manifest_sha256']=hashlib.sha256(mp.read_bytes()).hexdigest()
        provenance=json.loads((p/'provenance.json').read_bytes());provenance['manifest_sha256']=row['manifest_sha256']
        (p/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (S/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Documented source-derived commands and exact observation/reward code for all 72 legacy identities; kernels unchanged.')

if __name__=='__main__':main()
