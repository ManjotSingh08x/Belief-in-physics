"""Export offline 3D joint-probability views from verified saved predictions; no inference."""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def surface(flat):
    # Token = theta_bin * 50 + psi_bin; Plotly z rows are y (psi), columns x (theta).
    return np.asarray(flat).reshape(50, 50).T


PAGE = r'''<!doctype html><meta charset="utf-8">
<title>Sphere joint-angle probability</title>
<style>
body{font:16px system-ui;margin:24px;color:#17212b;background:#fafbfc}h1{font-size:23px;margin-bottom:8px}
p{max-width:1100px;line-height:1.45}label{margin-right:20px}input[type=range]{width:min(70vw,900px)}
#plot{height:70vh;min-height:500px}#status{font-variant-numeric:tabular-nums}button{padding:6px 12px}
</style>
<h1>Sphere: theta × azimuth joint probability</h1><p>__HEADING__</p>
<p>Predicting the joint token at <b>t+10</b> from the observed prefix through <b>t</b>.
The 500-token path has 490 directly supervised query positions (t=0…489).
Azimuth bins 49 and 0 are adjacent on the circle.</p>
<label>Source position <input id="position" type="range" min="0" max="489" value="15" step="1"></label>
<button id="play">Play</button><br><br>
<label><input id="zoom" type="checkbox" checked> Zoom height and colour to the shared peak probability</label>
<label><input id="actions" type="checkbox" checked> Show physical action endpoints</label>
<p id="status" aria-live="polite">Loading saved probabilities…</p><div id="plot"></div>
<p>Cyan marker: actual future joint token. Coloured markers: physical t+10 endpoints for four actions
at the next impulse, recomputed from the actual state. They are physical counterfactuals;
the neural surface is not a set of four mood-conditioned rollouts. Endpoints can occupy the same bin.</p>
<p>Drag to rotate; scroll to zoom. Both panels use the same probability scale. Values are the saved
float32 probabilities, without smoothing or peak creation. Teacher-forced diagnostic; one training seed.</p>
<script>__PLOTLY__</script><script>
const info=__INFO__;
function decode(s){const b=atob(s), a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return new Float32Array(a.buffer);}
const probabilities=[decode('__TRAINED__'),decode('__RANDOM__')];
const axis=Array.from({length:50},(_,i)=>i), graph=document.getElementById('plot');
const slider=document.getElementById('position'), zoom=document.getElementById('zoom'), actions=document.getElementById('actions');
let busy=false,timer=null;
function matrix(p){return axis.map(psi=>axis.map(theta=>p[theta*50+psi]));}
function markers(p,tokens,scene,actual){return {type:'scatter3d',mode:'markers',scene,
 x:tokens.map(v=>Math.floor(v/50)),y:tokens.map(v=>v%50),z:tokens.map(v=>p[v]),
 text:tokens.map((v,i)=>actual?'Actual token '+v:'Physical action '+i+' endpoint '+v),
 hovertemplate:'%{text}<br>theta bin %{x}<br>azimuth bin %{y}<br>P=%{z:.6g}<extra></extra>',
 marker:{size:actual?6:4,color:actual?'cyan':['orange','limegreen','red','royalblue'],symbol:actual?'circle':'diamond'},
 name:actual?'Actual future token':'Physical action endpoints',showlegend:scene==='scene',visible:actual||actions.checked};}
async function render(){
 if(busy)return;busy=true;
 try{
 const t=Number(slider.value), slices=probabilities.map(a=>a.subarray(t*2500,(t+1)*2500));
 const peak=Math.max(...slices[0],...slices[1]), maximum=zoom.checked?Math.min(1,peak*1.05):1;
 const traces=[];
 for(let i=0;i<2;i++){
  const scene=i?'scene2':'scene',p=slices[i];
  traces.push({type:'surface',scene,x:axis,y:axis,z:matrix(p),colorscale:[[0,'#000004'],[.25,'#51127c'],[.5,'#b73779'],[.75,'#fc8961'],[1,'#fcfdbf']],cmin:0,cmax:maximum,
   showscale:i===0,colorbar:{title:{text:'Joint probability'},x:1.01,len:.75},
   hovertemplate:'theta bin %{x}<br>azimuth bin %{y}<br>P=%{z:.6g}<extra></extra>'});
  traces.push(markers(p,[info.future[t]],scene,true),markers(p,info.physical[t],scene,false));
 }
 const scene={xaxis:{title:{text:'Theta bin'},range:[0,49]},yaxis:{title:{text:'Azimuth bin'},range:[0,49]},
  zaxis:{title:{text:'Joint probability'},range:[0,maximum]},aspectmode:'manual',aspectratio:{x:1,y:1,z:.8},
  camera:{eye:{x:1.5,y:1.5,z:1.1}},uirevision:'retain-camera'};
 await Plotly.react(graph,traces,{uirevision:'retain-camera',margin:{l:0,r:70,t:45,b:25},
  scene:{...scene,domain:{x:[0,.46],y:[0,1]}},scene2:{...scene,domain:{x:[.52,.98],y:[0,1]}},
  annotations:[{text:'Trained',x:.23,y:1.04,xref:'paper',yref:'paper',showarrow:false},
   {text:'Matched random',x:.75,y:1.04,xref:'paper',yref:'paper',showarrow:false}],
  legend:{orientation:'h',y:0,x:0}}, {responsive:true,displaylogo:false,modeBarButtonsToRemove:['sendChartToCloud'],toImageButtonOptions:{format:'png',width:1600,height:850}});
 const actual=info.future[t],unique=new Set(info.physical[t]).size;
 document.getElementById('status').textContent=`Path ${info.path} · source t=${t} → target ${t+10} · actual bin (${Math.floor(actual/50)}, ${actual%50}) · P(actual): trained ${slices[0][actual].toPrecision(4)}, random ${slices[1][actual].toPrecision(4)} · ${unique} distinct physical endpoint bins · height range 0–${maximum.toPrecision(4)}`;
 }catch(e){document.getElementById('status').textContent='Plot error: '+e.message;throw e;}finally{busy=false;}
}
slider.oninput=render;zoom.onchange=render;actions.onchange=render;
document.getElementById('play').onclick=function(){
 if(timer){clearInterval(timer);timer=null;this.textContent='Play';return;}
 this.textContent='Pause';timer=setInterval(()=>{if(!busy){slider.value=(Number(slider.value)+1)%490;render();}},300);
};
render();
</script>'''


def export(directory, output, plotly):
    evidence = json.loads((directory / 'analysis.json').read_text())
    model_name = evidence['model_name']
    if Path(model_name).name != model_name:
        raise ValueError('model name must be a single path component')
    verified = json.loads((directory / 'EXPORT_VERIFIED.json').read_text())
    data = []
    inputs = {}
    for tag in ('trained', 'random_init'):
        path = directory / f'{tag}_native_probabilities.npz'
        inputs[path.name] = digest(path)
        assert inputs[path.name] == verified['member_sha256'][path.name]
        with np.load(path, allow_pickle=False) as saved:
            values = {key: saved[key].copy() for key in saved.files}
        p = values['joint']
        assert p.shape == (4, 490, 2500) and int(values['k']) == 10
        assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
        assert np.max(np.abs(p.sum(-1) - 1)) < 1e-5
        assert np.array_equal(values['source_tokens'][:, 10:], values['future_tokens'])
        assert np.isin(values['future_tokens'], np.arange(2500)).all()
        assert (values['physical_action_counterfactual_tokens'] == values['future_tokens'][..., None]).any(-1).all()
        data.append(values)
    for key in ('trajectory_indices', 'source_tokens', 'future_tokens', 'physical_action_counterfactual_tokens'):
        assert np.array_equal(data[0][key], data[1][key]), key
    assert data[0]['trajectory_indices'].tolist() == evidence['protocol']['heatmap_paths']
    target = output / model_name
    target.mkdir(parents=True, exist_ok=True)
    script = plotly.read_text().replace('</script', r'<\/script')
    exp = evidence['experiment']
    heading = f"n={exp['n']} · gamma={exp['physics']['gamma']} · delta_v={exp['physics']['delta_v']} · alpha={exp['hmm']['alpha']} · 50×50 bins · 699.904M input tokens · seed0"
    pages = {}
    for i, trajectory in enumerate(data[0]['trajectory_indices']):
        info = {'path': int(trajectory), 'future': data[0]['future_tokens'][i].tolist(),
                'physical': data[0]['physical_action_counterfactual_tokens'][i].tolist()}
        page = PAGE.replace('__HEADING__', html.escape(heading)).replace('__PLOTLY__', script)
        page = page.replace('__INFO__', json.dumps(info))
        for tag, values in zip(('TRAINED', 'RANDOM'), data):
            raw = values['joint'][i].astype('<f4').tobytes()
            assert np.array_equal(np.frombuffer(raw, dtype='<f4').reshape(490, 2500), values['joint'][i])
            page = page.replace('__' + tag + '__', base64.b64encode(raw).decode())
        path = target / f'path-{int(trajectory)}.html'
        path.write_text(page)
        pages[path.name] = digest(path)
    manifest = {'model_name': model_name, 'input_sha256': inputs, 'html_sha256': pages,
                'plotly_sha256': digest(plotly), 'plotly_source': 'https://cdn.plot.ly/plotly-4.1.1.min.js',
                'script_sha256': digest(Path(__file__)), 'experiment': exp,
                'configuration_sha256': evidence['configuration_sha256'],
                'checkpoint_sha256': evidence['checkpoint_sha256'], 'axes': ['theta bin', 'azimuth bin', 'joint probability'],
                'query_positions': [0, 489], 'k': 10, 'training_context_tokens': 500,
                'probability_encoding': 'exact saved float32, little endian; no smoothing or normalization changes',
                'physical_markers': 'physical action endpoints, not neural mood-conditioned rollouts'}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    links = ''.join(f'<li><a href="{name}">Held-out path {name[5:-5]}</a></li>' for name in pages)
    (target / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Sphere joint 3D surfaces</h1><p>' + html.escape(heading) + '</p><ul>' + links + '</ul>')
    models = []
    for file in sorted(output.glob('*/manifest.json')):
        item = json.loads(file.read_text())
        setting = item['experiment']
        label = f"gamma={setting['physics']['gamma']}, delta_v={setting['physics']['delta_v']}, alpha={setting['hmm']['alpha']}"
        models.append(f'<li><a href="{html.escape(file.parent.name)}/index.html">{html.escape(label)}</a></li>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Sphere joint-angle 3D probabilities</h1><p>Choose a setting, then a held-out path. Each plot compares trained and matched random weights with a time slider.</p><ul>' + ''.join(models) + '</ul>')
    print(target / 'index.html')


def self_check():
    probability = np.zeros(2500, dtype=np.float32)
    probability[5 * 50 + 7] = 1
    image = surface(probability)
    assert image[7, 5] == 1 and image.sum() == 1
    assert np.array_equal(image.T.reshape(-1), probability)
    encoded = base64.b64encode(probability.astype('<f4').tobytes())
    assert np.array_equal(np.frombuffer(base64.b64decode(encoded), dtype='<f4'), probability)
    print('joint axis orientation and exact probability encoding verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plotly-js', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        if not all((args.analysis, args.output, args.plotly_js)):
            parser.error('--analysis, --output and --plotly-js are required')
        export(args.analysis, args.output, args.plotly_js)
