#!/usr/bin/env python3
"""Authoritative source for 21_gate_LRM05_evaluation_15organs.ipynb.

Build only; execute on Woody. The P8 diagnostic reconstructs one original outer-fold fit; deployment models remain frozen.
"""
from pathlib import Path
import nbformat

NAME = '21_gate_LRM05_evaluation_15organs'
CANDIDATE_THRESHOLD = 0.5
SUPPLEMENTARY_CANDIDATE_THRESHOLD = 0.2
CURRENT_CANDIDATES = r'''
# %% markdown
# Current exploratory candidates: main >= 0.5; supplementary 0.2 to < 0.5

The user-approved main candidate threshold is **0.5**, inclusive. Supplementary
candidates have scores **0.2 <= score < 0.5** and remain separate. The T/NK gate
is retained as review information and does **not** veto candidates. This rule
selects records for investigation; it does not establish gamma-delta identity.
Receptor conflicts remain flagged for review, and original annotations and
the earlier putative-label overlay are preserved without relabeling.

The self-contained cell below applies this rule to the saved SPL, PANGEA and
15-organ scores, without training or receptor reconstruction. It exports a
separate current candidate-selection table, summary and hashed manifest.
Its evidence flags are inherited from the saved threshold audit, not a new
integration of subsequent receptor audits. Consult those audits for later evidence.
PANGEA repeated atlas records are not necessarily distinct biological cells.

**Everything after this section is retained historical evaluation/diagnostics.**
The original 0.939474 threshold there is a comparison baseline, not the current
candidate cutoff. A clean Run All retains both views without overwriting the
historical model registry. These candidate rules were chosen after
examining these datasets; their results are not an independent validation of
this choice. Threshold changes do not retrain the model or establish NK specificity.

# %% code
import runpy
from pathlib import Path
from IPython.display import display, Markdown
PROJECT = next(p for p in [Path.cwd(), Path.cwd().parent]
               if (p/'build_21_gate_LRM05_evaluation_15organs.py').exists())
candidate_selection = runpy.run_path(str(PROJECT/'build_21_gate_LRM05_evaluation_15organs.py'))['run_gate_discordance_audit_candidates'](PROJECT)
display(Markdown('**Main: score >= 0.5; supplementary: 0.2 <= score < 0.5. Gate is review only.**'))
display(candidate_selection['summary'])
# %% markdown
## Candidate UMAP, expression and evidence review

The following review reuses all saved UMAP coordinates and normalized expression.
Rows separate main and supplementary candidates; columns show location, gate
state and available receptor evidence. A second view highlights the existing
source NK-like scope, which differs between PANGEA and 15-organ and is not a truth
label. The 15-organ T/NK view uses the previous annotation/receptor-based scope,
independent of either model's predictions. Missing coordinates remain in tables.

Expression summaries are descriptive: shared classifier genes cannot serve as
independent validation. Source-study/sample/organ and gate-stratified tables
retain heterogeneity; no cell-level significance tests imply donor replication.
Previously completed GSE178991/GSE200173 receptor searches and the bounded V8
reconstruction are linked by dataset and exact cell ID, without rerunning them.
Their evidence types and quality remain separate; absent receptors mean unresolved.
The old putative labels and all original fields are unchanged.

# %% code
from IPython.display import Image
candidate_review = runpy.run_path(str(PROJECT/'build_21_gate_LRM05_evaluation_15organs.py'))['run_gate_discordance_audit_candidate_review'](PROJECT)
display(Markdown(candidate_review['interpretation']))
for preview in candidate_review['previews']:
    display(Image(filename=preview))
'''


def run_gate_discordance_audit_candidates(project):
    """Apply the approved candidate rule; preserve all existing evidence fields."""
    import hashlib, inspect, json, os, platform
    import pandas as pd
    from datetime import datetime, timezone
    project = Path(project).resolve()
    prefix = 'gate_discordance_audit_candidate'
    pred = project/'predictions'/NAME
    table = project/'results/tables'/NAME
    source = pred/'gate_discordance_audit_threshold_cells.parquet'
    params_path = table/'parameters.json'
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def save(value, path):
        temporary = Path(str(path)+'.tmp')
        if path.suffix == '.parquet': value.to_parquet(temporary, index=False)
        elif path.suffix == '.csv': value.to_csv(temporary, index=False)
        else: temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
        os.replace(temporary, path)
    inputs = {str(p.relative_to(project)): sha(p) for p in [source, params_path]}
    data = pd.read_parquet(source)
    assert not data.duplicated(['dataset', 'cell_id']).any()
    assert set(data.dataset) == {'SPL', 'PANGEA', '15-organ'}
    assert data.p_gdT_given_T.notna().all() and data.p_gdT_given_T.between(0, 1).all()
    params = json.loads(params_path.read_text())
    assert data.gate_T_eligible.eq(data.p_T.ge(params['threshold_T'])).all()
    assert data.gate_NK_call.eq(data.p_T.le(params['threshold_NK'])).all()
    assert data.gate_abstain.eq(~(data.gate_T_eligible | data.gate_NK_call)).all()
    assert data[['gate_T_eligible', 'gate_NK_call', 'gate_abstain']].sum(axis=1).eq(1).all()
    # These additional fields cannot change original labels or historical calls.
    data['gate_discordance_audit_candidate_threshold'] = CANDIDATE_THRESHOLD
    data['gate_discordance_audit_candidate'] = data.p_gdT_given_T.ge(CANDIDATE_THRESHOLD)
    data['gate_discordance_audit_supplementary_candidate'] = (
        data.p_gdT_given_T.ge(SUPPLEMENTARY_CANDIDATE_THRESHOLD) &
        data.p_gdT_given_T.lt(CANDIDATE_THRESHOLD))
    data['gate_discordance_audit_candidate_receptor_conflict_review'] = (
        (data.gate_discordance_audit_candidate | data.gate_discordance_audit_supplementary_candidate) &
        (data.threshold_audit_ab_receptor_support | data.threshold_audit_mixed_conflict))
    rows = []
    for dataset, group in data.groupby('dataset', sort=True, observed=True):
      for band, field, lower, upper in [
          ('Main', 'gate_discordance_audit_candidate', CANDIDATE_THRESHOLD, None),
          ('Supplementary', 'gate_discordance_audit_supplementary_candidate', SUPPLEMENTARY_CANDIDATE_THRESHOLD, CANDIDATE_THRESHOLD)]:
        selected = group[field]
        gd = group.threshold_audit_gd_receptor_support
        ab = group.threshold_audit_ab_receptor_support
        rows.append({'dataset': dataset, 'candidate_group': band,
            'threshold_lower_inclusive': lower, 'threshold_upper_exclusive': upper,
            'n_records': len(group), 'n_candidates': int(selected.sum()),
            'gate_T_eligible_candidates': int((selected & group.gate_T_eligible).sum()),
            'gate_NK_call_candidates': int((selected & group.gate_NK_call).sum()),
            'gate_abstain_candidates': int((selected & group.gate_abstain).sum()),
            'gd_support_denominator': int(gd.sum()),
            'gd_support_selected': int((selected & gd).sum()),
            'gd_support_retention': float((selected & gd).sum()/gd.sum()) if gd.sum() else None,
            'ab_support_denominator': int(ab.sum()),
            'ab_support_selected': int((selected & ab).sum())})
    summary = pd.DataFrame(rows)
    assert (summary[['gate_T_eligible_candidates', 'gate_NK_call_candidates',
                     'gate_abstain_candidates']].sum(axis=1) == summary.n_candidates).all()
    outputs = [pred/f'{prefix}_cells.parquet', table/f'{prefix}_summary.csv',
               table/f'{prefix}_parameters.json']
    config = {'model': 'optimized_M05', 'score': 'p_gdT_given_T',
              'threshold': CANDIDATE_THRESHOLD, 'comparison': '>=',
              'supplementary_lower_inclusive': SUPPLEMENTARY_CANDIDATE_THRESHOLD,
              'supplementary_upper_exclusive': CANDIDATE_THRESHOLD,
              'gate_is_filter': False, 'purpose': 'exploratory candidate selection',
              'threshold_origin': 'user-approved after evaluation-data inspection',
              'no_refit': True, 'original_annotations_preserved': True,
              'receptor_evidence_scope': 'inherited saved threshold-audit snapshot',
              'inputs': inputs, 'python': platform.python_version(), 'pandas': pd.__version__,
              'function_sha256': hashlib.sha256(inspect.getsource(run_gate_discordance_audit_candidates).encode()).hexdigest()}
    config['signature'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    save(data, outputs[0]); save(summary, outputs[1]); save(config, outputs[2])
    assert all(sha(project/p) == h for p, h in inputs.items())
    local = Path('/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/gdt_lr_12model')
    remote = Path('/data/users/kaelynwen/gdT_project/gdt_lr_12model')
    manifest = {'complete': True, 'signature': config['signature'],
                'completed_utc': datetime.now(timezone.utc).isoformat(),
                'outputs': [{'local_path': str(local/p.relative_to(project)),
                             'Woody_path': str(remote/p.relative_to(project)),
                             'sha256': sha(p), 'bytes': p.stat().st_size} for p in outputs],
                'resume': "python -c \"import runpy; from pathlib import Path; runpy.run_path('build_21_gate_LRM05_evaluation_15organs.py')['run_gate_discordance_audit_candidates'](Path.cwd())\""}
    save(manifest, table/f'{prefix}_manifest.json')
    return {'summary': summary}



def run_gate_discordance_audit_candidate_review(project):
    """Review approved disjoint candidate bands on unchanged atlas coordinates."""
    import hashlib, inspect, json, os, sys, subprocess
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    project=Path(project).resolve();prefix='gate_discordance_audit_candidate'
    pred=project/'predictions'/NAME;table=project/'results/tables'/NAME
    figdir=project/'results/figures'/NAME;check=project/'checkpoints'/NAME;log=project/'logs'/NAME
    for folder in [figdir,check,log]:folder.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(check))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    source=pred/f'{prefix}_cells.parquet'
    evidence_specs=[
        (pred/'gate_discordance_audit_GSE178991_cells.parquet','GSE178991',
         ['receptor_search_matched_record','receptor_search_patient_timepoint_verified',
          'receptor_search_patient_timepoint_conflict','receptor_search_paired_ab_annotation',
          'receptor_search_any_ab_annotation','receptor_search_contig_quality_available']),
        (pred/'GSE200173_gate_discordance_audit_receptor_search_cells.parquet','GSE200173',
         ['receptor_search_matched_record','receptor_search_paired_ab','receptor_search_paired_gd']),
        (table/'gate_discordance_audit_GSE178991_V8_SRR14925077_reconstruction_cells.csv','SRR14925077',
         ['reconstruction_selected_spots','reconstruction_paired_ab_2_Q20_UMIs',
          'reconstruction_paired_gd_2_Q20_UMIs','reconstruction_mixed_ab_gd'])]
    inputs=[source,table/f'{prefix}_parameters.json']+[x[0] for x in evidence_specs]
    inputs += [pred/f'{ds}_gate_discordance_audit_UMAP_source_data.parquet' for ds in ['PANGEA','15-organ']]
    inputs += [check/f for f in ['audit_panel_alignment.py','audit_pdf_text.py','audit_figure_collisions.py']]
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def save(v,p):
        tmp=Path(str(p)+'.tmp')
        if p.suffix=='.parquet':v.to_parquet(tmp,index=False)
        elif p.suffix=='.csv':v.to_csv(tmp,index=False)
        elif isinstance(v,str):tmp.write_text(v)
        else:tmp.write_text(json.dumps(v,indent=2,sort_keys=True))
        os.replace(tmp,p);return p
    available={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ['Arial','Helvetica','Liberation Sans'] if f in available),None)
    assert font,'An approved font is required'
    plt.rcParams.update({'font.family':font,'font.size':7,'font.weight':'normal',
        'axes.titleweight':'normal','axes.labelweight':'normal','pdf.fonttype':42,'svg.fonttype':'none',
        'text.color':'black','axes.labelcolor':'black','axes.titlecolor':'black',
        'xtick.color':'black','ytick.color':'black','legend.labelcolor':'black',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6})
    contract={'inputs':{str(p.relative_to(project)):sha(p) for p in inputs},
        'function_sha256':hashlib.sha256(inspect.getsource(run_gate_discordance_audit_candidate_review).encode()).hexdigest(),
        'main_lower_inclusive':CANDIDATE_THRESHOLD,'supplementary_lower_inclusive':SUPPLEMENTARY_CANDIDATE_THRESHOLD,
        'supplementary_upper_exclusive':CANDIDATE_THRESHOLD,'gate_is_filter':False,
        'font':font,'python':sys.version,'pandas':pd.__version__,'numpy':np.__version__,
        'matplotlib':matplotlib.__version__,'coordinates':'unchanged; no sampling',
        'figure_question':'Where are main and supplementary candidates, and what evidence supports review?',
        'archetype':'quantitative grid','figure_roles':['candidate location','gate state','receptor evidence'],
        'statistics':'descriptive; atlas records not independent biological replicates; no P values'}
    signature=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    mp=check/f'{prefix}_review_manifest.json';prior=json.loads(mp.read_text()) if mp.exists() else {}
    manifest={'signature':signature,'contract':contract,'complete':False,'units':{}}
    save(manifest,mp)
    def valid(unit):
        u=prior.get('units',{}).get(unit,{})
        if prior.get('signature')==signature and u.get('outputs') and all((project/p).exists() and sha(project/p)==h for p,h in u['outputs'].items()):
            manifest['units'][unit]=u;save(manifest,mp);return True
        return False
    def done(unit,paths):
        manifest['units'][unit]={'outputs':{str(p.relative_to(project)):sha(p) for p in paths}}
        save(manifest,mp);print('Saved',unit,flush=True)
    def export(fig,stem):
        fig.canvas.draw();alignment=log/f'{stem}_alignment.json'
        require_matplotlib_panel_alignment(fig,json_out=alignment,tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
        outputs=[alignment]
        for ext in ['pdf','svg','png']:
            path=figdir/f'{stem}.{ext}';tmp=figdir/f'{stem}.tmp.{ext}'
            fig.savefig(tmp,format=ext,dpi=300,facecolor='white');os.replace(tmp,path);outputs.append(path)
        plt.close(fig)
        for tool,tag,args in [('audit_pdf_text.py','text_audit',['--min-pt','5','--json']),
                              ('audit_figure_collisions.py','collision_audit',['--json'])]:
            r=subprocess.run([sys.executable,str(check/tool),str(figdir/f'{stem}.pdf'),*args],capture_output=True,text=True)
            path=log/f'{stem}_{tag}.json';save(r.stdout,path);outputs.append(path)
            if r.returncode:raise RuntimeError(f'{stem} {tool}: {r.returncode}\n{r.stdout}\n{r.stderr}')
        return outputs
    d=pd.read_parquet(source)
    params=json.loads((table/f'{prefix}_parameters.json').read_text())
    assert params['threshold']==CANDIDATE_THRESHOLD and params['supplementary_lower_inclusive']==SUPPLEMENTARY_CANDIDATE_THRESHOLD
    d['candidate_group']=np.select([d.gate_discordance_audit_candidate,d.gate_discordance_audit_supplementary_candidate],['Main','Supplementary'],default='Below 0.2')
    d['gate_state']=np.select([d.gate_T_eligible,d.gate_NK_call],['T-eligible','NK-call'],default='abstain')
    for path,study,fields in evidence_specs:
        e=pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path)
        assert e.cell_id.is_unique and d.loc[d.dataset.eq('PANGEA'),'cell_id'].is_unique
        e=e[['cell_id']+fields].copy();e['dataset']='PANGEA'
        assert set(e.cell_id)<=set(d.loc[d.dataset.eq('PANGEA'),'cell_id'])
        e=e.rename(columns={f:study+'_'+f for f in fields})
        d=d.merge(e,on=['dataset','cell_id'],how='left',validate='one_to_one',sort=False)
    def yes(c):return d[c].eq(True).fillna(False)
    d['candidate_review_gd_evidence']=yes('threshold_audit_gd_receptor_support')|yes('GSE200173_receptor_search_paired_gd')|yes('SRR14925077_reconstruction_paired_gd_2_Q20_UMIs')
    d['candidate_review_ab_evidence']=yes('threshold_audit_ab_receptor_support')|yes('GSE200173_receptor_search_paired_ab')|yes('SRR14925077_reconstruction_paired_ab_2_Q20_UMIs')|(yes('GSE178991_receptor_search_paired_ab_annotation')&yes('GSE178991_receptor_search_patient_timepoint_verified'))
    d['candidate_review_mixed_conflict']=yes('threshold_audit_mixed_conflict')|yes('SRR14925077_reconstruction_mixed_ab_gd')|(d.candidate_review_ab_evidence&d.candidate_review_gd_evidence)
    d['candidate_review_receptor_conflict']=d.candidate_review_ab_evidence|d.candidate_review_mixed_conflict
    d['candidate_review_unresolved']=~(d.candidate_review_gd_evidence|d.candidate_review_ab_evidence)
    d['candidate_review_has_coordinates']=False;d['UMAP1']=np.nan;d['UMAP2']=np.nan
    for ds in ['PANGEA','15-organ']:
        mask=d.dataset.eq(ds);xy=pd.read_parquet(pred/f'{ds}_gate_discordance_audit_UMAP_source_data.parquet').set_index('cell_id')
        assert xy.index.is_unique and set(xy.index)==set(d.loc[mask,'cell_id'])
        xy=xy.reindex(d.loc[mask,'cell_id'])
        np.testing.assert_allclose(d.loc[mask,['p_T','p_gdT_given_T']],xy[['p_T','p_gdT_given_T']],rtol=0,atol=1e-12)
        d.loc[mask,['UMAP1','UMAP2']]=xy[['UMAP1','UMAP2']].to_numpy()
        d.loc[mask,'candidate_review_has_coordinates']=np.isfinite(xy[['UMAP1','UMAP2']]).all(axis=1).to_numpy()
    selected=d[d.candidate_group.isin(['Main','Supplementary'])].copy()
    flags=['source_annotation_NK','source_annotation_gdT','candidate_review_gd_evidence','candidate_review_ab_evidence','candidate_review_mixed_conflict','candidate_review_receptor_conflict','candidate_review_unresolved','candidate_review_has_coordinates']
    for f in flags:selected[f]=selected[f].eq(True).fillna(False)
    def counts(keys):return selected.groupby(keys,dropna=False,observed=True).agg(n_records=('cell_id','size'),**{f:(f,'sum') for f in flags}).reset_index()
    summary=counts(['dataset','candidate_group','gate_state'])
    genes=['CD3D','CD3E','CD3G','TRDC','TRGC1','TRGC2','TRAC','TRBC1','TRBC2','KLRD1','FCER1G','TYROBP','NKG7','GNLY']
    features=['log1p_'+g for g in genes]+['audit_total_counts','audit_n_genes_by_counts','audit_pct_counts_mt']
    stats=[]
    for ds,z in selected.groupby('dataset',observed=True):
      for scope,v in [('All candidates',z),('Source NK-like',z[z.source_annotation_NK])]:
       for (band,gate),g in v.groupby(['candidate_group','gate_state'],observed=True):
        strata=[('all','all','all','all',g)]+[(str(a),str(b),str(c),str(e),w) for (a,b,c,e),w in g.groupby(['source_study','source_sample','source_organ','source_annotation'],dropna=False,observed=True)]
        for study,sample,organ,annotation,w in strata:
         for f in features:
            x=pd.to_numeric(w[f],errors='coerce').dropna()
            if len(x):stats.append({'dataset':ds,'scope':scope,'candidate_group':band,'gate_state':gate,'source_study':study,'source_sample':sample,'source_organ':organ,'source_annotation':annotation,'feature':f,'n_measured':len(x),'mean':float(x.mean()),'median':float(x.median()),'fraction_detected':float(x.gt(0).mean()) if f.startswith('log1p_') else None})
    stats=pd.DataFrame(stats)
    if not valid('tables'):
        paths=[save(d,pred/f'{prefix}_review_cells.parquet'),save(summary,table/f'{prefix}_review_summary.csv'),
            save(counts(['dataset','source_study','source_sample','source_organ','source_annotation','candidate_group','gate_state']),table/f'{prefix}_review_by_source.csv'),
            save(selected[selected.candidate_review_receptor_conflict],pred/f'{prefix}_receptor_conflict_review.parquet'),
            save(stats,table/f'{prefix}_expression_QC.csv')]
        done('tables',paths)
    gate_colors={'T-eligible':'#7B2CBF','NK-call':'#D73027','abstain':'#E69F00'}
    previews=[];plot_counts=[]
    for ds in ['PANGEA','15-organ']:
        a=d[d.dataset.eq(ds)];finite=a[a.candidate_review_has_coordinates]
        views=[('full',finite,False),('source_NK_like',finite,True)]
        if ds=='15-organ':
            scope=a.annotation_Tlineage.eq(True)|a.original_annotation_T_NK_ambiguous.eq(True)|a.strict_paired_abTCR.eq(True)
            views.append(('T_NK',a[scope&a.candidate_review_has_coordinates],False))
        for view,v,nk_only in views:
            stem=f'{ds}_{prefix}_{view}_UMAP';previews.append(str(figdir/f'{stem}.png'))
            for band in ['Main','Supplementary']:
                inband=a.candidate_group.eq(band)&(a.source_annotation_NK.eq(True) if nk_only else True)
                plot_counts.append({'dataset':ds,'view':view,'candidate_group':band,'n_candidates_dataset_scope':int(inband.sum()),'n_missing_coordinates_dataset_scope':int((inband&~a.candidate_review_has_coordinates).sum()),'n_background_plotted':len(v),'n_candidates_plotted':int((v.candidate_group.eq(band)&(v.source_annotation_NK.eq(True) if nk_only else True)).sum())})
            if valid(stem):continue
            fig,axs=plt.subplots(2,3,figsize=(7.2,5.8));fig.subplots_adjust(left=.075,right=.98,bottom=.22,top=.85,wspace=.3,hspace=.45)
            lim=[(v[c].min()-.025*np.ptp(v[c]),v[c].max()+.025*np.ptp(v[c])) for c in ['UMAP1','UMAP2']]
            for row,(band,label,color) in enumerate([('Main','Main: score >= 0.5','#7B2CBF'),('Supplementary','Supplementary: 0.2 to < 0.5','#0072B2')]):
                z=v[v.candidate_group.eq(band)&(v.source_annotation_NK.eq(True) if nk_only else True)]
                for ax in axs[row]:
                    ax.scatter(v.UMAP1,v.UMAP2,s=.35,c='#D9D9D9',alpha=.4,lw=0)
                    ax.set(xlim=lim[0],ylim=lim[1],xlabel='UMAP1',ylabel='UMAP2',xticks=[],yticks=[])
                    ax.set_aspect('equal',adjustable='box')
                axs[row,0].scatter(z.UMAP1,z.UMAP2,s=2,c=color,alpha=.8,lw=0)
                axs[row,0].set_title(label+'\n'+f'n = {len(z):,} records',fontsize=7)
                for state,col in gate_colors.items():
                    q=z[z.gate_state.eq(state)];axs[row,1].scatter(q.UMAP1,q.UMAP2,s=2,c=col,alpha=.8,lw=0)
                axs[row,1].set_title('Gate state; no exclusion')
                axs[row,2].scatter(z.UMAP1,z.UMAP2,s=1.5,c=color,alpha=.25,lw=0)
                for flag,marker,col in [('candidate_review_gd_evidence','o','black'),('candidate_review_ab_evidence','x','#0072B2')]:
                    q=z[z[flag]]
                    marker_style={'facecolors':'none','edgecolors':col} if marker=='o' else {'c':col}
                    axs[row,2].scatter(q.UMAP1,q.UMAP2,s=10,marker=marker,lw=.6,**marker_style)
                axs[row,2].set_title('Available receptor evidence')
            fig.suptitle(ds+' | '+view.replace('_',' ')+' | optimized_M05',fontsize=9,y=.96)
            handles=[Line2D([],[],marker='o',ls='',color=c,label=s,markersize=4) for s,c in gate_colors.items()]
            handles += [Line2D([],[],marker='o',ls='',color='black',markerfacecolor='none',label='TRG/TRD evidence',markersize=4),Line2D([],[],marker='x',ls='',color='#0072B2',label='TRA/TRB evidence / review',markersize=4)]
            fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.065),ncol=3,frameon=False,fontsize=6.5)
            fig.text(.075,.025,'Same saved coordinates; evidence quality varies. Missing TCR does not establish NK identity.',fontsize=6)
            done(stem,export(fig,stem))
        # Expression is shown within the source NK-like scope, separately by gate.
        stem=f'{ds}_{prefix}_source_NK_like_expression';previews.append(str(figdir/f'{stem}.png'))
        if not valid(stem):
            q=stats[(stats.dataset==ds)&(stats.scope=='Source NK-like')&(stats.source_sample=='all')&stats.feature.str.startswith('log1p_')]
            row_keys=[(band,gate) for band in ['Main','Supplementary'] for gate in gate_colors]
            fig,ax=plt.subplots(figsize=(7.2,3.8));fig.subplots_adjust(left=.28,right=.86,bottom=.34,top=.81)
            vmax=max(float(q['mean'].max()),.01)
            for yi,(band,gate) in enumerate(row_keys):
                t=q[(q.candidate_group==band)&(q.gate_state==gate)].set_index('feature').reindex(['log1p_'+g for g in genes])
                ax.scatter(range(len(genes)),[yi]*len(genes),s=60*t.fraction_detected,c=t['mean'],vmin=0,vmax=vmax,cmap='viridis',edgecolors='none')
            ax.set(xticks=range(len(genes)),xticklabels=genes,yticks=range(len(row_keys)),yticklabels=[b+' | '+g for b,g in row_keys],ylim=(5.6,-.6),xlim=(-.6,len(genes)-.4))
            plt.setp(ax.get_xticklabels(),rotation=90,ha='right',rotation_mode='anchor')
            cax=fig.add_axes([.89,.38,.015,.32]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,vmax),cmap='viridis'),cax=cax);cb.set_label('Mean log1p expression',fontsize=6.5);cb.ax.tick_params(labelsize=6)
            cax.set_label('<colorbar>')  # Auxiliary scale, not a second comparable data panel.
            cb.solids.set_rasterized(False)
            fig.suptitle(ds+' | source NK-like candidates | descriptive expression',fontsize=9,y=.95)
            size_handles=[ax.scatter([],[],s=60*x,c='black',edgecolors='none',label=f'{int(100*x)}%') for x in [.25,.5,1.]]
            fig.legend(handles=size_handles,title='Fraction with detected transcript',loc='lower center',bbox_to_anchor=(.56,.055),ncol=3,frameon=False,fontsize=6.5,title_fontsize=6.5)
            fig.text(.03,.025,'RNA overlaps classifier inputs; expression is not independent identity validation. No cell-level tests.',fontsize=6)
            done(stem,export(fig,stem))
    counts_path=save(pd.DataFrame(plot_counts),table/f'{prefix}_UMAP_counts.csv');done('plot_counts',[counts_path])
    interpretation='''## Main and supplementary candidates remain separate

Main candidates use optimized_M05 >= 0.5. Supplementary candidates use 0.2 <= score < 0.5. All three gate states remain eligible for review. No original annotation or historical putative label is changed.

The source NK-like definition is inherited: PANGEA NK_* annotations and the previous 15-organ T/NK-ambiguous scope are not equivalent truth sets. PANGEA repeated run records are not necessarily distinct biological cells. Tables retain exact IDs, source study, sample, organ, original annotation and gate state. The 15-organ T/NK view uses the previously established annotation/receptor scope, independent of the model calls.

Receptor evidence combines the saved threshold-audit flags with already completed GSE200173 searches, membership-verified GSE178991 processed alpha/beta annotations, and the bounded SRR14925077 reconstruction. Original evidence columns remain separate: processed annotations lack full contig-quality flags; raw reconstructed pairs require the existing non-imputed, two-Q20-UMI-per-chain rule. These categories are review evidence, not interchangeable gold labels. Receptor conflicts are exported separately without removing candidates. Missing receptors remain unresolved; 15-organ lacks a gamma/delta-positive receptor reference. No receptor search or assembly was repeated.

UMAP overlays use unchanged coordinates, no refit or sampling; missing coordinates remain in the tables and the count report. Figures are descriptive, with vector PDF marks and editable black text. Expression dots use mean saved normalized log1p expression and fraction detected within each candidate/gate group. Individual source-unit summaries are saved; pooled dots can reflect donor/study mixture and classifier-feature circularity. No significance test treats cells as independent donors. These plots do not establish that gate NK-call candidates are true NK or true gamma-delta T cells.

This is a conservative Nature-style review set, not a claim of journal-specific compliance. The two candidate thresholds were chosen after inspecting evaluation data; this is exploratory, not a new independent validation. Each table/figure unit is atomically saved, hashed and reused only for matching input, configuration, schema-version-in-code and environment signatures. Resume by calling run_gate_discordance_audit_candidates(Path.cwd()) and run_gate_discordance_audit_candidate_review(Path.cwd()) from the authoritative generator. The completion manifest is written only after every required figure and table is present.
'''
    report=save(interpretation,table/f'{prefix}_interpretation.md');done('interpretation',[report])
    assert all(sha(project/p)==h for p,h in contract['inputs'].items())
    manifest['complete']=True;save(manifest,mp)
    paths=[project/p for u in manifest['units'].values() for p in u['outputs']]+[mp,project/'build_21_gate_LRM05_evaluation_15organs.py']
    local=Path('/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/gdt_lr_12model');remote=Path('/data/users/kaelynwen/gdT_project/gdt_lr_12model')
    save(pd.DataFrame([{'local_path':str(local/p.relative_to(project)),'Woody_path':str(remote/p.relative_to(project)),'sha256':sha(p)} for p in paths]),table/f'{prefix}_output_manifest.csv')
    return {'interpretation':interpretation,'previews':previews}


RUNTIME = r'''
# %% markdown
# 21_gate_LRM05_evaluation_15organs

## Question and frozen comparison
Compare the original M05 evaluation with an unchanged M05 child plus the frozen
T/NK LR root. Historical **controlled_M05** remains a reference. The intervention
comparison uses **optimized_M05**, **CE–M05** (`P(T) × P(gdT|T)`) and
**LR gate + optimized_M05** (the frozen hard decision).
CE–M05 is a factorization of two existing LR probabilities, not a newly trained
joint neural-network loss. A hard decision supplies a PR/ROC point, not a third
continuous score. No CE operating threshold is invented here.

Historical cell sets, evaluation labels, and UMAP coordinates remain fixed.
Song uses saved held-out-donor child predictions; the independent GSE164378 root
is applied to those cells. SPL reuses its frozen receptor-evidence cohorts.
PANGEA and 15-organ reuse hash-verified, complete notebook-22 predictions.
No fitting, calibration, feature selection, threshold optimization, or UMAP
retraining is performed on these evaluation data. Gate-excluded candidates are
retained in the exported cell tables.

## Figure contract and interpretation
PR is above ROC. Curves compare identical cells within each endpoint. Three
continuous scores are shown; the frozen gated rule is a black diamond. AP is
average precision, not trapezoidal PR area. All abstentions remain in the
denominator and count as non-positive for end-to-end gdT detection.
PANGEA labels are annotation concordance; 15-organ uses the original hybrid
annotation/receptor endpoint. Neither is universal cellular ground truth.
An NK-like annotation or a root NK call does not prove NK identity. UMAP is
descriptive geometry, not an independent accuracy test. Organs from one donor
are not independent biological replicates. No cell-bootstrap confidence interval
or significance test is used to imply independent donor replication.

Figures use normal black Arial/Helvetica/Liberation Sans text, editable vector
PDF text and vector points, plus PNG previews. This is a conservative Nature-style
layout; exact journal compliance requires a selected journal and its guidelines.
# %% code
from pathlib import Path
from datetime import datetime, timezone
import os, sys, json, hashlib, platform, importlib.metadata, string, subprocess
import numpy as np
import pandas as pd
import anndata as ad
import joblib
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from sklearn.metrics import (precision_recall_curve, roc_curve, roc_auc_score,
                             average_precision_score, brier_score_loss)
from IPython.display import display, Image, Markdown

ANALYSIS_ID = '21_gate_LRM05_evaluation_15organs'
REMOTE = Path('/data/users/kaelynwen/gdT_project/gdt_lr_12model')
LOCAL = Path('/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/gdt_lr_12model')
PROJECT = Path(os.environ.get('GDT_LR_PROJECT', str(REMOTE if REMOTE.exists() else LOCAL)))
TABLE = PROJECT/'results/tables'/ANALYSIS_ID
FIGURE = PROJECT/'results/figures'/ANALYSIS_ID
PREDICTION = PROJECT/'predictions'/ANALYSIS_ID
CHECKPOINT = PROJECT/'checkpoints'/ANALYSIS_ID
LOG = PROJECT/'logs'/ANALYSIS_ID
for directory in (TABLE, FIGURE, PREDICTION, CHECKPOINT, LOG):
    directory.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CHECKPOINT))
from audit_panel_alignment import require_matplotlib_panel_alignment
sys.path.insert(0, str(PROJECT/'src'))
from gdt_lr_utils import predict_lr
np.random.seed(17)
font = next((f for f in ('Arial','Helvetica','Liberation Sans')
             if f in {x.name for x in font_manager.fontManager.ttflist}), None)
if font is None: raise RuntimeError('A permitted publication font is required')
plt.rcParams.update({'font.family':font, 'font.size':7, 'font.weight':'normal',
    'axes.titlesize':7, 'axes.titleweight':'normal', 'axes.labelsize':7,
    'xtick.labelsize':6, 'ytick.labelsize':6, 'legend.fontsize':6,
    'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none', 'text.color':'black',
    'axes.labelcolor':'black','axes.titlecolor':'black','xtick.color':'black',
    'ytick.color':'black','legend.labelcolor':'black', 'axes.spines.top':False,
    'axes.spines.right':False, 'axes.linewidth':.6, 'lines.linewidth':1})
INPUTS = {}
def sha(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()
def track(path, expected=None):
    path = Path(path)
    key = str(path)
    if key not in INPUTS: INPUTS[key] = sha(path)
    if expected is not None and INPUTS[key] != expected:
        raise ValueError('Input hash mismatch: '+key)
    return INPUTS[key]
def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()
def ids_hash(ids):
    return hashlib.sha256(json.dumps(list(map(str,ids)),ensure_ascii=False).encode()).hexdigest()
def atomic_json(value, path):
    tmp=Path(str(path)+'.tmp'); tmp.write_text(json.dumps(value,indent=2,default=str))
    os.replace(tmp,path)
def atomic_frame(frame, path):
    path=Path(path);tmp=Path(str(path)+'.tmp')
    if path.suffix=='.parquet': frame.to_parquet(tmp,index=False)
    else: frame.to_csv(tmp,index=False)
    os.replace(tmp,path)
SOURCE_HASH=track(PROJECT/f'build_{ANALYSIS_ID}.py')
track(CHECKPOINT/'audit_panel_alignment.py')
environment={'python':sys.version,'platform':platform.platform(),'font':font,
    'packages':{x:importlib.metadata.version(x) for x in
      ['numpy','pandas','anndata','scipy','scikit-learn','matplotlib','joblib','nbformat','nbclient']}}
atomic_json(environment,LOG/'environment.json')
MODELS={name:joblib.load(PROJECT/f'models/{name}.joblib') for name in ['controlled_M05','optimized_M05']}
GATE_PATH=PROJECT/'models/21_T_NK_lineage_gate_and_frozen_M05_deployment/21_T_NK_lineage_gate_and_frozen_M05_deployment_frozen_lineage_gate.joblib'
GATE=joblib.load(GATE_PATH)
MODEL_HASH={name:track(PROJECT/f'models/{name}.joblib') for name in MODELS}
MODEL_HASH['gate']=track(GATE_PATH)
THRESHOLDS={}
for name in MODELS:
    path=PROJECT/f'results/tables/{name.split("_")[0]}_fit_registry.csv'
    # Preserve the saved binary float: ordinary CSV parsing can move an exact
    # OOF threshold tie by one ULP and change one source operating-point call.
    track(path); rows=pd.read_csv(path,float_precision='round_trip').query('model_id == "M05"')
    assert len(rows)==1
    THRESHOLDS[name]=float(rows.iloc[0].threshold)
for bundle in [*MODELS.values(),GATE]: assert list(bundle['model'].classes_)==[0,1]
PARAMETERS={'thresholds':THRESHOLDS,'threshold_T':GATE['threshold_T'],
    'threshold_NK':GATE['threshold_NK'],'minimum_feature_coverage':.95,
    'random_seed':17,'model_hashes':MODEL_HASH,'no_refit':True,'no_external_tuning':True}
atomic_json(PARAMETERS,TABLE/'parameters.json')
display(pd.DataFrame([PARAMETERS]))
# %% markdown
## Checkpoint and feature contract
Each new Song/SPL donor is the smallest restart unit. Its parquet checkpoint
is reused only if the source code, model hashes, raw-input hash, cell ordering,
schema and checkpoint checksum agree. Writes are atomic; an interrupted donor
is recomputed, completed donors are retained. Re-run this notebook from the
first cell to resume. A completion manifest tracks every donor.

From a Woody terminal, the exact resume command is:
```bash
cd /data/users/kaelynwen/gdT_project/gdt_lr_12model
/data/conda_envs/kaelynwen/ENTER/bin/python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/21_gate_LRM05_evaluation_15organs.ipynb
```

Panel counts follow notebook 22 exactly: stable training gene IDs first, exact
symbols second, duplicate target features summed, explicit zeros for missing
features; normalize over the frozen panel to 10,000 and log1p. Feature reuse
across two model genes is an error. Historical SPL controlled_M05 is separately
reproduced using its original utility and required to match saved scores.
# %% code
SONG_PATH=PROJECT/'manifests/Song_D1_cohorts.h5ad'
track(SONG_PATH)
song=ad.read_h5ad(SONG_PATH,backed='r')
GENE_IDS=song.var.gene_ids.astype(str).to_dict()
def score(bundle, target, reference=None):
    symbols={};stable={}
    for i,s in enumerate(target.var_names.astype(str)): symbols.setdefault(s,[]).append(i)
    if 'gene_ids' in target.var:
        for i,s in enumerate(target.var.gene_ids.astype(str)): stable.setdefault(s,[]).append(i)
    raw=sparse.csc_matrix(target.X); columns=[];mapping=[];used=[]
    for gene in map(str,bundle['genes']):
        idx=stable.get((reference or {}).get(gene),[])
        method='training_ensembl_id'
        if not idx: idx=symbols.get(gene,[]);method='exact_symbol_fallback'
        if not idx: method='explicit_zero_fill'
        used.extend(idx)
        columns.append(sparse.csr_matrix(raw[:,idx].sum(axis=1)) if idx else sparse.csr_matrix((target.n_obs,1)))
        mapping.append({'model_gene':gene,'mapping_method':method,'resolved':bool(idx)})
    assert len(used)==len(set(used)), 'A feature maps to multiple model genes'
    panel=sparse.hstack(columns,format='csr',dtype=np.float32)
    total=np.asarray(panel.sum(axis=1)).ravel();assert (total>0).all()
    panel=sparse.diags((1e4/total).astype(np.float32))@panel
    panel.data=np.log1p(panel.data)
    p=bundle['model'].predict_proba(panel)[:,1]
    mapping=pd.DataFrame(mapping)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return p,mapping
PROGRESS={'complete':False,'units':{},'parameters':PARAMETERS,'source_hash':SOURCE_HASH}
def add_hierarchy(z):
    z=z.copy()
    z['P_gdT']=z.p_T*z.p_gdT_given_T
    root_ok=z.root_feature_coverage.ge(.95)
    passed=root_ok & z.child_feature_coverage.ge(.95) & z.p_T.ge(GATE['threshold_T'])
    z['hard_call']=np.select([passed & z.p_gdT_given_T.ge(THRESHOLDS['optimized_M05']),
        passed,root_ok & z.p_T.le(GATE['threshold_NK'])],
        ['predicted gdT','predicted abT','predicted NK'],default='abstain')
    z['gate_excluded_M05_positive']=z.p_gdT_given_T.ge(THRESHOLDS['optimized_M05']) & ~z.hard_call.eq('predicted gdT')
    return z
def donor_predictions(dataset, frame, raw, raw_path, child_oof=False):
    pieces=[]
    for donor,z in frame.groupby('donor',sort=True,observed=True):
        z=z.copy();stem=f'{dataset}_{donor}'
        path=CHECKPOINT/f'{stem}.parquet';meta=CHECKPOINT/f'{stem}.manifest.json'
        signature=digest({'source':SOURCE_HASH,'models':MODEL_HASH,'input':track(raw_path),
            'cells':ids_hash(z.cell_id),'frame':digest(z.to_dict('list')),'parameters':PARAMETERS,
            'genes':{k:list(map(str,v['genes'])) for k,v in MODELS.items()},'child_oof':child_oof})
        prior=json.loads(meta.read_text()) if meta.exists() else {}
        if prior.get('signature')==signature and path.exists() and prior.get('sha256')==sha(path):
            out=pd.read_parquet(path)
            assert out.columns.tolist()==prior['columns'] and out.cell_id.tolist()==z.cell_id.tolist()
            print('Validated checkpoint:',stem,flush=True)
        else:
            assert z.cell_id.is_unique and raw.obs_names.is_unique
            assert z.cell_id.isin(raw.obs_names).all()
            target=raw[z.cell_id.tolist()].to_memory()
            if 'counts' in target.layers: target.X=target.layers['counts']
            out=z.copy()
            out['p_T'],mapping=score(GATE,target)
            out['root_feature_coverage']=mapping.resolved.mean()
            atomic_frame(mapping,CHECKPOINT/f'{stem}_gate_feature_mapping.csv')
            if child_oof: out['child_feature_coverage']=1.0
            else:
                out['p_gdT_given_T'],mapping=score(MODELS['optimized_M05'],target,GENE_IDS)
                out['child_feature_coverage']=mapping.resolved.mean()
                atomic_frame(mapping,CHECKPOINT/f'{stem}_M05_feature_mapping.csv')
                historical=predict_lr(MODELS['controlled_M05']['model'],target,
                    MODELS['controlled_M05']['genes'],target.obs_names.tolist())
                discrepancy=float(np.max(np.abs(historical-out.p_gdT_M05.to_numpy())))
                assert discrepancy<1e-5, ('Historical SPL reproduction failed',discrepancy)
                out['historical_reproduction_max_abs_difference']=discrepancy
            out=add_hierarchy(out)
            atomic_frame(out,path)
            prior={'signature':signature,'sha256':sha(path),'columns':out.columns.tolist(),'n':len(out)}
            atomic_json(prior,meta)
            print('Saved donor:',stem,len(out),flush=True)
        pieces.append(out);PROGRESS['units'][stem]=prior
        atomic_json(PROGRESS,CHECKPOINT/'completion_manifest.json')
    result=pd.concat(pieces,ignore_index=True)
    assert result.cell_id.is_unique and set(result.cell_id)==set(frame.cell_id)
    atomic_frame(result,PREDICTION/f'{dataset}_LR_hierarchical_predictions.parquet')
    return result
# %% markdown
## Reference: Song held-out donor predictions
Saved source OOF predictions are used, never predictions from a child fitted on
the same cells. The frozen operating thresholds were selected on source OOF
data, so the operating-point results are developmental estimates, not a fully
nested unbiased validation of threshold selection. Source and external results
are kept separate; Song label definitions are inherited from the saved M05 OOF
tables and are not upgraded to a different evidence tier.
# %% code
oof={}
for version in ['controlled','optimized']:
    path=PROJECT/f'predictions/song_{version}_oof.parquet';track(path)
    z=pd.read_parquet(path).query('model_id == "M05"').copy()
    assert z.cell_id.is_unique
    oof[version]=z
reference=oof['controlled'].rename(columns={'p_gdT':'p_gdT_M05'}).merge(
    oof['optimized'].rename(columns={'p_gdT':'p_gdT_given_T'}),
    on=['cell_id','donor','model_id','y'],validate='one_to_one')
assert len(reference)==len(oof['controlled'])==len(oof['optimized'])
reference=donor_predictions('Song',reference,song,SONG_PATH,child_oof=True)
song.file.close()
# %% markdown
## External: SPL receptor evidence
The original strict paired-receptor, orphan TRD, orphan TRG and expanded
partial-evidence endpoints retain their original cells. Orphan-chain endpoints
are partial evidence, not paired-receptor gold. Each uses the same paired-abT
negative group and its prespecified positive group(s).
# %% code
spl_path=PROJECT/'predictions/M05_SPL_strict_expanded_partial_receptor_predictions.parquet'
track(spl_path)
spl=pd.read_parquet(spl_path).rename(columns={'p_gdT':'p_gdT_M05'})
assert spl.cell_id.is_unique
SPL_RAW=PROJECT/'manifests/CrossAtlas_D2_SPL_cohorts.h5ad';track(SPL_RAW)
raw=ad.read_h5ad(SPL_RAW,backed='r')
spl=donor_predictions('SPL',spl,raw,SPL_RAW)
raw.file.close()
# %% markdown
## External: original PANGEA and 15-organ evaluation
Existing notebook-22 prediction caches are accepted only after verification of
model/input checksums, per-unit cell identities, complete coverage and the
probability factorization. Raw expression and original baseline/coordinate
hashes are verified too. Historical versus current mapping remains available
as a separate control. Scores are not recomputed solely to reproduce an already
verified deterministic result. All original labels and cells are preserved.
# %% code
atlases={}
for dataset in ['PANGEA','15-organ']:
    path=PROJECT/'predictions/22_LR_hierarchical_CE'/f'{dataset}_LR_hierarchical_predictions.parquet'
    meta=path.with_suffix('.manifest.json');track(meta)
    manifest=json.loads(meta.read_text());assert manifest['complete']
    track(path,manifest['prediction_sha256'])
    assert manifest['source_hash']['gate']==MODEL_HASH['gate']
    assert manifest['source_hash']['M05']==MODEL_HASH['optimized_M05']
    assert manifest['source_hash']['mapping_reference']==track(SONG_PATH)
    for p,h in manifest['baseline_hashes'].items(): track(p,h)
    z=pd.read_parquet(path)
    assert z.cell_id.is_unique and len(z)==manifest['n_cells'] and z.columns.tolist()==manifest['columns']
    for unit,entry in manifest['units'].items():
        track(entry['input_path'],entry['input_sha256'])
        part=z[z._unit.eq(unit)]
        assert len(part)==entry['n'] and ids_hash(part.cell_id)==entry['cell_ids_sha256']
    assert np.allclose(z.P_gdT,z.p_T*z.p_gdT_given_T,rtol=0,atol=1e-14)
    assert np.array_equal(z.hard_call.eq('predicted gdT'),add_hierarchy(z).hard_call.eq('predicted gdT'))
    for col in ['p_T','p_gdT_given_T','P_gdT','p_gdT_M05']:
        assert z[col].notna().all() and z[col].between(0,1).all()
    atlases[dataset]=z
    atomic_frame(z,PREDICTION/f'{dataset}_LR_hierarchical_predictions.parquet')
    print('Verified frozen atlas:',dataset,len(z),flush=True)
# %% markdown
## Matched-cell curves and operating points
The reference label is explicit for every row of every table. Continuous
curves compare historical controlled_M05, optimized_M05 and CE–M05. An additional
controlled_M05 current-mapping curve is exported as a technical control for
the atlases. Hard-gate decisions are kept separate from continuous-score metrics.
Brier is probability squared error; a smaller value alone does not establish
improved calibration. A fixed gate cannot increase gdT detections at the same
child cutoff. High recall must be assessed with the excluded candidates retained.
# %% code
SCORES=[('controlled_M05','p_gdT_M05','#777777'),
        ('optimized_M05','p_gdT_given_T','#F02B1F'),('CE–M05','P_gdT','#075DF3')]
ENDPOINTS={}
def endpoint(key,frame,label,evidence,unit):
    z=frame.copy();z['y']=label.astype(int)
    assert z.cell_id.is_unique and z.y.isin([0,1]).all()
    ENDPOINTS[key]={'data':z,'evidence':evidence,'unit':unit}
endpoint('Song',reference,reference.y,'Source OOF labels; development operating point','donor')
for donor,z in reference.groupby('donor',observed=True):
    endpoint('Song_'+str(donor),z,z.y,'Source OOF labels; development operating point','donor')
spl_groups={'strict_gold':['paired_gdT'],'orphan_TRD_partial':['orphan_TRD'],
    'orphan_TRG_partial':['orphan_TRG'],'expanded_partial_evidence':['paired_gdT','orphan_TRD','orphan_TRG']}
for name,positive in spl_groups.items():
    z=spl[spl.receptor_evidence_group.isin(positive+['paired_abT'])]
    evidence='Paired receptor evidence' if name=='strict_gold' else 'Partial receptor evidence; not paired gold'
    endpoint('SPL_'+name,z,z.receptor_evidence_group.isin(positive),evidence,'donor')
    for donor,part in z.groupby('donor',observed=True):
        endpoint('SPL_'+name+'_'+str(donor),part,part.receptor_evidence_group.isin(positive),evidence,'donor')
z=atlases['PANGEA'];z=z[z.silver_evaluation_eligible.astype(bool)]
endpoint('PANGEA',z,z.silver_y,'Author annotation concordance','Series')
part=z[~z.Series.eq('GSE149356')]
endpoint('PANGEA_excluding_GSE149356',part,part.silver_y,'Author annotation concordance; overlap sensitivity','Series')
for unit,part in z.groupby('Series',observed=True):
    endpoint('PANGEA_'+str(unit),part,part.silver_y,'Author annotation concordance','Series')
z=atlases['15-organ'];z=z[z.primary_hybrid_eligible.astype(bool)]
endpoint('15-organ',z,z.primary_hybrid_y,'Original hybrid annotation/receptor endpoint','organ (one donor)')
for unit,part in z.groupby('organ',observed=True):
    endpoint('15-organ_'+str(unit),part,part.primary_hybrid_y,'Original hybrid annotation/receptor endpoint','organ (one donor)')
metrics=[];points=[];curves=[];flow=[]
for name,entry in ENDPOINTS.items():
    z=entry['data'];y=z.y.to_numpy();n1=int(y.sum());n0=len(y)-n1
    base={'endpoint':name,'evidence':entry['evidence'],'unit':entry['unit'],'n':len(y),'n_positive':n1,'n_negative':n0}
    flow.append(base)
    scores=SCORES+([('controlled_M05; current mapping','p_gdT_M05_controlled_current_mapping','#777777')]
                   if 'p_gdT_M05_controlled_current_mapping' in z else [])
    for model,col,color in scores:
        p=z[col].to_numpy()
        metrics.append({**base,'model':model,'AUROC':roc_auc_score(y,p) if n0 and n1 else np.nan,
            'AUPRC':average_precision_score(y,p) if n0 and n1 else np.nan,
            'Brier':brier_score_loss(y,p),'status':'estimable' if n0 and n1 else 'single_class'})
        if n0 and n1:
            precision,recall,cutoffs=precision_recall_curve(y,p)
            fpr,tpr,roc_cutoffs=roc_curve(y,p,drop_intermediate=False)
            curves.extend(pd.DataFrame({'endpoint':name,'model':model,'curve':'PR','x':recall,'y':precision,
                'threshold':np.r_[cutoffs,np.nan]}).to_dict('records'))
            curves.extend(pd.DataFrame({'endpoint':name,'model':model,'curve':'ROC','x':fpr,'y':tpr,
                'threshold':roc_cutoffs}).to_dict('records'))
    decisions=[('controlled_M05',z.p_gdT_M05.ge(THRESHOLDS['controlled_M05'])),
       ('optimized_M05',z.p_gdT_given_T.ge(THRESHOLDS['optimized_M05'])),
       ('LR gate + optimized_M05',z.hard_call.eq('predicted gdT'))]
    assert (decisions[2][1] <= decisions[1][1]).all()
    for model,pred in decisions:
        pred=np.asarray(pred);tp=int((pred & (y==1)).sum());fp=int((pred & (y==0)).sum())
        fn=n1-tp;tn=n0-fp
        points.append({**base,'model':model,'TP':tp,'FP':fp,'FN':fn,'TN':tn,
            'precision':tp/(tp+fp) if tp+fp else np.nan,'recall':tp/n1 if n1 else np.nan,
            'FPR':fp/n0 if n0 else np.nan,'F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
            'n_positive_calls':int(pred.sum())})
METRICS=pd.DataFrame(metrics);POINTS=pd.DataFrame(points);CURVES=pd.DataFrame(curves)
atomic_frame(METRICS,TABLE/'continuous_score_metrics.csv')
atomic_frame(POINTS,TABLE/'frozen_operating_points.csv')
atomic_frame(CURVES,TABLE/'ROC_PR_curve_source_data.parquet')
atomic_frame(pd.DataFrame(flow),TABLE/'evaluation_cohort_flow.csv')
display(POINTS[POINTS.endpoint.isin(['Song','SPL_strict_gold','PANGEA','15-organ'])])
# %% code
FIGURE_MANIFEST=[]
def save_figure(fig,stem,claim,exclude_axes=None):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=LOG/f'{stem}_alignment.json',
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True,
        exclude_axes=exclude_axes or [])
    for ext,suffix in [('pdf','.pdf'),('svg','.svg'),('png','.png')]:
        path=FIGURE/(stem+suffix);tmp=FIGURE/(stem+'.tmp'+suffix)
        fig.savefig(tmp,format=ext,dpi=300,facecolor='white')
        os.replace(tmp,path)
    FIGURE_MANIFEST.append({'stem':stem,'claim':claim,'pdf_sha256':sha(FIGURE/f'{stem}.pdf'),
        'vector_points':True,'font':font,'source_tables':str(TABLE)})
    atomic_json(FIGURE_MANIFEST,LOG/'figure_manifest.json')
    plt.close(fig)
    print('Saved figure:',stem,flush=True)
def curve_figure(names,stem):
    fig,axs=plt.subplots(2,len(names),figsize=(7.2,5.15),squeeze=False)
    fig.subplots_adjust(left=.085,right=.985,bottom=.13,top=.90,wspace=.40,hspace=.58)
    for j,name in enumerate(names):
        z=ENDPOINTS[name]['data'];n1=int(z.y.sum());n0=len(z)-n1
        for i,kind in enumerate(['PR','ROC']):
            ax=axs[i,j]
            for model,col,color in SCORES:
                c=CURVES[(CURVES.endpoint==name)&(CURVES.model==model)&(CURVES.curve==kind)]
                if kind=='PR': ax.step(c.x.iloc[::-1],c.y.iloc[::-1],where='post',color=color,lw=1)
                else: ax.plot(c.x,c.y,color=color,lw=1)
            point=POINTS[(POINTS.endpoint==name)&POINTS.model.eq('LR gate + optimized_M05')].iloc[0]
            ax.scatter(point['recall'] if kind=='PR' else point['FPR'],
                point['precision'] if kind=='PR' else point['recall'],marker='D',s=19,
                color='black',zorder=8,clip_on=False)
            if kind=='PR': ax.axhline(n1/len(z),color='black',ls=':',lw=.6)
            else: ax.plot([0,1],[0,1],color='black',ls=':',lw=.6)
            ax.set(xlim=(-.015,1.015),ylim=(-.015,1.015),xticks=[0,.5,1],yticks=[0,.5,1],
                xlabel='Recall' if kind=='PR' else 'False positive rate',
                ylabel='Precision' if kind=='PR' else 'True positive rate')
            ax.set_box_aspect(1)
            title=name.replace('PANGEA_excluding_GSE149356','PANGEA\nexcluding GSE149356').replace('SPL_','SPL\n').replace('_partial','\npartial evidence').replace('expanded_partial_evidence','expanded\npartial evidence')
            ax.set_title(title+f'\nn+ = {n1:,}; n− = {n0:,}',pad=7)
            ax.text(-.22,1.21,string.ascii_lowercase[i*len(names)+j],transform=ax.transAxes,fontsize=8)
    handles=[Line2D([],[],color=c,label=n) for n,_,c in SCORES]
    handles.append(Line2D([],[],color='black',marker='D',ls='',label='LR gate + optimized_M05'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=2,frameon=False)
    save_figure(fig,stem,'Matched-cell PR/ROC; frozen hard gate shown only as an operating point.')
curve_figure(['Song','SPL_strict_gold'], 'Song_SPL_ROC_PR')
curve_figure(['PANGEA','PANGEA_excluding_GSE149356','15-organ'],'PANGEA_15organ_ROC_PR')
curve_figure(['Song_'+str(d) for d in sorted(reference.donor.unique())],'Song_ROC_PR_by_donor')
curve_figure(['SPL_orphan_TRD_partial','SPL_orphan_TRG_partial','SPL_expanded_partial_evidence'],'SPL_partial_evidence_ROC_PR')
organ_endpoints=[k for k in ENDPOINTS if k.startswith('15-organ_') and ENDPOINTS[k]['data'].y.nunique()==2]
for start in range(0,len(organ_endpoints),3):
    names=organ_endpoints[start:start+3]
    curve_figure(names,'15organ_M05_hybrid_ROC_PR_by_organ_'+'_'.join(x.removeprefix('15-organ_') for x in names))
# %% markdown
## UMAP: unchanged geometry, all available cells
The original coordinate tables are joined one-to-one by cell ID. Cells without
original QC-passing coordinates remain in numerical evaluation; their missing
coordinates are recorded separately. All panels share the same full coordinate
bounds. No quantile clipping or model-dependent embedding is used.
Continuous probability panels share a 0–1 scale. The categorical comparison
uses the original gdT-related annotation and abT reference evidence. Prediction
and annotation agreement is descriptive, not a truth assignment. Original
fine source labels and T/NK subsets are provided separately, retaining NK-like
labels exactly. Gate-excluded predictions remain explicit in the cell tables.
# %% code
COORDINATES={'PANGEA':PROJECT/'results/tables/16_PANGEA_M05_silver_concordance_and_discovery/PANGEA_M05_UMAP_coordinates.csv',
    '15-organ':Path('/data/users/kaelynwen/gdTcell_Yuanhua/15organs_validation/results/tables/15organ_M05_UMAP_group_assignments.csv.gz')}
umap_flow=[];all_cell_counts=[]
def umap_axes(ax,z,bounds):
    ax.set(xlim=bounds[0],ylim=bounds[1],xlabel='UMAP1',ylabel='UMAP2',xticks=[],yticks=[])
    ax.set_aspect('equal',adjustable='box')
def draw_source(z,label,stem,source_colors):
    categories=sorted(z[label].dropna().astype(str).unique())
    # Exact source annotation names, never relabelled as ground truth.
    fig,ax=plt.subplots(figsize=(7.2,5.1));fig.subplots_adjust(left=.06,right=.44,bottom=.09,top=.92)
    for index,group in enumerate(categories):
        p=z[z[label].astype(str).eq(group)]
        ax.scatter(p.UMAP1,p.UMAP2,s=1.1,c=[source_colors[group]],lw=0,label=group)
    bounds=[(z[c].min()-.5,z[c].max()+.5) for c in ['UMAP1','UMAP2']]
    umap_axes(ax,z,bounds);ax.set_title(label+' (source annotation)')
    ax.legend(loc='center left',bbox_to_anchor=(1.03,.5),frameon=False,markerscale=3,
        fontsize=6,ncol=1 if len(categories)<=30 else 2,handletextpad=.4,labelspacing=.4)
    save_figure(fig,stem,'Original source annotations on unchanged UMAP; not truth assignments.')
for dataset,z in atlases.items():
    coords=pd.read_csv(COORDINATES[dataset]);track(COORDINATES[dataset])
    assert coords.cell_id.is_unique and coords.cell_id.isin(z.cell_id).all()
    extra=['Cell_type_in_each_tissue','Cell_type_in_merged_data'] if dataset=='15-organ' else []
    keep=['cell_id','UMAP1','UMAP2']+[c for c in extra if c in coords and c not in z]
    merged=z.merge(coords[keep],on='cell_id',how='left',validate='one_to_one')
    merged['historical_M05_positive']=merged.p_gdT_M05.ge(THRESHOLDS['controlled_M05'])
    merged['optimized_M05_positive']=merged.p_gdT_given_T.ge(THRESHOLDS['optimized_M05'])
    merged['gate_excluded_M05_positive']=merged.optimized_M05_positive & ~merged.hard_call.eq('predicted gdT')
    atomic_frame(merged,PREDICTION/f'{dataset}_gate_discordance_audit_UMAP_source_data.parquet')
    for group,part in merged.groupby('annotation_Level2' if dataset=='PANGEA' else 'published_annotation',dropna=False,observed=True):
        all_cell_counts.append({'dataset':dataset,'source_annotation':group,'n':len(part),
            'controlled_M05_positive':int(part.historical_M05_positive.sum()),
            'optimized_M05_positive':int(part.optimized_M05_positive.sum()),
            'LR_gate_optimized_M05_positive':int(part.hard_call.eq('predicted gdT').sum()),
            'gate_excluded_M05_positive':int(part.gate_excluded_M05_positive.sum())})
    umap_flow.append({'dataset':dataset,'prediction_cells':len(z),'embedded_cells':int(merged.UMAP1.notna().sum()),
        'not_embedded':int(merged.UMAP1.isna().sum()),'all_evaluation_cells_retained':True})
    v=merged.dropna(subset=['UMAP1','UMAP2']).copy()
    bounds=[(v[c].min()-.5,v[c].max()+.5) for c in ['UMAP1','UMAP2']]
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.9));fig.subplots_adjust(left=.06,right=.90,bottom=.17,top=.90,wspace=.24)
    for j,(model,col,color) in enumerate(SCORES):
        part=v.sort_values(col,kind='stable')
        sc=axs[j].scatter(part.UMAP1,part.UMAP2,c=part[col],s=.8,cmap='viridis',vmin=0,vmax=1,lw=0)
        umap_axes(axs[j],v,bounds);axs[j].set_title(model)
    cbax=fig.add_axes([.935,.23,.012,.55]);fig.colorbar(sc,cax=cbax,label='gdT score')
    save_figure(fig,dataset+'_M05_probability_UMAP','Fixed embedding and common probability scale.',exclude_axes=[cbax])
    annotated=v.PANGEA_gdT.astype(bool) if dataset=='PANGEA' else v.original_annotation_gdT_related.astype(bool)
    ab=v.PANGEA_conventional_abT.astype(bool) if dataset=='PANGEA' else v.strict_paired_abTCR.astype(bool)
    # Names and colours follow the original annotation/prediction overlay convention.
    labels=['Background','abT reference','Annotation only','Prediction only','Both']
    palette=['#D9D9D9','#0057B8','#00A6D6','#D73027','#7B2CBF']
    fig,axs=plt.subplots(1,3,figsize=(7.2,3.0));fig.subplots_adjust(left=.06,right=.99,bottom=.24,top=.91,wspace=.23)
    decisions=[('controlled_M05',v.historical_M05_positive),('optimized_M05',v.optimized_M05_positive),
        ('LR gate + optimized_M05',v.hard_call.eq('predicted gdT'))]
    for j,(model,pred) in enumerate(decisions):
        groups=np.select([annotated & pred,pred & ~annotated,annotated & ~pred,ab],[4,3,2,1],default=0)
        for group in range(5):
            p=v.loc[groups==group]
            axs[j].scatter(p.UMAP1,p.UMAP2,s=.65 if group<2 else 2.6,c=palette[group],lw=0,alpha=.45 if group==0 else .9)
        umap_axes(axs[j],v,bounds);axs[j].set_title(model)
    handles=[Line2D([],[],marker='o',ls='',color=c,label=l,markersize=4) for l,c in zip(labels,palette)]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.01))
    save_figure(fig,dataset+'_M05_predicted_annotation_abT_UMAP','Frozen calls versus inherited evidence; candidate exclusion is not identity proof.')
    source='annotation_Level2' if dataset=='PANGEA' else 'Cell_type_in_merged_data'
    if source not in v: source='published_annotation'
    colors=list(plt.get_cmap('tab20').colors)+list(plt.get_cmap('tab20b').colors)+list(plt.get_cmap('tab20c').colors)
    source_colors={group:colors[i] for i,group in enumerate(sorted(v[source].dropna().astype(str).unique()))}
    draw_source(v,source,dataset+'_M05_source_subtype_UMAP',source_colors)
    if dataset=='15-organ':
        # Select exact source T/NK annotation categories, not broad marker flags.
        # This view-only subset never changes any evaluation cohort.
        focus=v[v[source].astype(str).str.startswith('T Cell ')|v[source].eq('NK/T Cell GNLY')]
        draw_source(focus,source,'15organ_M05_T_NK_source_subtype_UMAP_zoomin',source_colors)
atomic_frame(pd.DataFrame(umap_flow),TABLE/'UMAP_cohort_flow.csv')
atomic_frame(pd.DataFrame(all_cell_counts),TABLE/'gate_discordance_audit_annotation_counts.csv')
display(pd.DataFrame(umap_flow))
# %% markdown
## Receptor-supported retention and figure quality assurance
The paired-receptor SPL arm is evaluated independently of source annotation
agreement. Per-donor results are preserved, so pooled retention does not hide
a concentration of exclusions in one donor. PNGs are viewing previews; vector
PDF/SVG are the editable figure outputs. Missing-coordinate counts are explicit,
and no confidence intervals based on treating cells as independent donors are
claimed. These explain the corresponding static-validator cautions.
# %% code
paired=spl[spl.receptor_evidence_group.eq('paired_gdT')].copy()
rows=[]
for donor,z in paired.groupby('donor',observed=True):
    before=z.p_gdT_given_T.ge(THRESHOLDS['optimized_M05'])
    after=z.hard_call.eq('predicted gdT')
    rows.append({'donor':donor,'paired_gdT':len(z),'optimized_M05_positive':int(before.sum()),
        'LR_gate_optimized_M05_positive':int(after.sum()),'additional_exclusions':int((before & ~after).sum()),
        'optimized_M05_recall':float(before.mean()),'LR_gate_optimized_M05_recall':float(after.mean())})
atomic_frame(pd.DataFrame(rows),TABLE/'SPL_gate_discordance_audit_by_donor.csv')
atomic_frame(paired[paired.gate_excluded_M05_positive],PREDICTION/'SPL_gate_discordance_audit_paired_gdT.parquet')
display(pd.DataFrame(rows))
before=int(paired.p_gdT_given_T.ge(THRESHOLDS['optimized_M05']).sum())
after=int(paired.hard_call.eq('predicted gdT').sum())
display(Markdown(f'**SPL paired-receptor result:** optimized_M05 detects {before}/{len(paired)} '
    f'({before/len(paired):.1%}); the frozen hard gate retains {after}/{len(paired)} '
    f'({after/len(paired):.1%}). It excludes {before-after} additional receptor-supported gdT cells. '
    'This does not support replacing M05 with this hard gate when discovery recall is the primary goal. '
    'Keep the gate as evidence for review, and retain its excluded candidates. '
    'Any recall-oriented threshold redesign belongs to source development, followed by a fresh independent test.'))
qa=[]
for pdf in sorted(FIGURE.glob('*.pdf')):
    if '.tmp.' in pdf.name: continue
    for tool,args in [('audit_pdf_text.py',['--min-pt','5','--json']),
                      ('audit_figure_collisions.py',['--json'])]:
        track(CHECKPOINT/tool)
        result=subprocess.run([sys.executable,str(CHECKPOINT/tool),str(pdf),*args],capture_output=True,text=True)
        report=LOG/f'{pdf.stem}_{tool[:-3]}.json'
        report.write_text(result.stdout)
        qa.append({'figure':pdf.name,'check':tool,'exit_code':result.returncode,'report':str(report)})
        if result.returncode: print(result.stdout[:2500],result.stderr[:500])
atomic_frame(pd.DataFrame(qa),LOG/'figure_QA.csv')
assert all(row['exit_code']==0 for row in qa), 'Fix PDF typography or layout before declaring completion'
# %% markdown
## Conclusions, limitations and reproducibility
Read frozen operating-point retention alongside the continuous curves. A lower
annotation recall is a disagreement with the inherited label, not proof that
true gdT was lost; receptor-supported loss is the stronger safety check. Likewise,
removing NK-annotated candidates is not proof of improved specificity.
Review the exported gate-excluded candidates rather than deleting them. If
maximizing discovery recall is the priority, threshold/model redesign must use
source development data and be frozen before a new independent external test.
This notebook does not change the deployed model or choose a new threshold.

The output manifest lists every artifact and checksum; the input manifest records
all source hashes. Per-donor checkpoints allow exact restart. A clean Run All
reproduces the tables and figures from immutable inputs on Woody. Model files
are re-hashed at completion to verify no fitting or modification occurred.
# %% code
for model,h in MODEL_HASH.items():
    assert sha(GATE_PATH if model=='gate' else PROJECT/f'models/{model}.joblib')==h
PROGRESS['complete']=True
atomic_json(PROGRESS,CHECKPOINT/'completion_manifest.json')
atomic_json({'inputs':INPUTS,'source_hash':SOURCE_HASH,'generated_utc':datetime.now(timezone.utc).isoformat(),
    'cache_provenance':'Notebook 22 manifests retain original implementation hashes; raw inputs and models verified.'},TABLE/'input_manifest.json')
files=[]
for directory in [TABLE,FIGURE,PREDICTION,CHECKPOINT,LOG]:
    for path in sorted(directory.glob('*')):
        if path.is_file() and path.name!='output_manifest.csv' and '.tmp' not in path.name:
            files.append({'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)})
atomic_frame(pd.DataFrame(files),TABLE/'output_manifest.csv')
print('Complete:',ANALYSIS_ID,'No models fitted; no thresholds changed.')
display(Image(filename=str(FIGURE/'PANGEA_15organ_ROC_PR.png')))
'''

DISCORDANCE = r'''
# %% markdown
## Gate-discordance UMAP: retained gdT, excluded candidates and predicted NK

This section is independently runnable from the saved, checksum-verified
cell-level tables. It retains the preceding evaluation outputs and does not
train models, alter thresholds, change evaluation labels or refit UMAP.
The scientific question is where the frozen model decisions lie on the original
embedding. Spatial overlap is descriptive; it does not establish cellular identity.

The combined view uses mutually exclusive groups:

- `predicted gdT`: gdT retained after the frozen T/NK gate and M05 threshold;
- `optimized_M05 positive; predicted NK`: previous gdT candidates explicitly
  called NK by the root gate;
- `optimized_M05 negative; predicted NK`: the remaining root-NK calls, so the
  preceding group is not counted twice;
- `optimized_M05 positive; abstain/out-of-domain`: excluded candidates without
  an explicit NK call, retained as a separate group;
- other cells form the contextual background.

The four-panel view highlights retained gdT, excluded root-NK candidates, **all**
root-NK calls, and abstaining candidates separately. Its all-NK panel deliberately
includes the excluded root-NK candidates. Groups overlap across panels, but not
within the combined view. Root-NK calls outside the T/NK training domain do not
establish NK identity. Original author annotations remain in the source table.

All original coordinates are retained. Missing original UMAP coordinates are
counted separately and never replaced by a new embedding. Panels for the same
dataset share identical limits; PANGEA and 15-organ have distinct embeddings.
The export contract is editable vector PDF/SVG, a PNG preview, normal black
Arial/Helvetica/Liberation Sans text, source data, input hashes and automatic QA.
Restarting this section reuses only complete dataset units with matching input,
source-section and output hashes. Its own manifest is separate from the earlier
evaluation run's provenance.
# %% code
from pathlib import Path
from datetime import datetime, timezone
import os, sys, json, hashlib, platform, importlib.metadata, subprocess, runpy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from IPython.display import display, Image

gd_name='21_gate_LRM05_evaluation_15organs'
gd_remote=Path('/data/users/kaelynwen/gdT_project/gdt_lr_12model')
gd_local=Path('/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/gdt_lr_12model')
gd_project=Path(os.environ.get('GDT_LR_PROJECT',str(gd_remote if gd_remote.exists() else gd_local)))
gd_table=gd_project/'results/tables'/gd_name
gd_figure=gd_project/'results/figures'/gd_name
gd_prediction=gd_project/'predictions'/gd_name
gd_checkpoint=gd_project/'checkpoints'/gd_name
gd_log=gd_project/'logs'/gd_name
sys.path.insert(0,str(gd_checkpoint))
from audit_panel_alignment import require_matplotlib_panel_alignment
gd_font=next((f for f in ['Arial','Helvetica','Liberation Sans']
    if f in {x.name for x in font_manager.fontManager.ttflist}),None)
if gd_font is None: raise RuntimeError('A permitted publication font is required')
plt.rcParams.update({'font.family':gd_font,'font.weight':'normal','font.size':7,
    'axes.titlesize':8,'axes.titleweight':'normal','axes.labelsize':7,'legend.fontsize':6,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','text.color':'black',
    'axes.labelcolor':'black','axes.titlecolor':'black','xtick.color':'black','ytick.color':'black',
    'legend.labelcolor':'black','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6})
def gd_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def gd_json(value,path):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,indent=2,default=str));os.replace(tmp,path)
def gd_frame(value,path):
    tmp=Path(str(path)+'.tmp')
    if str(path).endswith('.parquet'):value.to_parquet(tmp,index=False)
    else:value.to_csv(tmp,index=False)
    os.replace(tmp,path)
gd_generator=gd_project/f'build_{gd_name}.py'
gd_section=runpy.run_path(str(gd_generator))['DISCORDANCE']
gd_source_hash=hashlib.sha256(gd_section.encode()).hexdigest()
gd_previous=pd.read_csv(gd_table/'output_manifest.csv').set_index('path')
gd_parameters=json.loads((gd_table/'parameters.json').read_text())
gd_labels=['Other cells','predicted gdT','optimized_M05 positive; predicted NK',
           'optimized_M05 negative; predicted NK','optimized_M05 positive; abstain/out-of-domain']
gd_colors=['#D9D9D9','#7B2CBF','#D73027','#0057B8','#E69F00']
gd_manifest_path=gd_checkpoint/'gate_discordance_audit_UMAP.manifest.json'
gd_previous_run=json.loads(gd_manifest_path.read_text()) if gd_manifest_path.exists() else {}
gd_manifest={'complete':False,'source_section_sha256':gd_source_hash,
    'generator_sha256':gd_sha(gd_generator),'host':platform.node(),'python':sys.version,
    'font':gd_font,'matplotlib':matplotlib.__version__,'numpy':np.__version__,'pandas':pd.__version__,
    'unit_definition':'one dataset; both figures and source data, after all QA checks',
    'labels':gd_labels,'colors':gd_colors,'coordinate_policy':'original; no refit or quantile clipping',
    'units':{}}
gd_json(gd_manifest,gd_manifest_path)
def gd_axes(ax,limits):
    ax.set(xlim=limits[0],ylim=limits[1],xlabel='UMAP1',ylabel='UMAP2',xticks=[],yticks=[])
    ax.set_aspect('equal',adjustable='box')
def gd_export(fig,stem):
    outputs=[]
    fig.canvas.draw()
    alignment=gd_log/f'{stem}_alignment.json'
    require_matplotlib_panel_alignment(fig,json_out=alignment,tolerance_pt=1.5,
                                      gutter_tolerance_pt=1.5,strict=True)
    outputs.append(alignment)
    for extension in ['.pdf','.svg','.png']:
        path=gd_figure/(stem+extension);tmp=gd_figure/(stem+'.tmp'+extension)
        fig.savefig(tmp,format=extension[1:],dpi=300,facecolor='white')
        os.replace(tmp,path);outputs.append(path)
    plt.close(fig)
    for tool,args in [('audit_pdf_text.py',['--min-pt','5','--json']),
                      ('audit_figure_collisions.py',['--json'])]:
        report=gd_log/f'{stem}_{tool[:-3]}.json'
        check=subprocess.run([sys.executable,str(gd_checkpoint/tool),str(gd_figure/(stem+'.pdf')),*args],
                             capture_output=True,text=True)
        report.write_text(check.stdout);outputs.append(report)
        if check.returncode: raise RuntimeError(check.stdout+'\n'+check.stderr)
    return outputs

for gd_dataset in ['PANGEA','15-organ']:
    gd_input=gd_prediction/f'{gd_dataset}_gate_discordance_audit_UMAP_source_data.parquet'
    gd_input_hash=gd_sha(gd_input)
    gd_record=str(gd_remote/'predictions'/gd_name/gd_input.name)
    if gd_record not in gd_previous.index:gd_record=str(gd_input)
    assert gd_input_hash==gd_previous.loc[gd_record,'sha256'],'Frozen UMAP/source table changed'
    gd_signature=hashlib.sha256(json.dumps({'input':gd_input_hash,'section':gd_source_hash,
        'parameters':gd_parameters,'font':gd_font,'matplotlib':matplotlib.__version__},sort_keys=True).encode()).hexdigest()
    gd_prior=gd_previous_run.get('units',{}).get(gd_dataset,{})
    if gd_prior.get('signature')==gd_signature and all(
        (gd_project/rel).is_file() and gd_sha(gd_project/rel)==h for rel,h in gd_prior.get('outputs',{}).items()
    ) and gd_prior.get('outputs'):
        gd_manifest['units'][gd_dataset]=gd_prior
        gd_json(gd_manifest,gd_manifest_path)
        print('Validated complete UMAP checkpoint:',gd_dataset,flush=True)
        continue
    gd_data=pd.read_parquet(gd_input)
    assert gd_data.cell_id.is_unique
    gd_positive=gd_data.p_gdT_given_T.ge(gd_parameters['thresholds']['optimized_M05'])
    assert np.array_equal(gd_positive,gd_data.optimized_M05_positive)
    gd_nk=gd_data.hard_call.eq('predicted NK')
    gd_abstain=gd_data.hard_call.isin(['abstain','abstain/out-of-domain'])
    gd_pass=gd_data.hard_call.eq('predicted gdT')
    assert (gd_pass<=gd_positive).all()
    gd_masks=[gd_pass,gd_positive & gd_nk,~gd_positive & gd_nk,gd_positive & gd_abstain]
    gd_membership=np.column_stack([m.to_numpy() for m in gd_masks])
    assert (gd_membership.sum(axis=1)<=1).all()
    assert np.array_equal(gd_positive & ~gd_pass,gd_masks[1]|gd_masks[3]),'Unclassified excluded M05 candidates'
    gd_group=np.select(gd_masks,[1,2,3,4],default=0)
    gd_data['gate_discordance_audit_UMAP_group']=[gd_labels[i] for i in gd_group]
    gd_data['gate_discordance_audit_UMAP_group_index']=gd_group
    gd_available=gd_data[['UMAP1','UMAP2']].notna().all(axis=1)
    gd_counts=pd.DataFrame([{'dataset':gd_dataset,'group':name,
        'all_cells':int((gd_group==i).sum()),'embedded_cells':int(((gd_group==i)&gd_available).sum()),
        'missing_original_coordinates':int(((gd_group==i)&~gd_available).sum())} for i,name in enumerate(gd_labels)])
    gd_outputs=[]
    for frame,path in [(gd_data,gd_prediction/f'{gd_dataset}_gate_discordance_audit_UMAP_groups.parquet'),
                       (gd_counts,gd_table/f'{gd_dataset}_gate_discordance_audit_UMAP_group_counts.csv')]:
        gd_frame(frame,path);gd_outputs.append(path)
    gd_v=gd_data.loc[gd_available].copy()
    assert np.isfinite(gd_v[['UMAP1','UMAP2']].to_numpy()).all()
    gd_limits=[]
    for coordinate in ['UMAP1','UMAP2']:
        lo,hi=gd_v[coordinate].min(),gd_v[coordinate].max();margin=.025*(hi-lo)
        gd_limits.append((lo-margin,hi+margin))
    gd_fig,gd_ax=plt.subplots(figsize=(7.2,5.2))
    gd_fig.subplots_adjust(left=.09,right=.97,bottom=.24,top=.91)
    gd_background=gd_v[gd_v.gate_discordance_audit_UMAP_group_index.eq(0)]
    gd_ax.scatter(gd_background.UMAP1,gd_background.UMAP2,s=.7,c=gd_colors[0],alpha=.3,lw=0)
    gd_foreground=gd_v[gd_v.gate_discordance_audit_UMAP_group_index.ne(0)]
    gd_order=np.random.default_rng(17).permutation(len(gd_foreground))
    gd_foreground=gd_foreground.iloc[gd_order]
    gd_ax.scatter(gd_foreground.UMAP1,gd_foreground.UMAP2,s=2,
        c=[gd_colors[i] for i in gd_foreground.gate_discordance_audit_UMAP_group_index],lw=0,alpha=.85)
    gd_axes(gd_ax,gd_limits);gd_ax.set_title(gd_dataset+' | frozen gate and M05 decisions')
    gd_handles=[Line2D([],[],marker='o',ls='',color=gd_colors[i],markersize=4,
        label=name+f' (n = {int(gd_counts.iloc[i].embedded_cells):,})') for i,name in enumerate(gd_labels)]
    gd_fig.legend(handles=gd_handles,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.035))
    gd_outputs+=gd_export(gd_fig,gd_dataset+'_gate_discordance_audit_UMAP')
    gd_fig,gd_axes_array=plt.subplots(2,2,figsize=(7.2,6.4))
    gd_fig.subplots_adjust(left=.08,right=.98,bottom=.09,top=.90,wspace=.16,hspace=.43)
    gd_panel_masks=[gd_v.hard_call.eq('predicted gdT'),gd_v.gate_discordance_audit_UMAP_group_index.eq(2),
                    gd_v.hard_call.eq('predicted NK'),gd_v.gate_discordance_audit_UMAP_group_index.eq(4)]
    gd_titles=['predicted gdT','optimized_M05 positive;\npredicted NK',
               'All predicted NK','optimized_M05 positive;\nabstain/out-of-domain']
    for i,(ax,mask,title,color) in enumerate(zip(gd_axes_array.flat,gd_panel_masks,gd_titles,
                                              [gd_colors[j] for j in [1,2,3,4]])):
        ax.scatter(gd_v.UMAP1,gd_v.UMAP2,s=.45,c=gd_colors[0],alpha=.25,lw=0)
        selected=gd_v.loc[mask]
        ax.scatter(selected.UMAP1,selected.UMAP2,s=1.8,c=color,lw=0,alpha=.9)
        gd_axes(ax,gd_limits);ax.set_title(title+f'\nn = {len(selected):,}',pad=6)
        ax.text(-.07,1.20,'abcd'[i],transform=ax.transAxes,fontsize=8)
    gd_outputs+=gd_export(gd_fig,gd_dataset+'_gate_discordance_audit_UMAP_panels')
    gd_manifest['units'][gd_dataset]={'signature':gd_signature,'input_path':str(gd_input),
        'input_sha256':gd_input_hash,'n_cells':len(gd_data),'n_embedded':len(gd_v),
        'n_missing_coordinates':int((~gd_available).sum()),'limits':gd_limits,
        'outputs':{str(path.relative_to(gd_project)):gd_sha(path) for path in gd_outputs}}
    gd_json(gd_manifest,gd_manifest_path)
    display(gd_counts)
    print('Saved and validated UMAP unit:',gd_dataset,flush=True)
gd_manifest['complete']=True
gd_manifest['completed_utc']=datetime.now(timezone.utc).isoformat()
gd_json(gd_manifest,gd_manifest_path)
display(Image(filename=str(gd_figure/'PANGEA_gate_discordance_audit_UMAP.png')))
display(Image(filename=str(gd_figure/'15-organ_gate_discordance_audit_UMAP.png')))
'''

RECEPTOR_AUDIT = r'''
# %% markdown
## Gate-discordance audit: receptor identity, RNA/QC and T/NK context

**Question.** Do excluded M05 candidates have independent gamma-delta receptor
support, and which measured features accompany exclusion? These are diagnostic
comparisons; neither source annotations nor UMAP clusters are ground truth.

SPL includes all 378 saved paired-gdT cells. The primary contrast is the 22
additional exclusions against the 205 M05-positive retained cells, within donor.
The remaining 151 M05-negative cells remain in the audit table. NK calls and
abstentions are separate. Original alpha-beta and gamma-delta receptor objects
are matched by intact library barcode and donor. Locus, CDR3 amino-acid and
nucleotide sequence, V/J calls, multiplicity, and available doublet fields are
exported. Missing productive/high-confidence/full-length flags remain unknown.
Exact paired-sequence sharing is assessed within donor, using one TRG and one
TRD only; extra-chain cells are kept and reported separately.

RNA depth is recomputed from validated integer counts, not inherited QC columns
computed on another expression representation. Marker values are
log1p(10,000 * gene counts / all-gene counts). Frozen model scores use their
original panel normalization. Exact gate log-odds contributions diagnose model
arithmetic, not a causal biological mechanism. Genes used by either classifier
are marked and cannot serve as independent validation. Within-donor depth-matched
controls provide a descriptive sensitivity check (nearest retained cell in
log-count/log-gene space, with replacement; matching distances are exported).
No P-values treat cells or repeated clonotypes as independent donors.

PANGEA links are rechecked against original processed receptor records and the
previous checksum-verified audit. Unlinked cells remain unknown. GSE149356 lacks
original productive/high-confidence/full-length flags; GSE165045 uses the saved
strict contig QC rule. Samples are library identifiers, not verified donors.
15-organ represents one donor. Its focused view uses existing T-lineage/NK-T
annotations OR strict paired-alpha-beta evidence, independently of gate output.
The full atlas is preserved; cells outside this viewing subset are not deleted.
PANGEA already uses the original T/NK cohort, so no gate-defined crop is applied.
Original UMAP coordinates are never refitted. Study/organ facets share limits.

**Figure contract.** SPL figures separate measured QC and marker associations
from receptor evidence; atlas views show evidence coverage and frozen decisions.
PDFs contain vector points and editable black text. Tables retain all cells,
missing values, denominators and source fields. Complete donor/dataset units
are saved atomically and reused only after input, code and output hash checks.
The authoritative function is in the existing generator; this cell has no
in-memory dependency on preceding notebook cells. Resume by rerunning this cell.
# %% code
from pathlib import Path
import os, runpy
from IPython.display import display, Markdown, Image
_audit_remote=Path('/data/users/kaelynwen/gdT_project/gdt_lr_12model')
_audit_local=Path('/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/gdt_lr_12model')
_audit_project=Path(os.environ.get('GDT_LR_PROJECT',str(_audit_remote if _audit_remote.exists() else _audit_local)))
_audit_result=runpy.run_path(str(_audit_project/'build_21_gate_LRM05_evaluation_15organs.py'))['run_gate_discordance_audit'](_audit_project)
display(Markdown(_audit_result['interpretation']))
for _audit_png in _audit_result['previews']:
    display(Image(filename=_audit_png))
'''


def run_gate_discordance_audit(project):
    """Read-only frozen-model audit; independent, restartable donor/dataset units."""
    import os, sys, json, hashlib, inspect, platform, subprocess, re
    from datetime import datetime, timezone
    import numpy as np
    import pandas as pd
    import anndata as ad
    import joblib
    from scipy import sparse
    from scipy.spatial.distance import cdist
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    project=Path(project)
    name='21_gate_LRM05_evaluation_15organs'
    table=project/'results/tables'/name; pred=project/'predictions'/name
    figdir=project/'results/figures'/name; check=project/'checkpoints'/name; log=project/'logs'/name
    for p in [table,pred,figdir,check,log]: p.mkdir(exist_ok=True,parents=True)
    sys.path.insert(0,str(check))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    prefix='gate_discordance_audit'
    font=next((f for f in ['Arial','Helvetica','Liberation Sans'] if f in {x.name for x in font_manager.fontManager.ttflist}),None)
    if font is None: raise RuntimeError('Install an approved publication font')
    plt.rcParams.update({'font.family':font,'font.weight':'normal','font.size':7,'axes.titlesize':7,
        'axes.titleweight':'normal','axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,
        'legend.fontsize':6,'pdf.fonttype':42,'svg.fonttype':'none','text.color':'black',
        'axes.labelcolor':'black','axes.titlecolor':'black','xtick.color':'black','ytick.color':'black',
        'legend.labelcolor':'black','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6})
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        return h.hexdigest()
    def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,default=str).encode()).hexdigest()
    def js(v,p):
        tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(v,indent=2,default=str));os.replace(tmp,p)
    def save(v,p):
        tmp=Path(str(p)+'.tmp')
        if p.suffix=='.parquet':v.to_parquet(tmp,index=False)
        else:v.to_csv(tmp,index=False)
        os.replace(tmp,p);return p
    params=json.loads((table/'parameters.json').read_text())
    gatepath=project/'models/21_T_NK_lineage_gate_and_frozen_M05_deployment/21_T_NK_lineage_gate_and_frozen_M05_deployment_frozen_lineage_gate.joblib'
    model=joblib.load(gatepath);child=joblib.load(project/'models/optimized_M05.joblib')
    assert sha(gatepath)==params['model_hashes']['gate']
    assert sha(project/'models/optimized_M05.joblib')==params['model_hashes']['optimized_M05']
    rawpath=project/'manifests/CrossAtlas_D2_SPL_cohorts.h5ad'
    atlas=project.parent/'Atlas_dataset 2'
    receptor_paths=[atlas/'adata_TILC_TCR_onlyseq.h5ad',atlas/'adata_TILC_TCRgd_onlyseq.h5ad']
    oldcheck=project/'checkpoints/22_LR_hierarchical_CE'
    oldpred=project/'predictions/22_LR_hierarchical_CE/gate_discordance_audit_cells.parquet'
    oldmanifest=json.loads((oldcheck/'gate_discordance_audit_manifest.json').read_text())
    assert oldmanifest['complete'] and sha(oldpred)==oldmanifest['cell_output_sha256']
    external=Path('/data/users/kaelynwen/Subtype_evaluation/External_1')
    if not external.exists():external=project.parent/'Subtype_evaluation/External_1'
    tcrpath=external/'TCR_Label/GSE149356_Processed_TCR.csv.gz'
    supplements=sorted(oldcheck.glob('GSE165045*csv.gz'))
    inputs=[rawpath,*receptor_paths,gatepath,project/'models/optimized_M05.joblib',oldpred,
        oldcheck/'gate_discordance_audit_manifest.json',tcrpath,*supplements,table/'parameters.json',
        pred/'SPL_LR_hierarchical_predictions.parquet',
        *[pred/f'{d}_gate_discordance_audit_UMAP_source_data.parquet' for d in ['PANGEA','15-organ']]]
    hashes={str(p):sha(p) for p in inputs}
    contract={'inputs':hashes,'code':digest(inspect.getsource(run_gate_discordance_audit)),
        'params':params,'python':sys.version,'font':font,'packages':{'numpy':np.__version__,
        'pandas':pd.__version__,'matplotlib':matplotlib.__version__,'anndata':ad.__version__},
        'seed':17,'no_fit':True,'scope':'all saved paired-gdT SPL; original atlas coordinates',
        'matching':'within donor, nearest standardized log1p(raw UMI), log1p(detected genes), with replacement'}
    signature=digest(contract);mpath=check/f'{prefix}_receptor_manifest.json'
    prior=json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest={'complete':False,'signature':signature,'contract':contract,'units':{},'host':platform.node()}
    js(manifest,mpath)
    def valid(unit):
        u=prior.get('units',{}).get(unit,{})
        if prior.get('signature')==signature and u.get('outputs') and all((project/k).exists() and sha(project/k)==v for k,v in u['outputs'].items()):
            manifest['units'][unit]=u;js(manifest,mpath);print('Validated audit unit:',unit,flush=True);return True
        return False
    def done(unit,outputs,meta=None):
        manifest['units'][unit]={'outputs':{str(p.relative_to(project)):sha(p) for p in outputs},'metadata':meta or {}}
        js(manifest,mpath);print('Saved audit unit:',unit,flush=True)
    def export(fig,stem):
        paths=[];fig.canvas.draw()
        alignment=log/(stem+'_alignment.json')
        require_matplotlib_panel_alignment(fig,json_out=alignment,tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
        paths.append(alignment)
        for ext in ['pdf','png']:
            p=figdir/(stem+'.'+ext);tmp=figdir/(stem+'.tmp.'+ext)
            fig.savefig(tmp,format=ext,dpi=300,facecolor='white');os.replace(tmp,p);paths.append(p)
        plt.close(fig)
        for tool,args in [('audit_pdf_text.py',['--min-pt','5','--json']),('audit_figure_collisions.py',['--json'])]:
            r=subprocess.run([sys.executable,str(check/tool),str(figdir/(stem+'.pdf')),*args],capture_output=True,text=True)
            p=log/(stem+'_'+tool[:-3]+'.json');p.write_text(r.stdout);paths.append(p)
            if r.returncode:raise RuntimeError(r.stdout+'\n'+r.stderr)
        return paths
    genes=['CD3D','CD3E','CD3G','TRDC','TRGC1','TRGC2','TRAC','TRBC1','TRBC2','KLRD1','FCER1G','TYROBP','NKG7','GNLY']
    spl=pd.read_parquet(pred/'SPL_LR_hierarchical_predictions.parquet')
    spl=spl[spl.receptor_evidence_group.eq('paired_gdT')].copy().sort_values('cell_id').reset_index(drop=True)
    assert len(spl)==378 and spl.cell_id.is_unique
    spl['optimized_M05_positive']=spl.p_gdT_given_T.ge(params['thresholds']['optimized_M05'])
    assert int((spl.optimized_M05_positive & ~spl.hard_call.eq('predicted gdT')).sum())==22
    spl['barcode_raw']=spl.cell_id.str.split('|').str[-1]
    spl['audit_group']=np.select([spl.hard_call.eq('predicted gdT'),spl.optimized_M05_positive & spl.hard_call.eq('predicted NK'),spl.optimized_M05_positive],
        ['predicted gdT','optimized_M05 positive; predicted NK','optimized_M05 positive; abstain/out-of-domain'],default='optimized_M05 negative')
    labels=['predicted gdT','optimized_M05 positive; predicted NK','optimized_M05 positive; abstain/out-of-domain']
    colors=['#7B2CBF','#D73027','#E69F00']
    # Reconstruct original receptor slots and retain explicit quality availability.
    evidence=[];obs_frames={};quality=[]
    def present(v):return pd.notna(v) and str(v).strip().lower() not in ['', 'nan','none','na']
    for path in receptor_paths:
        obj=ad.read_h5ad(path,backed='r');o=obj.obs.copy();obj.file.close()
        assert o.index.is_unique and spl.barcode_raw.isin(o.index).all()
        o=o.reindex(spl.barcode_raw).copy();o.index=spl.cell_id
        assert o.donor_id.astype(str).tolist()==spl.donor.astype(str).tolist()
        obs_frames[path.name]=o
        for field in ['productive','high_confidence','full_length','is_cell']:
            quality.append({'dataset':'SPL','source':path.name,'field':field,'available':any(field in c for c in o.columns)})
        for cid,row in o.iterrows():
            for col in [c for c in o if re.fullmatch(r'IR_(VJ|VDJ)_\d+_locus',c)]:
                chain=str(row[col]);slot=col[:-6]
                if chain not in ['TRA','TRB','TRG','TRD']:continue
                rec={'cell_id':cid,'source':path.name,'slot':slot,'chain':chain}
                for field in ['cdr3','cdr3_nt','v_gene','j_gene','expr','expr_raw']:
                    v=row.get(slot+'_'+field);rec[field]=str(v) if present(v) else None
                rec['valid_reported_sequence']=all(present(rec[c]) for c in ['cdr3','v_gene','j_gene'])
                rec['cdr3_nt_available']=present(rec['cdr3_nt']);evidence.append(rec)
    ev=pd.DataFrame(evidence);assert ev.valid_reported_sequence.all()
    counts=ev.groupby(['cell_id','chain']).size().unstack(fill_value=0).reindex(index=spl.cell_id,columns=['TRA','TRB','TRG','TRD'],fill_value=0)
    np.testing.assert_array_equal(counts.to_numpy(),spl[['n_TRA','n_TRB','n_TRG','n_TRD']].to_numpy())
    gdobs=obs_frames[receptor_paths[1].name]
    for c in ['scrublet_score','is_doublet','chain_pairing','manual_annot_v6','n_counts','n_genes']:
        spl['reported_'+c]=gdobs[c].to_numpy()
    spl['single_TRG_TRD']=spl.n_TRG.eq(1)&spl.n_TRD.eq(1)
    spl['paired_CDR3_nt_available']=spl.cell_id.map(ev.groupby('cell_id').cdr3_nt_available.all())
    pairkeys=ev.groupby('cell_id').apply(lambda z:json.dumps(sorted(tuple(r[c] for c in ['chain','cdr3_nt','v_gene','j_gene']) for _,r in z.iterrows())),include_groups=False)
    spl['paired_sequence_key']=spl.cell_id.map(pairkeys)
    spl['same_pair_retained_cells']=0
    for (donor,key),z in spl[spl.single_TRG_TRD & spl.paired_CDR3_nt_available].groupby(['donor','paired_sequence_key'],observed=True):
        spl.loc[z.index,'same_pair_retained_cells']=int(z.hard_call.eq('predicted gdT').sum())
    raw=ad.read_h5ad(rawpath,backed='r')
    pieces=[];effects=[]
    for donor,z in spl.groupby('donor',sort=True,observed=True):
        unit='SPL_'+str(donor)
        cellpath=check/f'SPL_{donor}_{prefix}_receptor_cells.parquet'
        effectpath=check/f'SPL_{donor}_{prefix}_gene_contributions.parquet'
        if valid(unit):
            q=pd.read_parquet(cellpath);assert q.cell_id.tolist()==z.cell_id.tolist();pieces.append(q);effects.append(pd.read_parquet(effectpath));continue
        target=raw[z.cell_id.tolist()].to_memory();x=sparse.csr_matrix(target.layers['counts']);x.eliminate_zeros()
        assert np.isfinite(x.data).all() and (x.data>=0).all() and np.allclose(x.data,np.round(x.data))
        q=z.copy();total=np.asarray(x.sum(axis=1)).ravel();ng=np.asarray((x>0).sum(axis=1)).ravel()
        q['audit_total_counts']=total;q['audit_n_genes_by_counts']=ng
        q['audit_pct_counts_mt']=100*np.asarray(x[:,target.var_names.str.startswith('MT-')].sum(axis=1)).ravel()/total
        for gene in genes:
            ix=np.flatnonzero(target.var_names==gene)
            q['count_'+gene]=np.asarray(x[:,ix].sum(axis=1)).ravel() if len(ix) else np.nan
            q['log1p_'+gene]=np.log1p(1e4*q['count_'+gene]/total)
        # Reproduce the frozen gate exactly, including its panel-only normalization.
        columns=[];resolved=[]
        for gene in model['genes']:
            ix=np.flatnonzero(target.var_names==gene);resolved.append(bool(len(ix)))
            columns.append(sparse.csr_matrix(x[:,ix].sum(axis=1)) if len(ix) else sparse.csr_matrix((len(q),1)))
        panel=sparse.hstack(columns,format='csr',dtype=np.float32)
        panel=sparse.diags((1e4/np.asarray(panel.sum(axis=1)).ravel()).astype(np.float32))@panel;panel.data=np.log1p(panel.data)
        prob=model['model'].predict_proba(panel)[:,1]
        assert np.max(abs(prob-q.p_T.to_numpy()))<1e-7
        scale=model['model'].named_steps['scale'];lr=model['model'].named_steps['lr']
        contribution=scale.transform(panel).multiply(lr.coef_[0]).toarray()
        assert np.max(abs(contribution.sum(axis=1)+lr.intercept_[0]-model['model'].decision_function(panel)))<1e-7
        q['gate_log_odds']=contribution.sum(axis=1)+lr.intercept_[0]
        q['gate_reproduction_max_abs_difference']=np.max(abs(prob-q.p_T.to_numpy()))
        q['margin_to_T_threshold']=q.p_T-params['threshold_T'];q['margin_to_NK_threshold']=q.p_T-params['threshold_NK']
        control=q.hard_call.eq('predicted gdT').to_numpy();focal=q.gate_excluded_M05_positive.to_numpy()
        rows=[]
        for call in ['predicted NK','abstain']:
            m=focal & q.hard_call.eq(call).to_numpy()
            if m.any() and control.any():
                for i,gene in enumerate(model['genes']):rows.append({'donor':donor,'hard_call':call,'gene':gene,'resolved':resolved[i],
                    'n_focal':int(m.sum()),'n_controls':int(control.sum()),'mean_difference':float(contribution[m,i].mean()-contribution[control,i].mean())})
        e=pd.DataFrame(rows,columns=['donor','hard_call','gene','resolved','n_focal','n_controls','mean_difference'])
        q['matched_control_cell_id']='';q['matched_control_distance']=np.nan
        for gene in genes:q['matched_difference_'+gene]=np.nan
        if focal.any() and control.any():
            cov=np.log1p(q[['audit_total_counts','audit_n_genes_by_counts']].to_numpy());sd=cov.std(axis=0);sd[sd==0]=1
            dist=cdist(cov[focal]/sd,cov[control]/sd);idx=dist.argmin(axis=1)
            refs=q.iloc[np.flatnonzero(control)[idx]]
            q.loc[q.index[focal],'matched_control_cell_id']=refs.cell_id.to_numpy();q.loc[q.index[focal],'matched_control_distance']=dist.min(axis=1)
            for gene in genes:q.loc[q.index[focal],'matched_difference_'+gene]=q.loc[q.index[focal],'log1p_'+gene].to_numpy()-refs['log1p_'+gene].to_numpy()
        paths=[save(q,cellpath),save(e,effectpath)]
        done(unit,paths,{'n_cells':len(q),'integer_counts':True,'feature_coverage':float(np.mean(resolved))})
        pieces.append(q);effects.append(e)
    raw.file.close()
    s=pd.concat(pieces,ignore_index=True);e=pd.concat(effects,ignore_index=True)
    outputs=[save(s,pred/f'SPL_{prefix}_receptor_cells.parquet'),save(ev,pred/f'SPL_{prefix}_receptor_sequences.csv'),
        save(pd.DataFrame(quality),table/f'SPL_{prefix}_receptor_quality.csv'),save(s[s.gate_excluded_M05_positive],table/f'SPL_{prefix}_22_cells.csv'),
        save(e,table/f'SPL_{prefix}_gene_contributions.csv')]
    membership=pd.DataFrame({'gene':genes,'in_gate':[g in model['genes'] for g in genes],'in_M05':[g in child['genes'] for g in genes]})
    outputs.append(save(membership,table/f'SPL_{prefix}_feature_membership.csv'))
    donor_summary=s.groupby(['donor','audit_group'],observed=True).agg(n=('cell_id','size'),median_p_T=('p_T','median'),
        median_total_counts=('audit_total_counts','median'),median_n_genes=('audit_n_genes_by_counts','median'),
        median_pct_mt=('audit_pct_counts_mt','median'),single_pair=('single_TRG_TRD','sum'),
        median_scrublet_score=('reported_scrublet_score','median')).reset_index()
    outputs.append(save(donor_summary,table/f'SPL_{prefix}_receptor_by_donor.csv'))
    # Per-donor contrasts, rather than treating cells as independent replicates.
    rows=[]
    for donor,z in s.groupby('donor',observed=True):
        retained=z[z.hard_call.eq('predicted gdT')]
        for call in ['predicted NK','abstain']:
            focal=z[z.gate_excluded_M05_positive & z.hard_call.eq(call)]
            if len(focal)==0 or len(retained)==0:continue
            for gene in genes:rows.append({'donor':donor,'hard_call':call,'gene':gene,'n_focal':len(focal),'n_controls':len(retained),
                'mean_difference':focal['log1p_'+gene].mean()-retained['log1p_'+gene].mean(),
                'depth_matched_mean_difference':focal['matched_difference_'+gene].mean(),
                'focal_detected_fraction':focal['count_'+gene].gt(0).mean(),'control_detected_fraction':retained['count_'+gene].gt(0).mean()})
    markers=pd.DataFrame(rows);outputs.append(save(markers,table/f'SPL_{prefix}_marker_differences.csv'))
    outputs.append(save(e.sort_values('mean_difference').groupby(['donor','hard_call'],sort=False).head(10),table/f'SPL_{prefix}_top_contributions.csv'))
    if not valid('SPL_figures'):
        fig,axes=plt.subplots(2,2,figsize=(7.2,5.7));fig.subplots_adjust(left=.12,right=.97,bottom=.23,top=.9,hspace=.55,wspace=.40)
        fields=[('p_T','Frozen gate P(T)'),('audit_total_counts','Raw UMI counts'),('audit_n_genes_by_counts','Detected genes'),('reported_scrublet_score','Reported Scrublet score')]
        donors=sorted(s.donor.unique());rng=np.random.default_rng(17)
        for ax,(field,title) in zip(axes.flat,fields):
            for j,group in enumerate(labels):
                z=s[s.audit_group.eq(group)];y=np.array([donors.index(d) for d in z.donor])+rng.uniform(-.13,.13,len(z))
                ax.scatter(z[field],y,s=7 if j else 3,c=colors[j],marker=['o','x','^'][j],alpha=.8,linewidths=.6 if j==1 else 0)
            ax.set(yticks=range(len(donors)),yticklabels=donors,xlabel=title,ylabel='Donor');ax.invert_yaxis()
            if field=='p_T':
                ax.axvline(params['threshold_T'],color='black',ls='--',lw=.7);ax.axvline(params['threshold_NK'],color='black',ls=':',lw=.7);ax.set_xlim(0,1)
            if field=='audit_total_counts':
                from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
                ax.set_xscale('log');ax.xaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda value,position:f'{value:g}'))
                ax.xaxis.set_minor_formatter(NullFormatter())
        fig.suptitle('SPL | M05-positive cells with reported TRG + TRD',y=.97,fontsize=8)
        handles=[Line2D([],[],marker=m,ls='',color=c,label=l+f' (n = {int(s.audit_group.eq(l).sum())})',markersize=4) for l,c,m in zip(labels,colors,['o','x','^'])]
        fig.legend(handles=handles,loc='lower center',ncol=1,frameon=False,bbox_to_anchor=(.5,.015))
        paths=export(fig,f'SPL_{prefix}_receptor_QC')
        fig,axes=plt.subplots(1,2,figsize=(7.2,4.8));fig.subplots_adjust(left=.11,right=.95,bottom=.25,top=.86,wspace=.42)
        for ax,(field,title) in zip(axes,[('mean_difference','Same-donor retained controls'),('depth_matched_mean_difference','Depth-matched retained controls')]):
            for j,(call,marker,color) in enumerate([('predicted NK','x','#D73027'),('abstain','^','#E69F00')]):
                z=markers[markers.hard_call.eq(call)]
                yy=np.array([genes.index(g) for g in z.gene])+(j-.5)*.24
                ax.scatter(z[field],yy,s=13,c=color,marker=marker,linewidths=.6 if j==0 else 0)
            ax.axvline(0,color='black',lw=.7);ax.set(yticks=range(len(genes)),yticklabels=genes,xlabel='Excluded minus retained\nmean log-normalized expression',title=title);ax.invert_yaxis()
        bound=float(np.nanmax(np.abs(markers[['mean_difference','depth_matched_mean_difference']].to_numpy())))*1.1
        for ax in axes:ax.set_xlim(-bound,bound)
        fig.suptitle('SPL | marker differences within donor',fontsize=8,y=.96)
        fig.text(.5,.06,'Each point is one donor; descriptive differences, not independent identity evidence.',ha='center',fontsize=6)
        fig.legend(handles=[Line2D([],[],marker=m,ls='',color=c,label=l,markersize=4) for l,c,m in [('predicted NK','#D73027','x'),('abstain','#E69F00','^')]],loc='lower center',bbox_to_anchor=(.5,.105),ncol=2,frameon=False)
        paths+=export(fig,f'SPL_{prefix}_receptor_markers');done('SPL_figures',paths)
    done('SPL_aggregate',outputs)
    # Atlas receptor joins are validated against their primary files.
    old=pd.read_parquet(oldpred)
    atlas_counts=[];view_counts=[];stratified=[]
    for dataset in ['PANGEA','15-organ']:
        unit=dataset+'_receptor_UMAP'
        data=pd.read_parquet(pred/f'{dataset}_gate_discordance_audit_UMAP_source_data.parquet')
        oldpart=old[old.dataset.eq(dataset)].set_index('cell_id').reindex(data.cell_id)
        assert oldpart.index.is_unique and oldpart.dataset.notna().all()
        np.testing.assert_allclose(data[['p_T','p_gdT_given_T']],oldpart[['p_T','p_gdT_given_T']],rtol=0,atol=1e-12)
        assert oldpart.hard_call.tolist()==data.hard_call.tolist()
        for c in ['receptor_evidence','receptor_file_matched','TRD_TRG_with_CDR3','strict_paired_abTCR_reconstructed','strict_1TRG_1TRD','mixed_ab_gd','source_study','source_sample','source_organ','source_annotation','v_gene_TRD','v_gene_TRG']:
            data[c]=oldpart[c].to_numpy()
        if dataset=='PANGEA':
            tcr=pd.read_csv(tcrpath);assert tcr.barcode.is_unique
            sub=data.Series.eq('GSE149356');keys=data.loc[sub,'Sample'].astype(str)+'_'+data.loc[sub,'cell_id'].str.split('@').str[-1].str.replace(r'-1$','',regex=True)
            assert keys.is_unique;joined=tcr.set_index('barcode').reindex(keys)
            support=joined.chain_TRD.eq('TRD')&joined.chain_TRG.eq('TRG')
            for c in ['cdr3_TRD','cdr3_TRG','cdr3_nt_TRD','cdr3_nt_TRG']:support &= joined[c].map(present)
            np.testing.assert_array_equal(support,data.loc[sub,'TRD_TRG_with_CDR3'].fillna(False))
            for c in ['cdr3_TRD','cdr3_TRG','cdr3_nt_TRD','cdr3_nt_TRG']:data.loc[sub,c]=joined[c].to_numpy()
            for path in supplements:
                sample=path.name.split('_')[1];tab=pd.read_csv(path,low_memory=False)
                sub=data.Series.eq('GSE165045')&data.Sample.eq(sample);keys=data.loc[sub,'cell_id'].str.split('@').str[-1]
                ok=np.ones(len(tab),bool)
                for c in ['is_cell','high_confidence','full_length','productive']:ok &= tab[c].astype(str).str.lower().isin(['true','t','1','yes'])
                for c in ['v_gene','j_gene','cdr3','cdr3_nt']:ok &= tab[c].map(present)
                cc=tab.loc[ok].groupby(['barcode','chain']).size().unstack(fill_value=0).reindex(index=keys,columns=['TRA','TRB','TRG','TRD','Multi'],fill_value=0).fillna(0)
                ab=cc.TRA.eq(1)&cc.TRB.eq(1)&cc.TRG.eq(0)&cc.TRD.eq(0)&cc.Multi.eq(0)
                np.testing.assert_array_equal(ab,data.loc[sub,'strict_paired_abTCR_reconstructed'].fillna(False))
            scope=pd.Series(True,index=data.index);scope_rule='Original Level2_TandNK cohort; no additional model-based filter'
        else:
            scope=data.annotation_Tlineage.fillna(False).astype(bool)|data.original_annotation_T_NK_ambiguous.fillna(False).astype(bool)|data.strict_paired_abTCR.fillna(False).astype(bool)
            scope_rule='annotation_Tlineage OR original_annotation_T_NK_ambiguous OR strict_paired_abTCR'
        masks=[data.hard_call.eq('predicted gdT'),data.optimized_M05_positive & data.hard_call.eq('predicted NK'),~data.optimized_M05_positive & data.hard_call.eq('predicted NK'),data.optimized_M05_positive & data.hard_call.isin(['abstain','abstain/out-of-domain'])]
        group=np.select(masks,[1,2,3,4],default=0)
        names=['Other cells',labels[0],labels[1],'optimized_M05 negative; predicted NK',labels[2]]
        palette=['#D9D9D9','#7B2CBF','#D73027','#0057B8','#E69F00']
        data['gate_discordance_audit_UMAP_group']=[names[i] for i in group];data['T_NK_view_eligible']=scope
        has_xy=data[['UMAP1','UMAP2']].notna().all(axis=1)
        for group_name,z in data.groupby('gate_discordance_audit_UMAP_group',observed=True):
            for evidence_name,q in z.groupby('receptor_evidence',dropna=False,observed=True):atlas_counts.append({'dataset':dataset,'group':group_name,'receptor_evidence':evidence_name,'n':len(q)})
        for included in [False,True]:
            for group_name,z in data[scope.eq(included)].groupby('gate_discordance_audit_UMAP_group',observed=True):view_counts.append({'dataset':dataset,'in_T_NK_view':included,'group':group_name,'n_all':len(z),'n_embedded':int(z[['UMAP1','UMAP2']].notna().all(axis=1).sum()),'rule':scope_rule})
        for field in ['source_study','source_sample','source_organ']:
            q=data.groupby([field,'gate_discordance_audit_UMAP_group','receptor_evidence'],dropna=False,observed=True).size().reset_index(name='n')
            q=q.rename(columns={field:'unit'});q['dataset']=dataset;q['unit_type']=field;stratified.append(q)
        out=pred/f'{dataset}_{prefix}_receptor_UMAP.parquet'
        if valid(unit):continue
        paths=[save(data,out)]
        v=data[scope & has_xy].copy();limits=[]
        for c in ['UMAP1','UMAP2']:
            lo,hi=v[c].min(),v[c].max();pad=.025*(hi-lo);limits.append((lo-pad,hi+pad))
        def axes_style(ax):
            ax.set(xlim=limits[0],ylim=limits[1],xlabel='UMAP1',ylabel='UMAP2',xticks=[],yticks=[]);ax.set_aspect('equal',adjustable='box')
        def paint(ax,z):
            background=z[z.gate_discordance_audit_UMAP_group.eq('Other cells')]
            ax.scatter(background.UMAP1,background.UMAP2,s=.7,c=palette[0],alpha=.4,lw=0)
            fore=z[~z.gate_discordance_audit_UMAP_group.eq('Other cells')].sample(frac=1,random_state=17)
            ax.scatter(fore.UMAP1,fore.UMAP2,s=2,c=fore.gate_discordance_audit_UMAP_group.map(dict(zip(names,palette))),alpha=.8,lw=0)
            axes_style(ax)
        fig,axes=plt.subplots(1,2,figsize=(7.2,4.7));fig.subplots_adjust(left=.07,right=.97,bottom=.30,top=.82,wspace=.20)
        paint(axes[0],v);axes[0].set_title(f'Frozen decisions | n = {len(v):,}')
        axes[1].scatter(v.UMAP1,v.UMAP2,s=.5,c='#D9D9D9',alpha=.35,lw=0)
        for col,marker,color in [('TRD_TRG_with_CDR3','o','#7B2CBF'),('strict_paired_abTCR_reconstructed','x','#0057B8')]:
            q=v[v[col].fillna(False).astype(bool)];axes[1].scatter(q.UMAP1,q.UMAP2,s=8,c=color,marker=marker,linewidths=.6 if marker=='x' else 0)
        axes_style(axes[1]);axes[1].set_title('Available receptor support')
        fig.suptitle(dataset+' | T/NK context and receptor coverage',fontsize=8,y=.96)
        handles=[Line2D([],[],marker='o',ls='',color=c,label=n,markersize=3) for n,c in zip(names[1:],palette[1:])]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.12),ncol=2,frameon=False)
        fig.text(.5,.065,'Right: purple circles = processed TRG + TRD; blue crosses = strict paired alpha-beta.',ha='center',fontsize=6)
        fig.text(.5,.035,'Unmarked cells have no linked qualifying evidence; this is not evidence of NK identity.',ha='center',fontsize=6)
        paths+=export(fig,f'{dataset}_{prefix}_receptor_UMAP')
        # All studies/organ strata receive a facet; no selection on favorable results.
        field='source_study' if dataset=='PANGEA' else 'source_organ'
        values=sorted(v[field].fillna('missing').unique())
        for start in range(0,len(values),6):
            fig,axes=plt.subplots(2,3,figsize=(7.2,5.7));fig.subplots_adjust(left=.06,right=.98,bottom=.18,top=.90,wspace=.22,hspace=.38)
            for ax,value in zip(axes.flat,values[start:start+6]):
                q=v[v[field].fillna('missing').eq(value)];paint(ax,q)
                nred=int(q.gate_discordance_audit_UMAP_group.eq(labels[1]).sum());ax.set_title(str(value)+f'\nn = {len(q):,}; excluded NK = {nred:,}')
            for ax in list(axes.flat)[len(values[start:start+6]):]:ax.set_visible(False)
            fig.suptitle(dataset+' | '+field+' | original shared UMAP coordinates',fontsize=8,y=.98)
            fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.025),ncol=2,frameon=False)
            paths+=export(fig,f'{dataset}_{prefix}_receptor_UMAP_{field}_{start//6+1:02d}')
        done(unit,paths,{'view_rule':scope_rule,'all_cells':len(data),'in_view':int(scope.sum()),'embedded_in_view':len(v)})
    outputs=[save(pd.DataFrame(atlas_counts),table/f'{prefix}_receptor_by_group.csv'),
        save(pd.DataFrame(view_counts),table/f'{prefix}_receptor_view_counts.csv'),
        save(pd.concat(stratified,ignore_index=True),table/f'{prefix}_receptor_by_source.csv')]
    focal=s[s.gate_excluded_M05_positive]
    interpretation=(f'### Observed receptor audit\n\nSPL: {len(focal)} additional exclusions comprise '
        f'{int(focal.hard_call.eq("predicted NK").sum())} NK calls and {int(focal.hard_call.eq("abstain").sum())} abstentions. '
        f'All have reported TRG+TRD locus/CDR3/V/J support, with {int(focal.paired_CDR3_nt_available.sum())} having nucleotide CDR3 fields. '
        f'{int(focal.single_TRG_TRD.sum())} have one TRG and one TRD; the remainder have extra TRG. '
        f'{int((focal.single_TRG_TRD & focal.same_pair_retained_cells.gt(0)).sum())} excluded single-pair cells share an exact paired nucleotide/V/J key with a retained cell from the same donor. '
        'Productive/high-confidence/full-length flags are unavailable; reported doublet flags do not prove singlets. '
        'Raw integer count QC is recomputed. Feature availability is uniform within SPL and does not establish the cause of individual exclusions.\n\n'
        'PANGEA receptor coverage and all unresolved candidates are enumerated in the group/source tables. '
        'No receptor match is not negative evidence. The T/NK atlas subset is only a viewing rule; full atlas predictions and earlier figures remain intact. '
        'Donor/organ/sample contrasts are descriptive. No model, threshold, evaluation label or embedding was changed.')
    report=table/f'{prefix}_receptor_interpretation.md';report.write_text(interpretation);outputs.append(report)
    done('summary',outputs)
    manifest['complete']=True;manifest['completed_utc']=datetime.now(timezone.utc).isoformat();js(manifest,mpath)
    files=[]
    for u in manifest['units'].values():
        for rel,h in u['outputs'].items():files.append({'local_path':'/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/gdt_lr_12model/'+rel,'Woody_path':'/data/users/kaelynwen/gdT_project/gdt_lr_12model/'+rel,'sha256':h})
    save(pd.DataFrame(files).drop_duplicates('local_path'),table/f'{prefix}_receptor_output_manifest.csv')
    return {'interpretation':interpretation,'previews':[str(figdir/f'{x}_{prefix}_{y}.png') for x,y in [('SPL','receptor_QC'),('SPL','receptor_markers'),('PANGEA','receptor_UMAP'),('15-organ','receptor_UMAP')]]}


SOURCE_AUDIT = r'''
# %% markdown
## Gate-discordance audit: GSE164378 training labels and protein evidence

This audit inspects the frozen gate's **training reference**, not a new external
test set. It uses saved donor-held-out RNA predictions, original annotations,
and measured ADT counts. The protein panel contributed to the original source
annotations: agreement with proteins is an annotation-consistency check, not
independent validation of those annotations. No label or model is changed.

The exact 3P/5P cell identifiers are compared before any receptor linkage.
Shared donor, time or a stripped barcode cannot establish the same cell across
separate capture libraries. The official antibody table maps `TCR-1` to the B1
TCR gamma-delta reagent and `TCR-2` to IP26 TCR alpha-beta. Exact source feature
names are retained. Per-cell ADT values are transformed as
`log1p(count / exp(mean(log1p(all 228 ADT counts))))`; raw counts are also exported.

For review only, within each lane, both CD3 reagents plus TCR-1 (or TCR-2) are
compared with the 95th, 99th and 99.5th percentiles in source-labelled B/Mono
cells. These are empirical background comparisons, **not validated positivity
cutoffs or replacement labels**. They may be affected by ambient ADT and
doublets. No quantile is chosen to improve the gate. Reported nCount/nFeature
fields are retained as reported, not reinterpreted as newly computed QC.

All 8 donors and all source subtypes remain visible. OOF means the donor was
held out during model fitting; the operating thresholds were selected from the
pooled OOF predictions, so these are diagnostic threshold-selection results,
not a fresh independent estimate of threshold performance. Repeated time points
are not independent donors. Figures are descriptive with no cell-level tests.

Official inputs are downloaded atomically into the existing GSE164378 folder.
Each donor is checkpointed with code/configuration/input/output hashes and a
progress manifest. Rerun the following self-contained cell to resume safely.
Raw inputs are required on Woody; local results can be viewed without loading
the large training matrix. Environment versions and both local/Woody output
paths are included in the manifest. Earlier evaluation outputs are preserved.
# %% code
from pathlib import Path
import os, runpy
from IPython.display import display, Markdown, Image
_source_remote=Path('/data/users/kaelynwen/gdT_project/gdt_lr_12model')
_source_local=Path('/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/gdt_lr_12model')
_source_project=Path(os.environ.get('GDT_LR_PROJECT',str(_source_remote if _source_remote.exists() else _source_local)))
_source_result=runpy.run_path(str(_source_project/'build_21_gate_LRM05_evaluation_15organs.py'))['run_gate_discordance_audit_source'](_source_project)
display(Markdown(_source_result['interpretation']))
display(_source_result['donor_table'])
for _source_png in _source_result['previews']:
    display(Image(filename=_source_png))
'''


def run_gate_discordance_audit_source(project):
    """Audit existing GSE164378 annotations without altering models or labels."""
    import os, sys, json, hashlib, inspect, platform, importlib.metadata, gzip
    from datetime import datetime, timezone
    import requests, h5py
    import numpy as np
    import pandas as pd
    from anndata.io import read_elem
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    project=Path(project); name='21_gate_LRM05_evaluation_15organs'
    prefix='GSE164378_gate_discordance_audit'
    data=project/'data/GSE164378'; table=project/'results/tables'/name
    pred=project/'predictions'/name; figdir=project/'results/figures'/name
    check=project/'checkpoints'/name
    for p in [data,table,pred,figdir,check]:p.mkdir(exist_ok=True,parents=True)
    def sha(p):
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        return h.hexdigest()
    def js(v,p):
        t=Path(str(p)+'.tmp');t.write_text(json.dumps(v,indent=2,default=str));os.replace(t,p)
    def save(v,p):
        t=Path(str(p)+'.tmp')
        if p.suffix=='.parquet':v.to_parquet(t,index=False)
        else:v.to_csv(t,index=False)
        os.replace(t,p);return p
    urls={f:'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164378/suppl/'+f for f in
          ['GSE164378_sc.meta.data_3P.csv.gz','GSE164378_sc.meta.data_5P.csv.gz','GSE164378_Antibody_HTO_info.xlsx']}
    for acc,chain in [('GSM5008743','TCRab'),('GSM5008744','TCRgd')]:
        for b in [1,2]:
            f=f'{acc}_{chain}_batch{b}_annotations.csv.gz'
            urls[f]=f'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5008nnn/{acc}/suppl/{f}'
    for f,u in urls.items():
        p=data/f
        if not p.exists():
            r=requests.get(u,timeout=120);r.raise_for_status()
            t=Path(str(p)+'.tmp');t.write_bytes(r.content)
            if f.endswith('.gz'):gzip.decompress(r.content)
            os.replace(t,p)
        if f.endswith('.gz'):
            with gzip.open(p,'rb') as z:
                while z.read(1024*1024):pass
    h5=data/'pbmc_seurat_v4.h5ad'
    oofpath=project/'predictions/21_T_NK_lineage_gate_and_frozen_M05_deployment/GSE164378_T_NK_gate_donor_blocked_OOF.parquet'
    param_path=table/'parameters.json';params=json.loads(param_path.read_text())
    gatepath=project/'models/21_T_NK_lineage_gate_and_frozen_M05_deployment/21_T_NK_lineage_gate_and_frozen_M05_deployment_frozen_lineage_gate.joblib'
    assert sha(gatepath)==params['model_hashes']['gate']
    inputs=[h5,oofpath,param_path,gatepath]+[data/f for f in urls]
    hashes={str(p.relative_to(project)):sha(p) for p in inputs}
    environment={'python':sys.version,'platform':platform.platform(),'packages':{k:importlib.metadata.version(k) for k in ['pandas','numpy','h5py','anndata','matplotlib','requests']}}
    contract={'input_hashes':hashes,'function':inspect.getsource(run_gate_discordance_audit_source),
              'environment':environment,'quantiles':[.95,.99,.995],'threshold_T':params['threshold_T'],'threshold_NK':params['threshold_NK']}
    signature=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    mp=check/f'{prefix}_manifest.json'
    previous=json.loads(mp.read_text()) if mp.exists() else {}
    manifest={'signature':signature,'input_hashes':hashes,'input_urls':urls,'environment':environment,
              'parameters':{k:contract[k] for k in ['quantiles','threshold_T','threshold_NK']},'units':{},'complete':False}
    if previous.get('signature')==signature:manifest['units']=previous.get('units',{})
    def valid(unit):
        u=manifest['units'].get(unit,{})
        return bool(u.get('outputs')) and all((project/p).exists() and sha(project/p)==h for p,h in u['outputs'].items())
    def done(unit,paths):
        manifest['units'][unit]={'outputs':{str(p.relative_to(project)):sha(p) for p in paths}}
        js(manifest,mp);print('Complete:',unit,flush=True)
    with h5py.File(h5) as h:
        obs=read_elem(h['obs']);adt=read_elem(h['obsm/protein_counts'])
    assert obs.index.is_unique and obs.index.equals(adt.index)
    values=adt.to_numpy(dtype=float)
    assert np.isfinite(values).all() and (values>=0).all() and np.equal(values,np.round(values)).all()
    assert adt.shape[1]==228 and obs.donor.nunique()==8
    m3=pd.read_csv(data/'GSE164378_sc.meta.data_3P.csv.gz',index_col=0)
    m5=pd.read_csv(data/'GSE164378_sc.meta.data_5P.csv.gz',index_col=0)
    assert m3.index.is_unique and m5.index.is_unique and obs.index.difference(m3.index).empty
    for c in ['donor','lane','celltype.l1','celltype.l2','celltype.l3']:
        assert obs[c].astype(str).eq(m3.loc[obs.index,c].astype(str)).all(),c
    assert len(obs.index.intersection(m5.index))==0
    antibodies=pd.read_excel(data/'GSE164378_Antibody_HTO_info.xlsx',sheet_name="228 antibody panel (3')")
    antibodies['source_feature']=antibodies['#protein'].str.replace('_','-',regex=False)
    for feature,clone in [('TCR-1','B1'),('TCR-2','IP26')]:
        assert antibodies.loc[antibodies.source_feature.eq(feature),'Clone'].tolist()==[clone]
    markers=['CD3-1','CD3-2','TCR-1','TCR-2','TCR-V-2','TCR-V-9','CD16','CD56-1','CD56-2','CD335','CD337','CD2']
    assert set(markers).issubset(adt.columns)
    clr=np.log1p(values/np.exp(np.log1p(values).mean(axis=1))[:,None])
    clr=pd.DataFrame(clr,index=adt.index,columns=adt.columns)
    oof=pd.read_parquet(oofpath).set_index('cell_id')
    assert oof.index.is_unique and oof.index.difference(obs.index).empty
    assert oof.donor.astype(str).equals(oof.outer_donor.astype(str))
    assert oof.donor.astype(str).eq(obs.loc[oof.index,'donor'].astype(str)).all()
    assert oof.source_celltype_l1.astype(str).eq(obs.loc[oof.index,'celltype.l1'].astype(str)).all()
    assert np.isfinite(oof.p_T_lineage_OOF).all() and oof.p_T_lineage_OOF.between(0,1).all()
    rows=[];cutoffs=[]
    for donor in sorted(obs.donor.astype(str).unique()):
        out=check/f'{prefix}_{donor}_cells.parquet';cuts=check/f'{prefix}_{donor}_protein_thresholds.csv'
        if valid(donor):
            rows.append(pd.read_parquet(out));cutoffs.append(pd.read_csv(cuts));print('Reuse:',donor,flush=True);continue
        z=obs.loc[obs.donor.astype(str).eq(donor)].copy()
        z.insert(0,'cell_id',z.index.astype(str));z['included_in_gate_OOF']=z.index.isin(oof.index)
        z['p_T_lineage_OOF']=oof.p_T_lineage_OOF.reindex(z.index)
        z['gate_call']=np.select([~z.included_in_gate_OOF,z.p_T_lineage_OOF.ge(params['threshold_T']),z.p_T_lineage_OOF.le(params['threshold_NK'])],
                                ['not in gate OOF','T-lineage eligible','predicted NK'],default='abstain/out-of-domain')
        z['paired_receptor_evidence']='not measured in this 3P capture; 5P is a separate cell set'
        for marker in markers:
            z[marker+'_raw']=adt.loc[z.index,marker];z[marker+'_CLR']=clr.loc[z.index,marker]
        donor_cuts=[]
        for q in [.95,.99,.995]:
            # Thresholds use all B/Mono cells in a lane, independent of gate calls.
            thresholds={}
            for lane in z.lane.astype(str).unique():
                bg=obs['lane'].astype(str).eq(lane)&obs['celltype.l1'].isin(['B','Mono'])
                if int(bg.sum())<100:raise ValueError('Insufficient lane background')
                thresholds[lane]=clr.loc[bg,['CD3-1','CD3-2','TCR-1','TCR-2']].quantile(q)
                donor_cuts.extend({'donor':donor,'lane':lane,'quantile':q,'feature':f,'cutoff_CLR':v,'n_background':int(bg.sum())} for f,v in thresholds[lane].items())
            above={f: z[f+'_CLR'].gt(z.lane.astype(str).map({lane:v[f] for lane,v in thresholds.items()})) for f in ['CD3-1','CD3-2','TCR-1','TCR-2']}
            for receptor in ['TCR-1','TCR-2']:
                z[f'CD3-1_CD3-2_{receptor}_above_background_q{q}']=above['CD3-1']&above['CD3-2']&above[receptor]
        save(z.reset_index(drop=True),out);save(pd.DataFrame(donor_cuts),cuts);done(donor,[out,cuts])
        rows.append(z.reset_index(drop=True));cutoffs.append(pd.DataFrame(donor_cuts))
    cells=pd.concat(rows,ignore_index=True);eligible=cells[cells.included_in_gate_OOF]
    assert len(eligible)==len(oof) and cells.cell_id.is_unique
    outputs=[save(cells,pred/f'{prefix}_source_cells.parquet'),save(pd.concat(cutoffs,ignore_index=True),table/f'{prefix}_protein_thresholds.csv'),
             save(antibodies,table/f'{prefix}_antibody_mapping.csv')]
    summaries=[]
    for fields in [['donor','celltype.l1','celltype.l2'],['donor','celltype.l3']]:
        a=cells.groupby(fields+['gate_call'],observed=True).size().rename('n').reset_index()
        a['denominator_all_source']=a.groupby(fields,observed=True)['n'].transform('sum')
        a['denominator_OOF']=a['n'].where(a.gate_call.ne('not in gate OOF'),0).groupby([a[c] for c in fields],observed=True).transform('sum')
        a['fraction_of_OOF']=np.where(a.gate_call.ne('not in gate OOF'),a.n/a.denominator_OOF.replace(0,np.nan),np.nan)
        fname='source_subtype_by_donor' if 'celltype.l2' in fields else 'source_celltype_l3_by_donor'
        outputs.append(save(a,table/f'{prefix}_{fname}.csv'));summaries.append(a)
    gd=cells[cells['celltype.l2'].eq('gdT')];eg=gd[gd.included_in_gate_OOF]
    donor=pd.crosstab(gd.donor,gd.gate_call).reindex(columns=['T-lineage eligible','predicted NK','abstain/out-of-domain','not in gate OOF'],fill_value=0)
    donor['n_source_gdT']=donor.sum(axis=1);donor['n_OOF_gdT']=donor.n_source_gdT-donor['not in gate OOF']
    donor['T_eligible_fraction']=donor['T-lineage eligible']/donor.n_OOF_gdT
    tcounts=eligible[eligible['celltype.l1'].isin(['CD4 T','CD8 T','other T'])].groupby('donor',observed=True).size()
    donor['gdT_fraction_of_source_T_OOF']=donor.n_OOF_gdT/tcounts
    outputs.append(save(donor.reset_index(),table/f'{prefix}_gdT_by_donor.csv'))
    group=['donor','celltype.l1','celltype.l2','gate_call']
    med=cells.groupby(group,observed=True)[[f+'_CLR' for f in markers]+['nCount_RNA','nFeature_RNA','nCount_ADT','nFeature_ADT']].median().reset_index()
    outputs.append(save(med,table/f'{prefix}_protein_QC_by_donor.csv'))
    flags=[c for c in cells if c.startswith('CD3-1_CD3-2_')]
    review=cells.groupby(group,observed=True)[flags].sum().reset_index()
    review['n_cells']=cells.groupby(group,observed=True).size().to_numpy()
    outputs.append(save(review,table/f'{prefix}_protein_review_counts.csv'))
    focal=cells['celltype.l1'].eq('NK') & cells[flags].any(axis=1)
    outputs.append(save(cells.loc[focal],table/f'{prefix}_NK_protein_review_cells.csv'))
    # Receptor files are deliberately not linked across 3P/5P capture libraries.
    receptor=[]
    for f in urls:
        if 'annotations.csv' not in f:continue
        d=pd.read_csv(data/f);ok=pd.Series(True,index=d.index)
        for c in ['is_cell','high_confidence','full_length','productive']:ok &= d[c].astype(str).str.lower().eq('true')
        for c in ['v_gene','j_gene','cdr3','cdr3_nt']:ok &= d[c].notna() & ~d[c].astype(str).str.lower().isin(['none','nan',''])
        chains=d.loc[ok].groupby('barcode').chain.agg(set)
        pair=('TRG','TRD') if 'TCRgd' in f else ('TRA','TRB')
        receptor.append({'file':f,'n_contigs':len(d),'n_unique_barcodes':d.barcode.nunique(),'n_strict_contigs':int(ok.sum()),
                         'n_barcodes_with_strict_pair':int(chains.map(lambda s:set(pair)<=s).sum()),
                         'library':'5P','linked_training_cells':0,'link_status':'separate 5P capture; no cell-level training join'})
    outputs.append(save(pd.DataFrame(receptor),table/f'{prefix}_receptor_coverage.csv'))
    provenance=[{'comparison':'source IDs versus official 3P','n_source':len(obs),'n_reference':len(m3),'exact_ID_overlap':len(obs.index.intersection(m3.index))},
                {'comparison':'source IDs versus official 5P','n_source':len(obs),'n_reference':len(m5),'exact_ID_overlap':len(obs.index.intersection(m5.index))}]
    outputs.append(save(pd.DataFrame(provenance),table/f'{prefix}_cell_linkage.csv'))
    qc=pd.DataFrame([
        {'field':f,'n_different_training_vs_GEO_3P':int((~obs[f].eq(m3.loc[obs.index,f])).sum())} for f in ['nCount_RNA','nFeature_RNA','nCount_ADT','nFeature_ADT']])
    outputs.append(save(qc,table/f'{prefix}_metadata_consistency.csv'))
    done('aggregate_tables',outputs)
    font=next((f for f in ['Arial','Helvetica','Liberation Sans'] if f in {x.name for x in font_manager.fontManager.ttflist}),None)
    if font is None:raise RuntimeError('No permitted font')
    plt.rcParams.update({'font.family':font,'font.size':8,'font.weight':'normal','axes.titleweight':'normal','pdf.fonttype':42,
                        'text.color':'black','axes.labelcolor':'black','axes.titlecolor':'black','xtick.color':'black','ytick.color':'black',
                        'legend.labelcolor':'black','axes.spines.top':False,'axes.spines.right':False})
    previews=[];figs=[]
    def export(fig,suffix):
        for ext in ['pdf','png']:
            p=figdir/f'{prefix}_{suffix}.{ext}';tmp=Path(str(p)+'.tmp');fig.savefig(tmp,format=ext,dpi=250);os.replace(tmp,p);figs.append(p)
            if ext=='png':previews.append(str(p))
        plt.close(fig)
    calls=['T-lineage eligible','predicted NK','abstain/out-of-domain'];colors=['#0072B2','#D55E00','#E69F00']
    fig,ax=plt.subplots(figsize=(7.2,4.5));fig.subplots_adjust(left=.13,right=.92,bottom=.23,top=.86)
    left=np.zeros(len(donor))
    for call,color in zip(calls,colors):
        vals=donor[call].to_numpy()/donor.n_OOF_gdT.to_numpy()*100
        ax.barh(donor.index.astype(str),vals,left=left,color=color,edgecolor='none',linewidth=0,label=call);left+=vals
    for i,(_,r) in enumerate(donor.iterrows()):ax.text(101,i,f"{int(r.n_OOF_gdT):,}",va='center',fontsize=7)
    ax.set_xlim(0,100);ax.set_xlabel('Percentage of source gdT in gate OOF');ax.set_ylabel('Held-out donor')
    ax.set_title('GSE164378 | source-annotated gdT | frozen OOF decisions',pad=20)
    ax.text(1.015,1.015,'n',transform=ax.transAxes);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,frameon=False,fontsize=7)
    fig.text(.5,.03,'Denominator excludes cells not in the gate cohort; source annotations are not receptor truth.',ha='center',fontsize=7)
    export(fig,'gdT_by_donor')
    masks=[eligible['celltype.l1'].eq('NK')]+[eligible['celltype.l2'].eq('gdT')&eligible.gate_call.eq(c) for c in calls]
    labels=['source NK','gdT\nT eligible','gdT\nNK call','gdT\nabstain'];palette=['#999999']+colors
    fig,axes=plt.subplots(2,3,figsize=(7.2,5.8));fig.subplots_adjust(left=.09,right=.98,bottom=.18,top=.86,wspace=.37,hspace=.70)
    for ax,m in zip(axes.flat,['CD3-1','CD3-2','TCR-1','TCR-2','CD56-1','CD16']):
        box=ax.boxplot([eligible.loc[v,m+'_CLR'].to_numpy() for v in masks],patch_artist=True,showfliers=False,
                       medianprops={'color':'black','linewidth':.8},boxprops={'linewidth':0},whiskerprops={'linewidth':.6},capprops={'linewidth':.6})
        for patch,c in zip(box['boxes'],palette):patch.set_facecolor(c);patch.set_edgecolor('none')
        ax.set_xticks(range(1,5),labels,fontsize=7);ax.set_title(m);ax.set_ylabel('ADT CLR',fontsize=7)
    fig.suptitle('GSE164378 | measured protein by source label and frozen gate call',y=.98,fontsize=10)
    fig.text(.5,.92,'TCR-1: gamma-delta (B1); TCR-2: alpha-beta (IP26)',ha='center',fontsize=8)
    fig.text(.5,.08,'Group n: '+', '.join(str(int(v.sum())) for v in masks)+'; boxes: median / IQR; whiskers: 1.5 IQR.',ha='center',fontsize=7)
    fig.text(.5,.045,'Pooled descriptive distributions; donor medians and all cells are exported. No identity relabeling.',ha='center',fontsize=7)
    export(fig,'protein')
    fig,axes=plt.subplots(2,4,figsize=(8,5.6));fig.subplots_adjust(left=.08,right=.98,bottom=.19,top=.90,wspace=.42,hspace=.40)
    # Every OOF gdT is shown; no sampling or rasterization.
    for ax,(d,z) in zip(axes.flat,eg.groupby('donor',observed=True,sort=True)):
        for c,color,marker in zip(calls,colors,['o','x','^']):
            q=z[z.gate_call.eq(c)];ax.scatter(q.p_T_lineage_OOF,q['TCR-1_CLR'],s=8,c=color,marker=marker,linewidths=.5,label=c)
        ax.axvline(params['threshold_T'],c='black',ls='--',lw=.6);ax.axvline(params['threshold_NK'],c='black',ls=':',lw=.6)
        ax.set_xlim(0,1);ax.set_ylim(0,eg['TCR-1_CLR'].max()*1.06);ax.set_title(str(d)+f' (n={len(z)})',fontsize=8)
    fig.supxlabel('Held-out-donor RNA gate score P(T)',y=.09);fig.supylabel('Measured TCR-1 ADT CLR',x=.01)
    handles,labs=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',ncol=3,frameon=False,fontsize=7)
    fig.suptitle('GSE164378 | source gdT | protein evidence versus RNA gate',fontsize=10)
    export(fig,'gdT_protein_by_donor');done('figures',figs)
    nk=eligible[eligible['celltype.l1'].eq('NK')]
    flag='CD3-1_CD3-2_TCR-1_above_background_q0.99'
    counts=eg.gate_call.value_counts();n_gd_review=int(eg.loc[eg.gate_call.ne(calls[0]),flag].sum());n_nk_review=int(nk[flag].sum())
    interpretation=(f'### GSE164378 training-label audit\n\n'
        f'All {len(obs):,} training-object cell IDs match official 3P metadata, with 0 matching 5P IDs. '
        'The public TCR files describe separate 5P capture libraries; none are assigned to the 3P training cells. '
        'Matching donor/time or stripping library prefixes would not establish same-cell receptor evidence.\n\n'
        f'The source object contains {len(gd):,} gdT-labelled cells; {len(eg):,} entered the saved gate OOF cohort. '
        f'Of these, {counts.get(calls[0],0):,} pass ({counts.get(calls[0],0)/len(eg):.2%}), '
        f'{counts.get(calls[1],0):,} are predicted NK, and {counts.get(calls[2],0):,} abstain. '
        f'P8 retains {int(donor.loc["P8",calls[0]])}/{int(donor.loc["P8","n_OOF_gdT"])} '
        f'({donor.loc["P8","T_eligible_fraction"]:.2%}).\n\n'
        f'At the descriptive lane-specific 99th-percentile B/Mono background comparison, '
        f'{n_gd_review} gate-excluded source gdT cells and {n_nk_review} source NK cells in the OOF cohort '
        'have both CD3 reagents and TCR-1 above background. These are review candidates, not confirmed identities '
        'or estimates of annotation error. The 95th and 99.5th percentile sensitivity results are exported. '
        'Background/antibody specificity, mixed droplets and original annotation use of proteins limit interpretation.\n\n'
        'Training-object and GEO ADT summary fields differ and are exported separately in the consistency table; '
        'the analysis uses the measured 228-column protein matrix in the exact training object. '
        'Source-label concordance cannot establish external biological accuracy. Pooled OOF-selected thresholds '
        'and original protein-informed labels prevent an independent validation claim. No model, cutoff or annotation was changed.')
    report=table/f'{prefix}_interpretation.md';tmp=Path(str(report)+'.tmp');tmp.write_text(interpretation);os.replace(tmp,report);done('interpretation',[report])
    manifest['complete']=True;manifest['completed_utc']=datetime.now(timezone.utc).isoformat();js(manifest,mp)
    paths=[project/p for u in manifest['units'].values() for p in u['outputs']]+[mp]+[data/f for f in urls]
    records=[{'local_path':'/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/gdt_lr_12model/'+str(p.relative_to(project)),
              'Woody_path':'/data/users/kaelynwen/gdT_project/gdt_lr_12model/'+str(p.relative_to(project)),'sha256':sha(p)} for p in paths]
    save(pd.DataFrame(records).drop_duplicates('local_path'),table/f'{prefix}_output_manifest.csv')
    return {'interpretation':interpretation,'donor_table':donor.reset_index(),'previews':previews}



P8_AUDIT = r'''
# %% markdown
## Gate-discordance audit: P8, gdT_2 and gdT_4

This diagnostic uses the existing source annotations without relabeling cells.
It separates donor/subtype composition, time, lane and measured RNA depth.
Within donor, source subtype, time and lane, excluded cells are paired to retained
cells by nearest RNA depth (log1p UMI and detected genes, standardized within
donor; maximum absolute difference 0.5 SD in each covariate). Matching is greedy,
deterministic and without replacement. Unmatched cells are exported and are not
represented by the matched comparison. This is descriptive covariate balancing,
not a causal experiment or an independent identity validation. The 0.25/0.5/1.0
SD sensitivity analysis uses the same fixed specification, not outcome selection.

Only the P8 outer-fold model is reconstructed from the original functions,
ordered training cells, saved fold gene panel and selected C. Its probabilities
must reproduce saved OOF probabilities within 1e-6 before any feature attribution
is reported. No deployment model, threshold or label is changed. Contributions
are coefficient times scaled log-normalized expression; their sum plus intercept
must reproduce the logit. Group differences explain the model mathematically,
not causal biology. RNA markers used by the gate are explicitly marked.

All donor units and the P8 reconstruction are hash-validated checkpoints. The
self-contained cell can be rerun to resume. Figures and tables remain diagnostic:
source proteins informed source annotation, and frozen cutoffs used pooled OOF.
# %% code
from pathlib import Path
import os, runpy
from IPython.display import display, Markdown, Image
_p8_remote=Path('/data/users/kaelynwen/gdT_project/gdt_lr_12model')
_p8_local=Path('/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/gdt_lr_12model')
_p8_project=Path(os.environ.get('GDT_LR_PROJECT',str(_p8_remote if _p8_remote.exists() else _p8_local)))
_p8_result=runpy.run_path(str(_p8_project/'build_21_gate_LRM05_evaluation_15organs.py'))['run_gate_discordance_audit_P8'](_p8_project)
display(Markdown(_p8_result['interpretation']))
for _p8_png in _p8_result['previews']:display(Image(filename=_p8_png))
'''


def run_gate_discordance_audit_P8(project):
    import os, sys, ast, json, hashlib, inspect, platform, importlib.metadata
    from datetime import datetime, timezone
    import numpy as np
    import pandas as pd
    import anndata as ad
    import joblib
    from scipy import sparse
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    project=Path(project); name='21_gate_LRM05_evaluation_15organs'
    prefix='GSE164378_gate_discordance_audit_P8'
    table=project/'results/tables'/name; figdir=project/'results/figures'/name; check=project/'checkpoints'/name
    def sha(p):
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        return h.hexdigest()
    def js(v,p):
        t=Path(str(p)+'.tmp');t.write_text(json.dumps(v,indent=2,default=str));os.replace(t,p)
    def save(v,p):
        t=Path(str(p)+'.tmp')
        if p.suffix=='.parquet':v.to_parquet(t,index=False)
        else:v.to_csv(t,index=False)
        os.replace(t,p);return p
    h5=project/'data/GSE164378/pbmc_seurat_v4.h5ad'
    cp=project/'predictions'/name/'GSE164378_gate_discordance_audit_source_cells.parquet'
    op=project/'predictions/21_T_NK_lineage_gate_and_frozen_M05_deployment/GSE164378_T_NK_gate_donor_blocked_OOF.parquet'
    oldnb=project/'notebooks/21_T_NK_lineage_gate_and_frozen_M05_deployment.ipynb'
    util=project/'src/gdt_lr_utils.py'
    oof=pd.read_parquet(op).set_index('cell_id');held=oof[oof.donor.eq('P8')]
    assert held.selected_C.nunique()==1
    C=float(held.selected_C.iloc[0])
    fp=project/f'checkpoints/21_T_NK_lineage_gate_and_frozen_M05_deployment/outer_P8__C_{C:g}.joblib'
    fold=joblib.load(fp);genes=fold['genes']
    assert held.feature_hash.eq(fold['feature_hash']).all() and held.run_signature.eq(fold['run_signature']).all()
    inputs=[h5,cp,op,oldnb,util,fp]
    env={'python':sys.version,'platform':platform.platform(),'packages':{k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','anndata','matplotlib']}}
    hashes={str(p.relative_to(project)):sha(p) for p in inputs}
    contract={'inputs':hashes,'function':inspect.getsource(run_gate_discordance_audit_P8),'environment':env,'calipers':[.25,.5,1.0]}
    sig=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest();mp=check/f'{prefix}_manifest.json'
    prior=json.loads(mp.read_text()) if mp.exists() else {}
    manifest={'signature':sig,'input_hashes':hashes,'environment':env,'units':prior.get('units',{}) if prior.get('signature')==sig else {},'complete':False}
    def valid(k):
        u=manifest['units'].get(k,{})
        return bool(u.get('outputs')) and all((project/p).exists() and sha(project/p)==h for p,h in u['outputs'].items())
    def done(k,paths):
        manifest['units'][k]={'outputs':{str(p.relative_to(project)):sha(p) for p in paths}};js(manifest,mp);print('Complete:',k,flush=True)
    cells=pd.read_parquet(cp).set_index('cell_id');g=cells[cells.included_in_gate_OOF & cells['celltype.l2'].eq('gdT')].copy()
    calls=['T-lineage eligible','predicted NK','abstain/out-of-domain'];g['excluded']=g.gate_call.ne(calls[0])
    outputs=[]
    for fields,suffix in [(['donor','celltype.l3'],'subtype'),(['donor','time'],'time'),(['donor','lane'],'lane'),(['donor','celltype.l3','time','lane'],'strata')]:
        z=g.groupby(fields+['gate_call'],observed=True).size().rename('n').reset_index()
        z['denominator']=z.groupby(fields,observed=True).n.transform('sum');z['fraction']=z.n/z.denominator
        outputs.append(save(z,table/f'{prefix}_{suffix}.csv'))
    # Source composition and the actual donor/class weights in the P8 training set.
    tr=oof.loc[oof.index[oof.index.isin(cells.index) & ~oof.donor.eq('P8')]].copy()
    tr=tr.reindex(cells.index[cells.index.isin(tr.index)])
    w=pd.Series(0.,index=tr.index)
    for (donor,y),z in tr.groupby(['donor','y'],observed=True):w.loc[z.index]=.5/len(z)
    w*=len(w)/w.sum()
    tr['weight']=w;tr['celltype.l2']=cells.loc[tr.index,'celltype.l2'];tr['celltype.l3']=cells.loc[tr.index,'celltype.l3']
    representation=tr.groupby(['donor','y','celltype.l2','celltype.l3'],observed=True).agg(n=('weight','size'),weight=('weight','sum')).reset_index()
    representation['fraction_of_total_training_weight']=representation.weight/w.sum()
    outputs.append(save(representation,table/f'{prefix}_training_representation.csv'))
    # Matching is repeated independently at each prespecified caliper.
    pairs=[];coverage=[]
    for donor,z in g.groupby('donor',observed=True,sort=True):
        pp=check/f'{prefix}_{donor}_matched_cells.parquet';qp=check/f'{prefix}_{donor}_matching_coverage.csv'
        if valid(donor):pairs.append(pd.read_parquet(pp));coverage.append(pd.read_csv(qp));continue
        cov=np.log1p(z[['nCount_RNA','nFeature_RNA']].astype(float));cov=(cov-cov.mean())/cov.std(ddof=0)
        pairrows=[];audit=[]
        for cal in [.25,.5,1.0]:
            for key,s in z.groupby(['celltype.l3','time','lane'],observed=True,sort=True):
                ex=s[s.excluded].sort_index();ret=s[~s.excluded].sort_index();used=set();matched=set()
                if len(ex) and len(ret):
                    dif=np.abs(cov.loc[ex.index].to_numpy()[:,None,:]-cov.loc[ret.index].to_numpy()[None,:,:])
                    ii,jj=np.where(dif.max(axis=2)<=cal)
                    candidates=sorted((float(np.linalg.norm(dif[i,j])),str(ex.index[i]),str(ret.index[j]),int(i),int(j)) for i,j in zip(ii,jj))
                    for dist,e,r,i,j in candidates:
                        if e in matched or r in used:continue
                        matched.add(e);used.add(r)
                        pairrows.append({'donor':str(donor),'celltype.l3':str(key[0]),'time':key[1],'lane':str(key[2]),'caliper_SD':cal,'excluded_cell_id':e,'retained_cell_id':r,'distance':dist,'excluded_gate_call':ex.loc[e,'gate_call'], 'nCount_RNA_abs_standardized_difference':dif[i,j,0],'nFeature_RNA_abs_standardized_difference':dif[i,j,1]})
                audit.append({'donor':str(donor),'celltype.l3':str(key[0]),'time':key[1],'lane':str(key[2]),'caliper_SD':cal,'n_excluded':len(ex),'n_retained':len(ret),'n_pairs':len(matched),'unmatched_excluded':len(ex)-len(matched)})
        cols=['donor','celltype.l3','time','lane','caliper_SD','excluded_cell_id','retained_cell_id','distance','excluded_gate_call','nCount_RNA_abs_standardized_difference','nFeature_RNA_abs_standardized_difference']
        a=pd.DataFrame(pairrows,columns=cols);b=pd.DataFrame(audit);save(a,pp);save(b,qp);done(str(donor),[pp,qp]);pairs.append(a);coverage.append(b)
    pairs=pd.concat(pairs,ignore_index=True);coverage=pd.concat(coverage,ignore_index=True)
    outputs += [save(pairs,table/f'{prefix}_matched_cells.parquet'),save(coverage,table/f'{prefix}_matching_coverage.csv')]
    print('Loading exact source RNA matrix',flush=True);source=ad.read_h5ad(h5)
    assert source.obs_names.is_unique and set(oof.index)<=set(source.obs_names)
    assert np.isfinite(source.X.data).all() and (source.X.data>=0).all() and np.equal(source.X.data,np.round(source.X.data)).all()
    # Execute only original function definitions, never original notebook execution cells.
    namespace={'np':np,'pd':pd,'sparse':sparse,'TARGET_SUM':1e4,'SEED':20260918}
    sys.path.insert(0,str(project/'src'));from gdt_lr_utils import build_lr
    namespace['build_lr']=build_lr
    wanted={'gate_training_weights','matrix_for_genes','fit_gate','predict_gate'}
    for cell in nbformat.read(oldnb,as_version=4).cells:
        if cell.cell_type!='code':continue
        tree=ast.parse(cell.source)
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in wanted:exec(compile(ast.Module(body=[node],type_ignores=[]),str(oldnb),'exec'),namespace)
    assert wanted<=set(namespace)
    # Original training order is the source AnnData order, not concatenated OOF order.
    train=oof.reindex(source.obs_names[source.obs_names.isin(oof.index) & ~source.obs.donor.eq('P8')])
    test=oof.reindex(source.obs_names[source.obs_names.isin(oof.index) & source.obs.donor.eq('P8')])
    modelp=check/f'{prefix}_reconstructed_model.joblib';verify=table/f'{prefix}_reconstruction.json'
    if valid('reconstructed_model'):model=joblib.load(modelp)
    else:
        print('Reconstructing only the saved P8 outer-fold fit',flush=True)
        model=namespace['fit_gate'](source,train,genes,C)
        p,_=namespace['predict_gate'](model,source,test.index,genes)
        err=float(np.abs(p-test.p_T_lineage_OOF.to_numpy()).max())
        assert err<1e-6,('OOF mismatch: attribution forbidden',err)
        t=Path(str(modelp)+'.tmp');joblib.dump(model,t);os.replace(t,modelp)
        js({'max_abs_probability_error':err,'n_heldout_cells':len(test),'C':C,'genes':len(genes),'seed':20260918,'deployment_model_changed':False},verify);done('reconstructed_model',[modelp,verify])
    # Marker expression normalized over all genes, separate from panel-normalized model input.
    marker_names=['CD3D','CD3E','CD3G','TRDC','TRGC1','TRGC2','TRAC','KLRD1','FCER1G','TYROBP','NKG7','GNLY','PRF1','GZMB']
    ri=source.obs_names.get_indexer(g.index);raw=source.X[ri].tocsr();tot=np.asarray(raw.sum(axis=1)).ravel()
    marker_rows=[]
    for m in marker_names:
        pos=np.flatnonzero(source.var_names==m);marker_rows.append({'gene':m,'source_columns':len(pos),'in_P8_gate_panel':m in genes})
        if len(pos):g[m+'_RNA_log1p_1e4']=np.log1p(np.asarray(raw[:,pos].sum(axis=1)).ravel()*1e4/tot)
    outputs.append(save(pd.DataFrame(marker_rows),table/f'{prefix}_marker_mapping.csv'))
    features=['nCount_RNA','nFeature_RNA','CD3-1_CLR','CD3-2_CLR','TCR-1_CLR','TCR-2_CLR','CD16_CLR','CD56-1_CLR']+[m+'_RNA_log1p_1e4' for m in marker_names if m+'_RNA_log1p_1e4' in g]
    med=g.groupby(['donor','celltype.l3','gate_call'],observed=True)[features].median().reset_index()
    outputs.append(save(med,table/f'{prefix}_protein_RNA_QC.csv'))
    comparisons=[]
    for (donor,subtype,cal),z in pairs.groupby(['donor','celltype.l3','caliper_SD'],observed=True):
        a=g.loc[z.excluded_cell_id];b=g.loc[z.retained_cell_id]
        for f in features:
            dif=a[f].to_numpy()-b[f].to_numpy()
            comparisons.append({'donor':donor,'celltype.l3':subtype,'caliper_SD':cal,'feature':f,'n_pairs':len(z),'median_excluded':a[f].median(),'median_retained':b[f].median(),'median_paired_difference':np.median(dif),'mean_paired_difference':np.mean(dif)})
    comp=pd.DataFrame(comparisons);outputs.append(save(comp,table/f'{prefix}_matched_comparison.csv'))
    p8=g[g.donor.eq('P8')].copy();X,_=namespace['matrix_for_genes'](source,p8.index,genes)
    coef=model.named_steps['lr'].coef_[0];contrib=model.named_steps['scale'].transform(X).multiply(coef).toarray()
    logit=contrib.sum(axis=1)+float(model.named_steps['lr'].intercept_[0])
    assert np.max(np.abs(logit-model.decision_function(X)))<1e-8
    assert np.max(np.abs(model.predict_proba(X)[:,1]-p8.p_T_lineage_OOF))<1e-6
    atr=[]
    for sub,z in p8.groupby('celltype.l3',observed=True):
        ii=p8.index.get_indexer(z.index);a=z.excluded.to_numpy()
        if not a.any() or a.all():continue
        delta=contrib[ii[a]].mean(axis=0)-contrib[ii[~a]].mean(axis=0)
        for j,gene in enumerate(genes):atr.append({'celltype.l3':sub,'gene':gene,'coefficient':coef[j],'mean_contribution_excluded_minus_retained':delta[j],'comparison':'all within subtype','n_excluded':int(a.sum()),'n_retained':int((~a).sum())})
    for sub,z in pairs[pairs.donor.eq('P8') & pairs.caliper_SD.eq(.5)].groupby('celltype.l3',observed=True):
        delta=(contrib[p8.index.get_indexer(z.excluded_cell_id)]-contrib[p8.index.get_indexer(z.retained_cell_id)]).mean(axis=0)
        for j,gene in enumerate(genes):atr.append({'celltype.l3':sub,'gene':gene,'coefficient':coef[j],'mean_contribution_excluded_minus_retained':delta[j],'comparison':'matched 0.5 SD','n_excluded':len(z),'n_retained':len(z)})
    attribution=pd.DataFrame(atr);outputs.append(save(attribution,table/f'{prefix}_feature_contributions.csv'))
    outputs.append(save(g.reset_index(),table/f'{prefix}_gdT_cells.parquet'));done('tables',outputs)
    font=next((f for f in ['Arial','Helvetica','Liberation Sans'] if f in {x.name for x in font_manager.fontManager.ttflist}),None)
    assert font
    plt.rcParams.update({'font.family':font,'font.size':8,'font.weight':'normal','axes.titleweight':'normal','pdf.fonttype':42,'text.color':'black','axes.labelcolor':'black','axes.titlecolor':'black','xtick.color':'black','ytick.color':'black','legend.labelcolor':'black','axes.spines.top':False,'axes.spines.right':False})
    figs=[];previews=[]
    def export(fig,suffix):
        for ext in ['pdf','png']:
            p=figdir/f'{prefix}_{suffix}.{ext}';t=Path(str(p)+'.tmp');fig.savefig(t,format=ext,dpi=240);os.replace(t,p);figs.append(p)
            if ext=='png':previews.append(str(p))
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.3));fig.subplots_adjust(left=.10,right=.87,bottom=.17,top=.78,wspace=.65)
    for ax,sub in zip(axes,['gdT_2','gdT_4']):
        ds=sorted(g.donor.unique())
        for i,d in enumerate(ds):
            z=g[g['celltype.l3'].eq(sub)&g.donor.eq(d)]
            if len(z):ax.scatter((~z.excluded).mean()*100,i,s=25,c='#0072B2',clip_on=False)
            ax.text(102,i,f'{(~z.excluded).sum()}/{len(z)}',va='center',fontsize=7)
        ax.set_yticks(range(len(ds)),ds);ax.set_xlim(0,100);ax.set_xlabel('T-lineage eligible (%)');ax.set_title(sub,pad=14)
    fig.suptitle('GSE164378 | donor-held-out gate retention by source subtype',y=.96,fontsize=10)
    fig.text(.5,.88,'Numbers: retained / evaluated. Source subtypes are not independent receptor truth.',ha='center',fontsize=7)
    export(fig,'subtype_retention')
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.7));fig.subplots_adjust(left=.10,right=.97,bottom=.18,top=.81,wspace=.45)
    for ax,sub in zip(axes,['gdT_2','gdT_4']):
        z=attribution[(attribution['celltype.l3']==sub)&attribution.comparison.eq('matched 0.5 SD')].sort_values('mean_contribution_excluded_minus_retained')
        z=pd.concat([z.head(6),z.tail(4)]).sort_values('mean_contribution_excluded_minus_retained')
        ax.barh(z.gene,z.mean_contribution_excluded_minus_retained,color=['#D55E00' if v<0 else '#0072B2' for v in z.mean_contribution_excluded_minus_retained],edgecolor='none',linewidth=0)
        ax.axvline(0,c='black',lw=.5);ax.set_title(sub+f' (pairs={int(z.n_excluded.iloc[0]) if len(z) else 0})');ax.set_xlabel('Difference in logit contribution',fontsize=7)
    fig.suptitle('P8 | model contributions after subtype / time / lane / depth matching',fontsize=9,y=.96)
    fig.text(.5,.06,'Excluded minus retained: negative values lower P(T). Model explanation, not causal biology.',ha='center',fontsize=7)
    export(fig,'feature_contributions');done('figures',figs)
    p8match=pairs[pairs.donor.eq('P8') & pairs.caliper_SD.eq(.5)]
    focal_comp=comp[comp.donor.eq('P8') & comp.caliper_SD.eq(.5)]
    findings=[]
    for sub in ['gdT_2','gdT_4']:
        z=p8[p8['celltype.l3'].eq(sub)];a=focal_comp[focal_comp['celltype.l3'].eq(sub)].set_index('feature')
        training_n=int(tr['celltype.l3'].eq(sub).sum())
        findings.append(f'{sub}: P8 retains {int((~z.excluded).sum())}/{len(z)} ({(~z.excluded).mean():.2%}); the other seven donors supply {training_n} training cells. At 0.5 SD, matched excluded-minus-retained mean ADT CLR differences are TCR-1 {a.loc["TCR-1_CLR","mean_paired_difference"]:+.3f} and CD16 {a.loc["CD16_CLR","mean_paired_difference"]:+.3f}.')
    txt=(f'### P8 diagnostic audit\n\n'
         f'P8 retains {int((~p8.excluded).sum())}/{len(p8)} source gdT. Loss occurs across time points and lanes; inspect exported strata. '
         f'Exact subtype/time/lane plus depth matching retains {len(p8match)} pairs out of {int(p8.excluded.sum())} excluded P8 cells at the fixed 0.5 SD caliper. '
         'Unmatched cells remain unresolved by this comparison. The 0.25 and 1.0 SD sensitivity results are exported.\n\n'
         + '\n\n'.join(findings) + '\n\n'
         + f'The reconstructed P8 TRDC coefficient is {coef[genes.index("TRDC")]:+.6f}. This conditional coefficient is not a marker-identity claim: correlated predictors, source class composition and regularization can affect its sign. In gdT_4, higher model-input TRDC contributes negatively to the excluded-minus-retained score difference. No single gene explains the full gate decision.\n\n'
         +
         f'The P8 outer-fold reconstruction reproduces saved OOF probabilities (maximum error {json.loads(verify.read_text())["max_abs_probability_error"]:.3g}). '
         'Feature contributions sum to the original logit and explain this fitted model only. Do not confuse them with biological identity markers or causal drivers.\n\n'
         'Donor/class balancing equalizes T versus NK, not gdT subtypes within T. Source gdT_2 and gdT_4 have unequal donor representation; their exact training weights are exported. '
         'Protein and RNA contrasts are descriptive; no cell-level significance test treats cells as independent donors. '
         'Source subtypes, protein-informed annotations and OOF-selected thresholds limit independence. No PANGEA or 15-organ relabeling follows from this source audit.\n\n'
         'Without new wet-lab work, prioritize existing same-cell productive TRG+TRD evidence with verified library/barcode linkage and mixed-droplet review. '
         'Existing suitable FASTQ/BAM may support receptor reconstruction (https://github.com/liulab-dfci/TRUST4); a gene-count matrix alone cannot recover CDR3 sequences. '
         'No detected receptor is missing evidence, not proof of NK identity. Expression/UMAP concordance alone supports candidates, not definitive relabeling.\n\n'
         'Resume on Woody: `cd /data/users/kaelynwen/gdT_project/gdt_lr_12model && /data/conda_envs/kaelynwen/ENTER/bin/python -c "from pathlib import Path; import runpy; runpy.run_path(\'build_21_gate_LRM05_evaluation_15organs.py\')[\'run_gate_discordance_audit_P8\'](Path.cwd())"`.')
    report=table/f'{prefix}_interpretation.md';t=Path(str(report)+'.tmp');t.write_text(txt);os.replace(t,report);done('interpretation',[report])
    manifest['complete']=True;manifest['completed_utc']=datetime.now(timezone.utc).isoformat();js(manifest,mp)
    paths=[project/p for u in manifest['units'].values() for p in u['outputs']]+[mp]
    records=[{'local_path':'/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/gdt_lr_12model/'+str(p.relative_to(project)),'Woody_path':'/data/users/kaelynwen/gdT_project/gdt_lr_12model/'+str(p.relative_to(project)),'sha256':sha(p)} for p in paths]
    save(pd.DataFrame(records).drop_duplicates('local_path'),table/f'{prefix}_output_manifest.csv')
    return {'interpretation':txt,'previews':previews}


ATLAS_AUDIT = r'''
# %% markdown
## Exploratory atlas reannotation and GSE168163 receptor linkage

The author-approved rule is fixed before assignment: retain the existing
`source_annotation_NK` scope (PANGEA `NK_*`; 15-organ's existing T/NK-ambiguous
source annotations), require `optimized_M05_positive` at its unchanged threshold,
and exclude matched paired alpha-beta or mixed-receptor conflicts from the
`putative γδ T cells` label. All three gate states remain eligible. This is an
additional exploratory label, never a replacement for evaluation truth.

GSE168163 linkage uses official ENA run metadata to connect exact original GEX
filenames and VDJ filenames within the same experiment/GSM, followed by an exact
barcode join. Multiple sequencing runs can represent the same library/barcode;
row counts and unique GSM/barcode counts are both reported. Productive,
full-length, high-confidence cell contigs with V/J/CDR3 are required for the
new strict paired-receptor calls. Incomplete TRG/TRD fragments are not positive
identity evidence. Existing receptor evidence elsewhere is retained with its
original quality limitations. 15-organ supplies alpha-beta evidence only.

Each downloaded run record is an atomic, hash-validated checkpoint. Input hashes,
the function source and the explicit rule define the run signature. Original
predictions, UMAPs, annotations and model files are not modified.

# %% code
import runpy
from pathlib import Path
from IPython.display import display, Markdown
PROJECT=next(p for p in [Path.cwd(),Path.cwd().parent] if (p/'build_21_gate_LRM05_evaluation_15organs.py').exists())
atlas_audit=runpy.run_path(str(PROJECT/'build_21_gate_LRM05_evaluation_15organs.py'))['run_gate_discordance_audit_atlas'](PROJECT)
display(Markdown(atlas_audit['interpretation']))
'''


def run_gate_discordance_audit_atlas(project):
    import hashlib, inspect, json, os, re, sys, platform, io
    import xml.etree.ElementTree as ET
    from datetime import datetime, timezone
    import pandas as pd
    import numpy as np
    import requests
    project=Path(project).resolve();prefix='gate_discordance_audit'
    root=project/'results/tables'/NAME;check=project/'checkpoints'/NAME;pred=project/'predictions'/NAME
    for p in [root,check,pred]:p.mkdir(parents=True,exist_ok=True)
    source=project/'predictions/22_LR_hierarchical_CE/gate_discordance_audit_cells.parquet'
    vdjp=project/'checkpoints/22_LR_hierarchical_CE/GSE168163_VDJ_Tcell.csv.gz'
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def write_text(s,p):
        t=Path(str(p)+'.tmp');t.write_text(s);os.replace(t,p);return p
    def js(x,p):return write_text(json.dumps(x,indent=2,sort_keys=True),p)
    def save(d,p):
        t=Path(str(p)+'.tmp')
        if p.suffix=='.parquet':d.to_parquet(t,index=False)
        else:d.to_csv(t,index=False)
        os.replace(t,p);return p
    contract={'input_sha256':{str(p.relative_to(project)):sha(p) for p in [source,vdjp]},
              'function_sha256':hashlib.sha256(inspect.getsource(run_gate_discordance_audit_atlas).encode()).hexdigest(),
              'rule':'existing source_annotation_NK AND optimized_M05_positive AND NOT paired_ab_or_mixed_conflict; gate independent',
              'linkage':'ENA exact GEX filename and VDJ filename in same experiment; exact barcode',
              'randomness':'none','schema_version':1}
    sig=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    mp=check/f'{prefix}_atlas_manifest.json'
    manifest=json.loads(mp.read_text()) if mp.exists() else {'signature':sig,'contract':contract,'units':{},'complete':False}
    if manifest['signature']!=sig:raise RuntimeError('Atlas audit signature changed; inspect prior run before reuse')
    def valid(unit):
        u=manifest['units'].get(unit,{})
        return bool(u.get('outputs')) and all((project/k).exists() and sha(project/k)==v for k,v in u['outputs'].items())
    def done(unit,paths):
        manifest['units'][unit]={'outputs':{str(p.relative_to(project)):sha(p) for p in paths},'utc':datetime.now(timezone.utc).isoformat()};js(manifest,mp)
    def fetch(url,p,unit):
        if not valid(unit):
            r=requests.get(url,timeout=60);r.raise_for_status();write_text(r.text,p);done(unit,[p])
        return p.read_text()
    base='https://www.ebi.ac.uk/ena/'
    url=base+'portal/api/filereport?accession=PRJNA706382&result=read_run&fields=run_accession,experiment_accession,sample_accession,secondary_sample_accession,experiment_title,library_name,library_strategy,library_source&format=tsv'
    report=fetch(url,check/f'GSE168163_{prefix}_ENA.tsv','ENA_report');runs=pd.read_csv(io.StringIO(report),sep='\t')
    d=pd.read_parquet(source);assert not d.duplicated(['dataset','cell_id']).any()
    z=d[d.source_study.eq('GSE168163')].copy();rna=z[['Sample','SRR','Sample Name']].drop_duplicates()
    assert rna.SRR.is_unique
    exps=runs.set_index('run_accession').loc[rna.SRR,'experiment_accession'].unique()
    rr=runs[runs.experiment_accession.isin(exps)].copy();meta=[]
    for run in rr.run_accession:
        print('Verifying ENA',run,flush=True)
        xml=fetch(base+'browser/api/xml/'+run,check/f'GSE168163_{prefix}_{run}.xml',run)
        tree=ET.fromstring(xml);node=tree.find('.//RUN');ref=node.find('EXPERIMENT_REF')
        opt=' '.join(x.findtext('VALUE','') for x in node.findall('./RUN_ATTRIBUTES/RUN_ATTRIBUTE') if x.findtext('TAG')=='options')
        names=re.findall(r'(?:--read[123]PairFiles=)([^\s]+)',opt)
        assert len(names)==3,(run,opt)
        stem=re.sub(r'_[IR][123]\.fastq\.gz$','',names[0]);assert all(re.sub(r'_[IR][123]\.fastq\.gz$','',f)==stem for f in names)
        meta.append({'SRR':run,'experiment':ref.attrib['accession'],'GSM':ref.attrib['refname'],'original_stem':stem,'official_url':base+'browser/api/xml/'+run})
    meta=pd.DataFrame(meta);cross=[]
    for row in rna.to_dict('records'):
        a=meta[meta.SRR.eq(row['SRR'])];assert len(a)==1;a=a.iloc[0]
        assert a.original_stem==row['Sample'] and a.GSM==row['Sample Name']
        core=re.search(r'Tcell_(\d+)_bt1$',a.original_stem);assert core
        vdj=meta[meta.experiment.eq(a.experiment)&meta.original_stem.eq('VDJ_Tcell_'+core[1]+'_bt1')]
        assert len(vdj)==1 and vdj.iloc[0].GSM==a.GSM
        cross.append({**row,'experiment':a.experiment,'VDJ_SRR':vdj.iloc[0].SRR,'VDJ_original_stem':vdj.iloc[0].original_stem,'sampleID':'TCell_'+core[1]})
    cross=pd.DataFrame(cross);z=z.merge(cross,on=['Sample','SRR','Sample Name'],validate='many_to_one')
    z['barcode']=z.cell_id.str.split('@').str[-1];assert z.barcode.str.fullmatch('[ACGT]{16}-1').all()
    z['GSM_barcode']=z['Sample Name']+'@'+z.barcode
    v=pd.read_csv(vdjp,low_memory=False);assert not v.duplicated(['sampleID','contig_id']).any()
    def flag(s):return s.astype(str).str.lower().eq('true')
    ok=flag(v.is_cell)&flag(v.high_confidence)&flag(v.full_length)&flag(v.productive)
    for col in ['v_gene','j_gene','cdr3','cdr3_nt']:ok &= v[col].notna()&~v[col].astype(str).isin(['None','nan',''])
    v['strict']=ok
    n=v.groupby(['sampleID','barcode','chain']).size().unstack(fill_value=0).add_prefix('raw_')
    strict=v[ok].groupby(['sampleID','barcode','chain']).size().unstack(fill_value=0).add_prefix('strict_')
    evidence=n.join(strict,how='left').fillna(0).reset_index()
    for c in ['TRA','TRB','TRG','TRD']:
        for pfx in ['raw_','strict_']:
            if pfx+c not in evidence:evidence[pfx+c]=0
    evidence['matched_receptor_record']=True
    z=z.merge(evidence,on=['sampleID','barcode'],how='left',validate='many_to_one')
    cols=[c for c in evidence if c.startswith(('raw_','strict_'))]
    z[cols]=z[cols].fillna(0).astype(int);z['matched_receptor_record']=z.matched_receptor_record.fillna(False).astype(bool)
    z['paired_ab_quality_supported']=(z.strict_TRA>=1)&(z.strict_TRB>=1)
    z['paired_gd_quality_supported']=(z.strict_TRG>=1)&(z.strict_TRD>=1)
    z['mixed_receptor_conflict']=((z.strict_TRA+z.strict_TRB)>0)&((z.strict_TRG+z.strict_TRD)>0)
    z['same_GSM_barcode_rows']=z.groupby('GSM_barcode').cell_id.transform('size')
    outputs=[save(meta,root/f'GSE168163_{prefix}_ENA_runs.csv'),save(cross,root/f'GSE168163_{prefix}_library_linkage.csv'),save(z,pred/f'GSE168163_{prefix}_receptor_cells.parquet')]
    quality=v.groupby(['chain','is_cell','high_confidence','full_length','productive'],dropna=False).size().reset_index(name='contigs')
    outputs.append(save(quality,root/f'GSE168163_{prefix}_receptor_quality.csv'))
    # Preserve all atlas columns and add a transparent exploratory overlay.
    out=d.copy();out['gate_discordance_audit_receptor_conflict']=False
    for c in ['strict_paired_abTCR_reconstructed','strict_paired_abTCR','mixed_ab_gd']:
        out['gate_discordance_audit_receptor_conflict'] |= out[c].fillna(False).astype(bool)
    out['gate_discordance_audit_GSE168163_linked']=False
    out['gate_discordance_audit_GSE168163_paired_ab']=False
    out['gate_discordance_audit_GSE168163_paired_gd']=False
    out['gate_discordance_audit_GSE168163_GSM_barcode']=''
    zz=z.set_index('cell_id');sel=out.source_study.eq('GSE168163');ids=out.loc[sel,'cell_id']
    for dest,src in [('gate_discordance_audit_GSE168163_paired_ab','paired_ab_quality_supported'),('gate_discordance_audit_GSE168163_paired_gd','paired_gd_quality_supported'),('gate_discordance_audit_GSE168163_GSM_barcode','GSM_barcode')]:out.loc[sel,dest]=ids.map(zz[src]).to_numpy()
    out.loc[sel,'gate_discordance_audit_GSE168163_linked']=ids.map(zz.matched_receptor_record).to_numpy()
    out.loc[sel,'gate_discordance_audit_receptor_conflict'] |= ids.map(zz.paired_ab_quality_supported|zz.mixed_receptor_conflict).to_numpy()
    candidate=out.source_annotation_NK.astype(bool)&out.optimized_M05_positive.astype(bool)
    out['gate_discordance_audit_putative_gdT']=candidate&~out.gate_discordance_audit_receptor_conflict
    out['gate_discordance_audit_receptor_conflict_review']=candidate&out.gate_discordance_audit_receptor_conflict
    out['gate_discordance_audit_label']=pd.Series(pd.NA,index=out.index,dtype='string')
    out.loc[out.gate_discordance_audit_putative_gdT,'gate_discordance_audit_label']='putative γδ T cells'
    # A missing overlay label means no reassignment, not a negative prediction.
    assert not (out.gate_discordance_audit_putative_gdT&out.gate_discordance_audit_receptor_conflict).any()
    assert candidate.sum()==out.gate_discordance_audit_putative_gdT.sum()+out.gate_discordance_audit_receptor_conflict_review.sum()
    pd.testing.assert_frame_equal(out[d.columns],d)
    outputs += [save(out,pred/f'{prefix}_putative_gdT_cells.parquet'),save(out[out.gate_discordance_audit_receptor_conflict_review],root/f'{prefix}_putative_gdT_receptor_conflicts.csv')]
    summaries=[]
    for dataset,s in out.groupby('dataset',observed=True):
        for group,mask in [('putative γδ T cells',s.gate_discordance_audit_putative_gdT),('receptor conflict review',s.gate_discordance_audit_receptor_conflict_review)]:
            for call in ['gate_T_eligible','gate_NK_call','gate_abstain']:
                summaries.append({'dataset':dataset,'group':group,'gate_state':call,'n_atlas_rows':int((mask&s[call]).sum())})
    summary=pd.DataFrame(summaries);outputs.append(save(summary,root/f'{prefix}_putative_gdT_summary.csv'))
    bysource=out[candidate].groupby(['dataset','source_study','source_organ','source_annotation','gate_discordance_audit_putative_gdT','gate_discordance_audit_receptor_conflict_review','gate_T_eligible','gate_NK_call','gate_abstain'],dropna=False,observed=True).size().reset_index(name='n_atlas_rows')
    outputs.append(save(bysource,root/f'{prefix}_putative_gdT_by_source.csv'))
    unique=z.drop_duplicates('GSM_barcode');link_summary={'n_atlas_rows':len(z),'n_unique_GSM_barcode':len(unique),'matched_rows':int(z.matched_receptor_record.sum()),'matched_unique':int(unique.matched_receptor_record.sum()),'paired_ab_rows':int(z.paired_ab_quality_supported.sum()),'paired_ab_unique':int(unique.paired_ab_quality_supported.sum()),'paired_gd_rows':int(z.paired_gd_quality_supported.sum()),'paired_gd_unique':int(unique.paired_gd_quality_supported.sum()),'raw_TRG_TRD_contigs':int(v.chain.isin(['TRG','TRD']).sum()),'productive_TRG_TRD_contigs':int((v.chain.isin(['TRG','TRD'])&flag(v.productive)).sum()),'unique_library_counts':len(cross['Sample Name'].unique()),'RNA_run_counts':len(cross)}
    outputs.append(js(link_summary,root/f'GSE168163_{prefix}_linkage_summary.json'))
    env={'python':sys.version,'platform':platform.platform(),'pandas':pd.__version__,'numpy':np.__version__,'requests':requests.__version__}
    outputs.append(js(env,root/f'{prefix}_atlas_environment.json'))
    interpretation=('## Exploratory atlas reannotation\n\n'+('```text\n'+summary.to_string(index=False)+'\n```')+'\n\n'
        +f'GSE168163: {len(z)} atlas rows represent {len(unique)} unique GSM/barcode units across {link_summary["unique_library_counts"]} libraries and {len(cross)} GEX runs. '
        +f'{link_summary["matched_rows"]} rows ({link_summary["matched_unique"]} unique units) match processed receptor records. '
        +f'Quality-filtered paired alpha-beta evidence occurs in {link_summary["paired_ab_rows"]} rows ({link_summary["paired_ab_unique"]} unique units); paired gamma-delta evidence in {link_summary["paired_gd_rows"]}. '
        +'The TRG/TRD records contain no productive contig. Their incomplete fragments are not counted as gamma-delta identity evidence, and their absence is not NK evidence.\n\n'
        +'The approved overlay is based on existing source_annotation_NK, unchanged optimized_M05_positive and no matched paired-alpha-beta/mixed-receptor conflict. '
        +'Every original column is retained unchanged. Gate calls do not determine inclusion. Candidate counts are atlas rows, not confirmed cell abundance; GSE168163 run duplicates are separately audited. '
        +'15-organ has alpha-beta evidence only; absence of those records supplies no positive gamma-delta evidence. Other PANGEA receptor records retain their original quality limitations. '
        +'These hypotheses must not be reused as reference truth to claim accuracy gains. Missing single-chain or mixed-droplet quality remains unresolved.\n\n'
        +'Resume on Woody: `cd /data/users/kaelynwen/gdT_project/gdt_lr_12model && /data/conda_envs/kaelynwen/ENTER/bin/python -c "from pathlib import Path; import runpy; runpy.run_path(\'build_21_gate_LRM05_evaluation_15organs.py\')[\'run_gate_discordance_audit_atlas\'](Path.cwd())"`.')
    outputs.append(write_text(interpretation,root/f'{prefix}_atlas_interpretation.md'));done('atlas_tables',outputs)
    manifest['complete']=True;js(manifest,mp)
    paths=[project/p for u in manifest['units'].values() for p in u['outputs']]+[mp]
    records=[{'local_path':'/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/gdt_lr_12model/'+str(p.relative_to(project)),'Woody_path':'/data/users/kaelynwen/gdT_project/gdt_lr_12model/'+str(p.relative_to(project)),'sha256':sha(p)} for p in paths]
    save(pd.DataFrame(records),root/f'{prefix}_atlas_output_manifest.csv')
    print(interpretation,flush=True)
    return {'interpretation':interpretation}


THRESHOLD_AUDIT = r'''
# %% markdown
## Lower-threshold audit: which cells are added?

This descriptive audit uses the unchanged **optimized_M05** probabilities.
It compares the frozen threshold with 0.5 and 0.2, without choosing a new deployed
threshold, retraining, changing annotations or applying the root as a hard filter.
The disjoint score intervals identify the additional records exactly. These
cutoffs were discussed after SPL inspection and are exploratory, not independent
validation. Original putative labels remain tied to their original rule.

SPL reported paired receptor groups supply the observed recovery/error counts.
PANGEA and 15-organ are counted as atlas records, not confirmed identities. The
existing GSE168163 quality-filtered linkage is incorporated separately from the
older receptor annotation. GSE149356 paired TRG/TRD records lack complete quality
flags; 15-organ has no gamma-delta receptor-positive reference. Missing receptors
are not negative evidence. Multiple PANGEA runs may contain the same biological
cell; known GSE168163 duplicates are flagged rather than silently removed.

All saved input hashes and the function source determine the audit signature.
Each dataset x source-study x source-sample unit is an atomic checkpoint (SPL:
each donor). Checkpoint checksums, row identities and schemas are validated before
reuse. Aggregation requires all expected units. No random sampling is used.
Tables are the primary outputs for exact-count lookup; existing UMAPs are unchanged.
QC/marker contrasts are descriptive within source unit and annotation/receptor
group, with at least five records per interval and baseline. Marker overlap with
model inputs is retained in the existing feature-membership audits; these are not
independent validation tests. No cell-level P values are computed.

# %% code
import runpy
from pathlib import Path
from IPython.display import display, Markdown
PROJECT=next(p for p in [Path.cwd(),Path.cwd().parent] if (p/'build_21_gate_LRM05_evaluation_15organs.py').exists())
threshold_audit=runpy.run_path(str(PROJECT/'build_21_gate_LRM05_evaluation_15organs.py'))['run_gate_discordance_audit_thresholds'](PROJECT)
display(Markdown(threshold_audit['interpretation']))
'''


def run_gate_discordance_audit_thresholds(project):
    import pandas as pd
    import numpy as np
    import hashlib, json, inspect, os, sys, platform
    from datetime import datetime, timezone
    project=Path(project).resolve();prefix='gate_discordance_audit_threshold'
    pred=project/'predictions'/NAME;table=project/'results/tables'/NAME;check=project/'checkpoints'/NAME
    inputs=[pred/'SPL_LR_hierarchical_predictions.parquet',pred/'SPL_gate_discordance_audit_receptor_cells.parquet',pred/'gate_discordance_audit_putative_gdT_cells.parquet',table/'parameters.json']
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def js(v,p):
        t=Path(str(p)+'.tmp');t.write_text(json.dumps(v,indent=2,sort_keys=True));os.replace(t,p);return p
    def save(v,p):
        t=Path(str(p)+'.tmp')
        if p.suffix=='.parquet':v.to_parquet(t,index=False)
        else:v.to_csv(t,index=False)
        os.replace(t,p);return p
    params=json.loads(inputs[-1].read_text());cut=params['thresholds']['optimized_M05']
    contract={'inputs':{str(p.relative_to(project)):sha(p) for p in inputs},'function':hashlib.sha256(inspect.getsource(run_gate_discordance_audit_thresholds).encode()).hexdigest(),'thresholds':[cut,.5,.2],'pandas':pd.__version__,'schema':1,'no_refit':True,'gate_is_filter':False}
    sig=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest();mp=check/f'{prefix}_manifest.json'
    manifest=json.loads(mp.read_text()) if mp.exists() else {'signature':sig,'contract':contract,'units':{},'complete':False}
    assert manifest['signature']==sig,'Changed inputs/configuration: do not reuse old checkpoints'
    s=pd.read_parquet(inputs[0]);s=s[s.receptor_evidence_group.isin(['paired_gdT','paired_abT'])].copy()
    q=pd.read_parquet(inputs[1]);assert q.cell_id.is_unique and len(q)==378
    extra=[c for c in q if c not in s]+['cell_id'];s=s.merge(q[extra],on='cell_id',how='left',validate='one_to_one')
    s['dataset']='SPL';s['source_study']='SPL';s['source_sample']=s.donor.astype(str);s['source_organ']='SPL'
    s['source_annotation']=s.receptor_evidence_group.astype(str)
    a=pd.read_parquet(inputs[2]);data=pd.concat([s,a],ignore_index=True,sort=False)
    assert not data.duplicated(['dataset','cell_id']).any()
    assert np.isfinite(data.p_gdT_given_T).all() and data.p_gdT_given_T.between(0,1).all()
    data['threshold_lower']=np.select([data.p_gdT_given_T.ge(cut),data.p_gdT_given_T.ge(.5),data.p_gdT_given_T.ge(.2)],[cut,.5,.2],default=0.)
    data['threshold_upper']=np.select([data.p_gdT_given_T.ge(cut),data.p_gdT_given_T.ge(.5),data.p_gdT_given_T.ge(.2)],[np.inf,cut,.5],default=.2)
    data['gate_T_eligible']=data.p_T.ge(params['threshold_T']);data['gate_NK_call']=data.p_T.le(params['threshold_NK']);data['gate_abstain']=~(data.gate_T_eligible|data.gate_NK_call)
    def truth(col):return data[col].eq(True).fillna(False) if col in data else pd.Series(False,index=data.index)
    data['threshold_audit_gd_receptor_support']=data.receptor_evidence_group.eq('paired_gdT').fillna(False)|truth('TRD_TRG_with_CDR3')|truth('gate_discordance_audit_GSE168163_paired_gd')
    data['threshold_audit_ab_receptor_support']=data.receptor_evidence_group.eq('paired_abT').fillna(False)|truth('strict_paired_abTCR')|truth('strict_paired_abTCR_reconstructed')|truth('gate_discordance_audit_GSE168163_paired_ab')
    data['threshold_audit_mixed_conflict']=truth('mixed_ab_gd')|(data.threshold_audit_gd_receptor_support&data.threshold_audit_ab_receptor_support)
    data['threshold_audit_receptor_file_linked']=data.dataset.eq('SPL')|truth('receptor_file_matched')|truth('gate_discordance_audit_GSE168163_linked')
    data['threshold_audit_no_paired_support']=~(data.threshold_audit_gd_receptor_support|data.threshold_audit_ab_receptor_support)
    data['threshold_audit_flagged_doublet']=truth('predicted_doublet')|truth('reported_is_doublet')
    data['threshold_audit_doublet_assessed']=truth('doublet_evidence_available')|data.reported_is_doublet.notna()
    data['threshold_audit_low_child_coverage']=data.child_feature_coverage.lt(.95)
    data['threshold_audit_low_root_coverage']=data.root_feature_coverage.lt(.95)
    keys=['dataset','source_study','source_sample'];units=[]
    for key,g in data.groupby(keys,dropna=False,observed=True,sort=True):
        name=json.dumps([str(x) for x in key]);uid=hashlib.sha256(name.encode()).hexdigest()[:20];path=check/f'{prefix}_{uid}.parquet'
        u=manifest['units'].get(name);valid=bool(u) and path.exists() and sha(path)==u['sha256']
        if valid:
            v=pd.read_parquet(path);assert list(v.columns)==list(data.columns) and v.cell_id.tolist()==g.cell_id.tolist()
        else:
            save(g,path);manifest['units'][name]={'path':str(path.relative_to(project)),'sha256':sha(path),'n':len(g),'utc':datetime.now(timezone.utc).isoformat()};js(manifest,mp)
        units.append(name)
    assert set(units)==set(manifest['units'])
    d=pd.concat([pd.read_parquet(project/manifest['units'][u]['path']) for u in units],ignore_index=True)
    assert len(d)==len(data) and not d.duplicated(['dataset','cell_id']).any()
    flags=['threshold_audit_gd_receptor_support','threshold_audit_ab_receptor_support','threshold_audit_mixed_conflict','threshold_audit_no_paired_support','threshold_audit_receptor_file_linked','source_annotation_NK','source_annotation_gdT','gate_T_eligible','gate_NK_call','gate_abstain','threshold_audit_flagged_doublet','threshold_audit_doublet_assessed','threshold_audit_low_child_coverage','threshold_audit_low_root_coverage']
    for c in flags:d[c]=d[c].eq(True).fillna(False)
    def counts(fields):
        agg={c:(c,'sum') for c in flags};agg['n_atlas_rows']=('cell_id','size')
        return d.groupby(fields,observed=True,dropna=False).agg(**agg).reset_index()
    band=counts(['dataset','threshold_lower','threshold_upper'])
    byunit=counts(keys+['source_organ','threshold_lower','threshold_upper'])
    byannot=counts(keys+['source_organ','source_annotation','threshold_lower','threshold_upper'])
    # Report cumulative metrics separately from interval counts.
    metrics=[]
    for dataset,z in d.groupby('dataset',observed=True):
        strata=[('all',z)]
        if dataset=='SPL':strata += [(str(k),v) for k,v in z.groupby('source_sample',observed=True)]
        for unit,v in strata:
            for t in [cut,.5,.2]:
                selected=v.p_gdT_given_T.ge(t);g=v.threshold_audit_gd_receptor_support;ab=v.threshold_audit_ab_receptor_support
                metrics.append({'dataset':dataset,'unit':unit,'threshold':t,'n_records':len(v),'n_selected':int(selected.sum()),'gd_support_denominator':int(g.sum()),'gd_support_selected':int((g&selected).sum()),'ab_support_denominator':int(ab.sum()),'ab_support_selected':int((ab&selected).sum()),'gd_support_retention':float((g&selected).sum()/g.sum()) if g.sum() else None,'gate_T_eligible_selected':int((selected&v.gate_T_eligible).sum())})
    metrics=pd.DataFrame(metrics)
    qc=['audit_total_counts','audit_n_genes_by_counts','audit_pct_counts_mt','root_feature_coverage','child_feature_coverage']+[c for c in d if c.startswith('log1p_')]
    strata=keys+['source_annotation','threshold_lower','threshold_upper']
    summaries=[];contrasts=[]
    for key,z in d.groupby(strata,observed=True,dropna=False,sort=True):
        for f in qc:
            a=pd.to_numeric(z[f],errors='coerce').dropna()
            if len(a):summaries.append(dict(zip(strata,key))|{'feature':f,'n_measured':len(a),'median':float(a.median()),'mean':float(a.mean())})
    stats=pd.DataFrame(summaries);base=stats[stats.threshold_lower.eq(cut)]
    join=keys+['source_annotation','feature']
    for lower in [.5,.2]:
        new=stats[stats.threshold_lower.eq(lower)]
        c=new.merge(base[join+['n_measured','median','mean']],on=join,suffixes=('_new','_baseline'),validate='one_to_one')
        c=c[c.n_measured_new.ge(5)&c.n_measured_baseline.ge(5)].copy()
        c['mean_difference']=c.mean_new-c.mean_baseline;c['median_difference']=c.median_new-c.median_baseline
        contrasts.append(c)
    contrasts=pd.concat(contrasts,ignore_index=True)
    unique=d[d.dataset.eq('PANGEA')&d.source_study.eq('GSE168163')].copy()
    unique_summary=unique.groupby(['threshold_lower','threshold_upper'],observed=True).agg(n_atlas_rows=('cell_id','size'),n_GSM_barcode=('gate_discordance_audit_GSE168163_GSM_barcode','nunique')).reset_index()
    outputs=[save(d,pred/f'{prefix}_cells.parquet'),save(band,table/f'{prefix}_bands.csv'),save(byunit,table/f'{prefix}_by_source.csv'),save(byannot,table/f'{prefix}_by_annotation.csv'),save(metrics,table/f'{prefix}_cumulative.csv'),save(stats,table/f'{prefix}_QC_markers.csv'),save(contrasts,table/f'{prefix}_within_source_contrasts.csv'),save(unique_summary,table/f'{prefix}_GSE168163_duplicate_audit.csv')]
    # Original columns are never overwritten in the input files; verify all input hashes again.
    assert all(sha(project/p)==h for p,h in contract['inputs'].items())
    env={'python':sys.version,'platform':platform.platform(),'pandas':pd.__version__,'numpy':np.__version__}
    outputs.append(js(env,table/f'{prefix}_environment.json'))
    interpretation=('## Lower-threshold audit results\n\nCumulative counts (original labels and model unchanged):\n\n```text\n'+metrics[metrics.unit.eq('all')].to_string(index=False)+'\n```\n\nDisjoint intervals; upper bounds are exclusive:\n\n```text\n'+band[['dataset','threshold_lower','threshold_upper','n_atlas_rows','threshold_audit_gd_receptor_support','threshold_audit_ab_receptor_support','threshold_audit_no_paired_support','gate_T_eligible','gate_NK_call','gate_abstain']].to_string(index=False)+'\n```\n\n'
        +'The exported cell table preserves exact identities, original labels, scores and existing receptor evidence. New threshold fields are diagnostic only; the existing putative overlay is not expanded. '
        +'Known GSE168163 run duplicates are reported separately. Do not add unique counts across score intervals: the same GSM/barcode may appear with different run scores. '
        +'Only SPL paired-group counts directly quantify recovery versus the chosen receptor reference; PANGEA coverage is incomplete and 15-organ has alpha-beta evidence only. '
        +'Absence of paired support is unresolved, not proof of NK. SPL and processed GSE149356 receptor quality limitations remain. '
        +'QC/marker summaries are per source unit and annotation; comparisons with at least five records per interval are descriptive, without P values or causal interpretation. '
        +'RNA evidence overlaps classifier features; expression and UMAP cannot independently validate the proposed identity. No new marker identities or model coefficients were inferred. '
        +'No deployment threshold is selected from these previously inspected evaluation data.\n\n'
        +'Resume on Woody: `cd /data/users/kaelynwen/gdT_project/gdt_lr_12model && /data/conda_envs/kaelynwen/ENTER/bin/python -c "from pathlib import Path; import runpy; runpy.run_path(\'build_21_gate_LRM05_evaluation_15organs.py\')[\'run_gate_discordance_audit_thresholds\'](Path.cwd())"`.')
    p=table/f'{prefix}_interpretation.md';t=Path(str(p)+'.tmp');t.write_text(interpretation);os.replace(t,p);outputs.append(p)
    manifest['complete']=True;manifest['outputs']={str(p.relative_to(project)):sha(p) for p in outputs};manifest['completed_utc']=datetime.now(timezone.utc).isoformat();js(manifest,mp)
    paths=outputs+[mp]+[project/manifest['units'][u]['path'] for u in units]
    rows=[{'local_path':'/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/gdt_lr_12model/'+str(p.relative_to(project)),'Woody_path':'/data/users/kaelynwen/gdT_project/gdt_lr_12model/'+str(p.relative_to(project)),'sha256':sha(p)} for p in paths]
    save(pd.DataFrame(rows),table/f'{prefix}_output_manifest.csv')
    print('Complete:',len(units),'source units;',len(d),'records',flush=True)
    return {'interpretation':interpretation}


THRESHOLD_UMAP = r'''
# %% markdown
## NK-like candidates added between 0.5 and the frozen threshold

The preceding threshold tables already counted added candidates. Earlier UMAPs
showed frozen-threshold gate discordance. This section adds the missing
**same-coordinate comparison** of the newly added NK-like candidates and the
original candidates, using saved evidence rather than repeating model inference.

Scope is the existing `source_annotation_NK` flag: PANGEA `NK_*` source labels;
15-organ's existing T/NK-ambiguous source labels. These two scopes are not equally
specific and neither is a truth assignment. The new interval is lower-inclusive
0.5 and upper-exclusive frozen optimized_M05 threshold (0.9394744727488467).
All gate states and receptor conflicts remain visible. Existing putative labels
are unchanged. Source gdT annotations supply descriptive expression context only.

Original UMAP coordinates are joined exactly by cell ID, with no refitting.
The full atlas view retains every available coordinate. The additional 15-organ
T/NK view uses the previously established annotation/receptor scope, independent
of predictions. PANGEA is already the original Level2_TandNK scope. Study/organ
facets include every stratum with at least one added candidate; no favorable
strata are selected. Missing coordinates remain in the cell and evidence tables.

RNA values reuse the previous count-normalized log1p marker audit. Marker
expression overlaps model inputs and is not independent confirmation. Receptor
support is retained at its original quality level: processed GSE149356 TRG/TRD
records have CDR3 but incomplete quality flags. 15-organ has alpha-beta evidence
only. Unmarked cells are unresolved, not confirmed NK. Known GSE168163 repeated
GSM/barcode records remain flagged, so atlas records are not independent replicates.

Each dataset table unit and each figure is saved atomically and checksummed.
Restart reuses only units matching input, code, configuration and environment
signatures. PDF figures contain vector points and embedded editable text, with
normal-weight Arial, Helvetica or Liberation Sans and black text. PNGs are previews.
No statistical tests treat cells as independent donors.

# %% code
import runpy
from pathlib import Path
from IPython.display import display, Markdown, Image
PROJECT=next(p for p in [Path.cwd(),Path.cwd().parent] if (p/'build_21_gate_LRM05_evaluation_15organs.py').exists())
threshold_umap=runpy.run_path(str(PROJECT/'build_21_gate_LRM05_evaluation_15organs.py'))['run_gate_discordance_audit_threshold_UMAP'](PROJECT)
display(Markdown(threshold_umap['interpretation']))
for preview in threshold_umap['previews']:
    display(Image(filename=preview))
'''


def run_gate_discordance_audit_threshold_UMAP(project):
    import hashlib, inspect, json, os, platform, sys
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from datetime import datetime, timezone
    project=Path(project).resolve();prefix='gate_discordance_audit_threshold_UMAP'
    pred=project/'predictions'/NAME;table=project/'results/tables'/NAME
    figdir=project/'results/figures'/NAME;check=project/'checkpoints'/NAME
    inputs=[pred/'gate_discordance_audit_threshold_cells.parquet',table/'parameters.json']+[pred/f'{ds}_gate_discordance_audit_UMAP_source_data.parquet' for ds in ['PANGEA','15-organ']]
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def js(v,p):
        tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(v,indent=2,sort_keys=True));os.replace(tmp,p);return p
    def save(d,p):
        tmp=Path(str(p)+'.tmp')
        if p.suffix=='.parquet':d.to_parquet(tmp,index=False)
        else:d.to_csv(tmp,index=False)
        os.replace(tmp,p);return p
    available={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ['Arial','Helvetica','Liberation Sans'] if f in available),None)
    assert font,'Install an approved font before rendering'
    plt.rcParams.update({'font.family':font,'font.size':8,'font.weight':'normal','axes.titleweight':'normal','axes.labelweight':'normal','pdf.fonttype':42,'ps.fonttype':42,'text.color':'black','axes.labelcolor':'black','axes.titlecolor':'black','xtick.color':'black','ytick.color':'black','legend.labelcolor':'black','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6})
    params=json.loads(inputs[1].read_text());cut=params['thresholds']['optimized_M05']
    contract={'inputs':{str(p.relative_to(project)):sha(p) for p in inputs},'code':hashlib.sha256(inspect.getsource(run_gate_discordance_audit_threshold_UMAP).encode()).hexdigest(),'threshold_lower':.5,'threshold_upper':cut,'scope':'existing source_annotation_NK','coordinates':'unchanged','seed':17,'python':sys.version,'pandas':pd.__version__,'matplotlib':matplotlib.__version__,'font':font}
    sig=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest();mp=check/f'{prefix}_manifest.json'
    manifest=json.loads(mp.read_text()) if mp.exists() else {'signature':sig,'contract':contract,'units':{},'complete':False}
    assert manifest['signature']==sig,'Inputs/code changed: existing units must not be silently reused'
    def valid(unit):
        u=manifest['units'].get(unit,{})
        return bool(u) and all((project/p).exists() and sha(project/p)==h for p,h in u['outputs'].items())
    def done(unit,paths):
        manifest['units'][unit]={'outputs':{str(p.relative_to(project)):sha(p) for p in paths},'utc':datetime.now(timezone.utc).isoformat()};js(manifest,mp)
        print('Saved',unit,flush=True)
    def export(fig,stem):
        paths=[];fig.canvas.draw()
        for ext in ['pdf','png']:
            p=figdir/f'{stem}.{ext}';tmp=figdir/f'{stem}.tmp.{ext}'
            fig.savefig(tmp,format=ext,dpi=220,facecolor='white');os.replace(tmp,p);paths.append(p)
        plt.close(fig);return paths
    data=pd.read_parquet(inputs[0]);summaries=[];previews=[];conclusions=[]
    oldcolor='#7B2CBF';newcolor='#D55E00';bgcolor='#D9D9D9'
    gate_colors={'T-eligible':'#7B2CBF','NK-call':'#D73027','abstain':'#E69F00'}
    genes=['CD3D','CD3E','CD3G','TRDC','TRGC1','TRGC2','TRAC','TRBC1','TRBC2','KLRD1','FCER1G','TYROBP','NKG7','GNLY']
    for ds in ['PANGEA','15-organ']:
        d=data[data.dataset.eq(ds)].copy();xy=pd.read_parquet(pred/f'{ds}_gate_discordance_audit_UMAP_source_data.parquet')
        assert xy.cell_id.is_unique and d.cell_id.is_unique and set(d.cell_id)==set(xy.cell_id)
        xy=xy.set_index('cell_id').reindex(d.cell_id)
        np.testing.assert_allclose(d[['p_T','p_gdT_given_T']],xy[['p_T','p_gdT_given_T']],rtol=0,atol=1e-12)
        d[['UMAP1','UMAP2']]=xy[['UMAP1','UMAP2']].to_numpy()
        d['gate_state']=np.select([d.gate_T_eligible,d.gate_NK_call],['T-eligible','NK-call'],default='abstain')
        source=d.source_annotation_NK.eq(True);original=source&d.p_gdT_given_T.ge(cut);added=source&d.p_gdT_given_T.ge(.5)&d.p_gdT_given_T.lt(cut)
        d['threshold_UMAP_original_NK_like']=original;d['threshold_UMAP_added_NK_like']=added
        d['threshold_UMAP_band']=np.select([original,added],['Original threshold','0.5 to original threshold'],default='Other cells')
        d['threshold_UMAP_has_coordinates']=np.isfinite(d[['UMAP1','UMAP2']]).all(axis=1)
        selected=d[original|added].copy()
        fields=['threshold_audit_gd_receptor_support','threshold_audit_ab_receptor_support','threshold_audit_no_paired_support','threshold_audit_mixed_conflict','threshold_audit_receptor_file_linked','threshold_audit_doublet_assessed','threshold_audit_flagged_doublet','gate_T_eligible','gate_NK_call','gate_abstain','threshold_UMAP_has_coordinates']
        def counts(keys):
            return selected.groupby(keys,dropna=False,observed=True).agg(n=('cell_id','size'),**{f:(f,'sum') for f in fields}).reset_index()
        summary=counts(['dataset','threshold_UMAP_band']);summaries.append(summary)
        unit=ds+'_tables'
        # Expression/QC are compared descriptively, retaining their source unit.
        groups=[('Original threshold',d[original]),('0.5 to original threshold',d[added]),('Source gdT annotation',d[d.source_annotation_gdT.eq(True)])]
        features=['log1p_'+g for g in genes]+['audit_total_counts','audit_n_genes_by_counts','audit_pct_counts_mt']
        stats=[]
        for label,z in groups:
            strata=[('all','all','all',z)]+[(str(a),str(b),str(c),v) for (a,b,c),v in z.groupby(['source_study','source_sample','source_annotation'],dropna=False,observed=True)]
            for study,sample,annotation,v in strata:
                for f in features:
                    q=pd.to_numeric(v[f],errors='coerce').dropna()
                    if len(q):stats.append({'dataset':ds,'group':label,'source_study':study,'source_sample':sample,'source_annotation':annotation,'feature':f,'n':len(q),'mean':float(q.mean()),'median':float(q.median()),'fraction_detected':float(q.gt(0).mean()) if f.startswith('log1p_') else np.nan})
        stats=pd.DataFrame(stats)
        keys=['dataset','source_study','source_sample','source_annotation','feature'];s=stats[~stats.source_sample.eq('all')]
        con=s[s.group.eq('0.5 to original threshold')].merge(s[s.group.eq('Original threshold')],on=keys,suffixes=('_added','_original'),validate='one_to_one')
        con=con[con.n_added.ge(5)&con.n_original.ge(5)].copy();con['mean_difference']=con.mean_added-con.mean_original;con['detection_difference']=con.fraction_detected_added-con.fraction_detected_original
        if not valid(unit):
            paths=[save(selected,pred/f'{ds}_{prefix}_cells.parquet'),save(summary,table/f'{ds}_{prefix}_summary.csv'),save(counts(['dataset','source_study','source_sample','source_organ','source_annotation','threshold_UMAP_band']),table/f'{ds}_{prefix}_by_source.csv'),save(selected[selected.threshold_audit_gd_receptor_support|selected.threshold_audit_ab_receptor_support|selected.threshold_audit_mixed_conflict],table/f'{ds}_{prefix}_receptor_records.csv'),save(stats,table/f'{ds}_{prefix}_markers.csv'),save(con,table/f'{ds}_{prefix}_within_source_contrasts.csv')]
            done(unit,paths)
        # Every panel shares full coordinate limits within the specified view.
        def bounds(v):
            return [(float(v[c].min()-.025*np.ptp(v[c])),float(v[c].max()+.025*np.ptp(v[c]))) for c in ['UMAP1','UMAP2']]
        def style(ax,lim):
            ax.set(xlim=lim[0],ylim=lim[1],xlabel='UMAP1',ylabel='UMAP2',xticks=[],yticks=[]);ax.set_aspect('equal',adjustable='box')
        v=d[d.threshold_UMAP_has_coordinates];views=[('full',v)]
        if ds=='15-organ':
            scope=d.annotation_Tlineage.eq(True)|d.original_annotation_T_NK_ambiguous.eq(True)|d.strict_paired_abTCR.eq(True)
            views.append(('T_NK',d[scope&d.threshold_UMAP_has_coordinates]))
        for view,v in views:
            stem=f'{ds}_{prefix}_{view}';previews.append(str(figdir/f'{stem}.png'))
            if valid(stem):continue
            lim=bounds(v);fig,axs=plt.subplots(2,3,figsize=(9.0,7.1));fig.subplots_adjust(left=.06,right=.98,bottom=.21,top=.86,wspace=.27,hspace=.38)
            for row,(flag,label,color) in enumerate([('threshold_UMAP_original_NK_like','Original threshold',oldcolor),('threshold_UMAP_added_NK_like','0.5 to original threshold',newcolor)]):
                z=v[v[flag]]
                for ax in axs[row]:
                    ax.scatter(v.UMAP1,v.UMAP2,s=.55,c=bgcolor,alpha=.45,lw=0);style(ax,lim)
                ref=v[v.p_gdT_given_T.ge(cut)]
                axs[row,0].scatter(ref.UMAP1,ref.UMAP2,s=1,c='#BAA3D1',alpha=.65,lw=0)
                axs[row,0].scatter(z.UMAP1,z.UMAP2,s=3,c=color,lw=0,alpha=.85)
                axs[row,0].set_title(label+f'\nNK-like records: {len(z):,}')
                for state,c in gate_colors.items():
                    q=z[z.gate_state.eq(state)];axs[row,1].scatter(q.UMAP1,q.UMAP2,s=3,c=c,lw=0,alpha=.85)
                axs[row,1].set_title('Gate state retained')
                axs[row,2].scatter(z.UMAP1,z.UMAP2,s=2,c=color,alpha=.3,lw=0)
                for flag2,mark,c in [('threshold_audit_gd_receptor_support','o','#000000'),('threshold_audit_ab_receptor_support','x','#0057B8')]:
                    q=z[z[flag2]]
                    if mark=='o':axs[row,2].scatter(q.UMAP1,q.UMAP2,s=28,facecolors='none',edgecolors=c,lw=.8)
                    else:axs[row,2].scatter(q.UMAP1,q.UMAP2,s=28,c=c,marker=mark,lw=.8)
                ng=int(z.threshold_audit_gd_receptor_support.sum());na=int(z.threshold_audit_ab_receptor_support.sum())
                axs[row,2].set_title(f'Receptor evidence\ngdT: {ng}; abT: {na}')
            fig.suptitle(ds+f' | {view} | unchanged UMAP coordinates',fontsize=11,y=.97)
            fig.text(.5,.915,f'Original: score >= {cut:.5f}; added: 0.5 <= score < {cut:.5f}',ha='center',fontsize=8)
            handles=[Line2D([],[],marker='o',ls='',c=c,label=l,markersize=4) for l,c in [('Original NK-like',oldcolor),('Added NK-like',newcolor),('All original M05 positives','#BAA3D1')]]
            fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.12),ncol=3,frameon=False)
            handles=[Line2D([],[],marker='o',ls='',c=c,label='Gate: '+l,markersize=4) for l,c in gate_colors.items()]
            fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.08),ncol=3,frameon=False)
            fig.text(.5,.052,'Receptors: black open circle = gamma-delta support; blue cross = alpha-beta support.',ha='center',fontsize=8)
            fig.text(.5,.025,'Unmarked does not mean NK. Source annotation and UMAP proximity are not independent identity evidence.',ha='center',fontsize=8)
            done(stem,export(fig,stem))
        # Facets retain the full shared coordinate system; candidate presence is the only inclusion rule.
        v=d[d.threshold_UMAP_has_coordinates];lim=bounds(v);field='source_study' if ds=='PANGEA' else 'source_organ'
        values=sorted(d.loc[added,field].dropna().astype(str).unique())
        for start in range(0,len(values),6):
            stem=f'{ds}_{prefix}_{field}_{start//6+1:02d}'
            if valid(stem):continue
            fig,axs=plt.subplots(2,3,figsize=(8.5,6.8));fig.subplots_adjust(left=.06,right=.98,bottom=.12,top=.89,wspace=.25,hspace=.34)
            for ax,value in zip(axs.flat,values[start:start+6]):
                z=v[v[field].astype(str).eq(value)];ax.scatter(z.UMAP1,z.UMAP2,s=.7,c=bgcolor,alpha=.5,lw=0)
                for flag,c in [('threshold_UMAP_original_NK_like',oldcolor),('threshold_UMAP_added_NK_like',newcolor)]:
                    q=z[z[flag]];ax.scatter(q.UMAP1,q.UMAP2,s=4,c=c,alpha=.85,lw=0)
                style(ax,lim);ax.set_title(value+f'\nOriginal: {z.threshold_UMAP_original_NK_like.sum()}; added: {z.threshold_UMAP_added_NK_like.sum()}')
            for ax in list(axs.flat)[len(values[start:start+6]):]:ax.set_visible(False)
            fig.suptitle(ds+' | '+field+' | unchanged shared coordinates',fontsize=10)
            fig.legend(handles=[Line2D([],[],marker='o',ls='',c=c,label=l,markersize=4) for l,c in [('Original NK-like',oldcolor),('Added NK-like',newcolor)]],loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.02))
            done(stem,export(fig,stem))
        stem=f'{ds}_{prefix}_markers';previews.append(str(figdir/f'{stem}.png'))
        if not valid(stem):
            pooled=stats[stats.source_sample.eq('all')&stats.feature.str.startswith('log1p_')]
            labels=[a for a,_ in groups];fig,axs=plt.subplots(2,1,figsize=(9,5.6));fig.subplots_adjust(left=.24,right=.89,bottom=.17,top=.88,hspace=.25)
            for ax,metric,title,vmax in zip(axs,['mean','fraction_detected'],['Mean log1p expression','Fraction with detected transcript'],[None,1]):
                matrix=pooled.pivot(index='group',columns='feature',values=metric).reindex(index=labels,columns=['log1p_'+g for g in genes])
                im=ax.pcolormesh(np.arange(len(genes)+1),np.arange(4),matrix.to_numpy(),cmap='viridis',vmin=0,vmax=vmax,shading='flat',edgecolors='none',linewidth=0,rasterized=False);ax.set_ylim(3,0);ax.set_yticks(np.arange(3)+.5,labels);ax.set_title(title)
                if metric=='fraction_detected':ax.set_xticks(np.arange(len(genes))+.5,genes,rotation=45,ha='right')
                else:ax.set_xticks([])
                cb=fig.colorbar(im,ax=ax,fraction=.025,pad=.025);cb.solids.set_rasterized(False)
            fig.suptitle(ds+' | RNA evidence (descriptive, not independent validation)',fontsize=10)
            fig.text(.5,.035,'Source gdT is an annotation comparator. Pooled summaries may reflect study, tissue or depth differences.',ha='center',fontsize=8)
            done(stem,export(fig,stem))
        z=d[added];old=d[original]
        conclusions.append(f'{ds}: original NK-like records {len(old):,}; added {len(z):,}; added receptor-supported gamma-delta {int(z.threshold_audit_gd_receptor_support.sum())}, alpha-beta {int(z.threshold_audit_ab_receptor_support.sum())}; gate T-eligible {int(z.gate_T_eligible.sum())}, NK-call {int(z.gate_NK_call.sum())}, abstain {int(z.gate_abstain.sum())}; added without coordinates {int((~z.threshold_UMAP_has_coordinates).sum())}.')
        new_nk=z[z.gate_NK_call]
        conclusions.append(f'Among the {len(new_nk):,} added NK-call records, {int(new_nk.threshold_audit_receptor_file_linked.sum())} have linked receptor files in the current audit, {int(new_nk.threshold_audit_gd_receptor_support.sum())} have gamma-delta support and {int(new_nk.threshold_audit_ab_receptor_support.sum())} have alpha-beta support. Missing coverage prevents an identity conclusion for the remaining records.')
        for gene in ['TRDC','TRGC1','CD3D','CD3E','CD3G']:
            conclusions.append(f'{ds} {gene} transcript detected: original NK-like {old["log1p_"+gene].gt(0).mean():.1%}; added NK-like {z["log1p_"+gene].gt(0).mean():.1%}. Descriptive pooled RNA evidence, not independent receptor confirmation.')
    interpretation='## Added NK-like candidate evidence and UMAP comparison\n\n'+'\n\n'.join(conclusions)+'\n\nEarlier work already supplied the frozen-threshold gate UMAPs and threshold evidence tables. This section adds the requested interval-specific NK-like comparisons on the original coordinates.\n\nThe three added PANGEA gamma-delta-supported records are GSE149356 HC1/HC2 NK_CD16 annotations; all are T-eligible, so they do not establish that the gate-excluded added candidates are gamma-delta T cells. Processed receptor QC flags remain incomplete. 15-organ has no gamma-delta receptor-positive reference; its added alpha-beta-supported records require conflict review.\n\nRNA marker overlap and UMAP colocalization are hypothesis-supporting patterns, not proof of identity. Absent receptor evidence cannot distinguish true NK from NK-like gamma-delta T cells. Original putative labels and all gate decisions are unchanged. The record table retains exact IDs, receptor quality provenance, QC, source metadata and known repeated-library identifiers. Expression contrasts require at least five original and added records in the same source sample and annotation; they remain descriptive and are not depth-matched causal tests.\n\nResume on Woody by running this notebook section; each verified table/figure unit is reused. No full model inference or UMAP fitting is required. All figure text is black, normal-weight '+font+'; PDFs retain vector points and text.\n'
    p=table/f'{prefix}_interpretation.md';tmp=Path(str(p)+'.tmp');tmp.write_text(interpretation);os.replace(tmp,p)
    paths=[p,save(pd.concat(summaries,ignore_index=True),table/f'{prefix}_summary.csv')];done('aggregate',paths)
    assert all(sha(project/p)==h for p,h in contract['inputs'].items())
    manifest['complete']=True;manifest['completed_utc']=datetime.now(timezone.utc).isoformat();js(manifest,mp)
    paths=[project/f for u in manifest['units'].values() for f in u['outputs']]+[mp]
    rows=[{'local_path':'/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/gdt_lr_12model/'+str(p.relative_to(project)),'Woody_path':'/data/users/kaelynwen/gdT_project/gdt_lr_12model/'+str(p.relative_to(project)),'sha256':sha(p)} for p in paths]
    save(pd.DataFrame(rows),table/f'{prefix}_output_manifest.csv')
    return {'interpretation':interpretation,'previews':[p for p in previews if '_T_NK' not in p]}


def run_gate_discordance_audit_receptor_search(project):
    """Incremental source audit; no model changes or raw-read assembly."""
    import hashlib,json,os,re,requests,inspect,xml.etree.ElementTree as ET
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime,timezone
    import pandas as pd
    project=Path(project).resolve();prefix='gate_discordance_audit_receptor_search'
    check=project/'checkpoints'/NAME;table=project/'results/tables'/NAME;pred=project/'predictions'/NAME
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def save(d,p):
        tmp=Path(str(p)+'.tmp')
        if p.suffix=='.parquet':d.to_parquet(tmp,index=False)
        else:d.to_csv(tmp,index=False)
        os.replace(tmp,p);return p
    retrievals=[]
    def fetch(url,dest):
        reused=dest.exists()
        if not reused:
            r=requests.get(url,timeout=45);r.raise_for_status();tmp=Path(str(dest)+'.tmp');tmp.write_bytes(r.content);os.replace(tmp,dest)
        return {'url':url,'path':str(dest.relative_to(project)),'sha256':sha(dest),'bytes':dest.stat().st_size,'reused':reused}
    cellpath=pred/'gate_discordance_audit_threshold_cells.parquet';registry=project/'results/tables/22_LR_hierarchical_CE/gate_discordance_audit_source_registry.csv'
    snapshots=project/'checkpoints/22_LR_hierarchical_CE/gate_discordance_audit_sources.json'
    inputs={str(p.relative_to(project)):sha(p) for p in [cellpath,registry,snapshots]}
    allcells=pd.read_parquet(cellpath);atlas=allcells[allcells.dataset.eq('PANGEA')].copy()
    focal=atlas[atlas.source_annotation_NK.eq(True)&atlas.threshold_lower.eq(.5)&atlas.gate_NK_call&~atlas.threshold_audit_gd_receptor_support&~atlas.threshold_audit_ab_receptor_support].copy()
    assert len(focal)==2882 and focal.cell_id.is_unique
    reg=pd.read_csv(registry);ledger=reg.merge(focal.groupby('source_study',observed=True).size().rename('n_focal'),on='source_study',validate='one_to_one');assert len(ledger)==106
    ledger['current_followup']='Prior series metadata screen reused; no new sample-level search in this pass'
    ledger.loc[ledger.source_study.eq('GSE168163'),'current_followup']='Earlier notebook 21 library-linkage audit supersedes old unresolved registry'
    # GSE200173: explicit same suffix for distinct mRNA/ADT/TCR GEO samples.
    gsm={}
    for n in range(6019628,6019646):
        acc='GSM'+str(n);url=f'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}&targ=self&form=text&view=full';dest=check/f'{prefix}_{acc}.txt'
        retrievals.append(fetch(url,dest));txt=dest.read_text();assert f'^SAMPLE = {acc}' in txt
        title=re.search(r'!Sample_title = (.+)',txt)[1];sup=re.search(r'!Sample_supplementary_file_1 = (.+)',txt)[1]
        gsm[acc]={'title':title,'supplement':sup,'url':url}
    study=atlas[atlas.source_study.eq('GSE200173')].copy();parts=[];links=[];quality=[]
    for i in range(6):
        rna='GSM'+str(6019628+i);adt='GSM'+str(6019634+i);tcr='GSM'+str(6019640+i)
        suffix=gsm[rna]['title'].removeprefix('mRNA_');assert gsm[tcr]['title']=='TCR_'+suffix and gsm[adt]['title']=='ADT_'+suffix
        url=gsm[tcr]['supplement'].replace('ftp://','https://');dest=check/(prefix+'_'+url.rsplit('/',1)[-1]);retrievals.append(fetch(url,dest))
        c=pd.read_csv(dest);assert c.contig_id.is_unique
        ok=pd.Series(True,index=c.index)
        for col in ['is_cell','high_confidence','full_length','productive']:ok &= c[col].astype(str).str.lower().eq('true')
        for col in ['v_gene','j_gene','cdr3','cdr3_nt']:ok &= c[col].notna()&~c[col].astype(str).str.lower().isin(['none','nan',''])
        cc=c[ok].groupby(['barcode','chain']).size().unstack(fill_value=0).reindex(columns=['TRA','TRB','TRG','TRD','Multi'],fill_value=0)
        z=study[study['Sample Name'].eq(rna)].copy();barcodes=z.cell_id.str.split('@').str[-1];assert barcodes.is_unique
        z['receptor_search_RNA_GSM']=rna;z['receptor_search_TCR_GSM']=tcr;z['receptor_search_library']=suffix
        z['receptor_search_matched_record']=barcodes.isin(c.barcode).to_numpy();q=cc.reindex(barcodes).fillna(0)
        for col in q:z['receptor_search_qualified_'+col]=q[col].to_numpy()
        z['receptor_search_paired_ab']=(q.TRA.eq(1)&q.TRB.eq(1)&q.TRG.eq(0)&q.TRD.eq(0)&q.Multi.eq(0)).to_numpy()
        z['receptor_search_paired_gd']=(q.TRG.eq(1)&q.TRD.eq(1)&q.TRA.eq(0)&q.TRB.eq(0)&q.Multi.eq(0)).to_numpy()
        z['receptor_search_focal']=z.cell_id.isin(focal.cell_id);parts.append(z)
        links.append({'RNA_GSM':rna,'ADT_GSM':adt,'TCR_GSM':tcr,'shared_library_suffix':suffix,'n_atlas':len(z),'n_matched':int(z.receptor_search_matched_record.sum()),'n_paired_ab':int(z.receptor_search_paired_ab.sum()),'n_paired_gd':int(z.receptor_search_paired_gd.sum()),'n_focal':int(z.receptor_search_focal.sum()),'n_focal_matched':int((z.receptor_search_focal&z.receptor_search_matched_record).sum()),'source_url':gsm[tcr]['url']})
        for chain,g in c.groupby('chain',dropna=False):quality.append({'TCR_GSM':tcr,'chain':chain,'n_contigs':len(g),'n_quality_pass':int(ok.loc[g.index].sum())})
        save(z,check/f'{prefix}_{rna}_cells.parquet')
        print('Matched',rna,'focal',int(z.receptor_search_focal.sum()),'records',int((z.receptor_search_focal&z.receptor_search_matched_record).sum()),flush=True)
    matched=pd.concat(parts,ignore_index=True);assert len(matched)==len(study) and matched.cell_id.is_unique
    # ENA run reports are metadata only. No FASTQ bytes are downloaded.
    ena={}
    for ds,prj in [('GSE173303','PRJNA725159'),('GSE178991','PRJNA741804')]:
        url=f'https://www.ebi.ac.uk/ena/portal/api/filereport?accession={prj}&result=read_run&fields=run_accession,experiment_accession,sample_accession,library_name,library_strategy,fastq_ftp,fastq_bytes,submitted_ftp,sample_title&format=tsv';dest=check/f'{prefix}_{ds}_ENA.txt';retrievals.append(fetch(url,dest));ena[ds]=pd.read_csv(dest,sep='\t');assert ena[ds].run_accession.is_unique
    e=ena['GSE173303'];z=focal[focal.source_study.eq('GSE173303')];run_to_sample={r:row['Sample Name'] for _,row in z[['Sample Name','SRR']].drop_duplicates().iterrows() for r in row.SRR.split(':')}
    experiments=e[e.run_accession.isin(run_to_sample)].experiment_accession.unique();todo=e[e.experiment_accession.isin(experiments)]
    def inspect_run(run):
        url='https://www.ebi.ac.uk/ena/browser/api/xml/'+run;dest=check/f'{prefix}_{run}.xml';rec=fetch(url,dest);root=ET.fromstring(dest.read_text());ref=root.find('.//EXPERIMENT_REF')
        attrs={a.findtext('TAG'):a.findtext('VALUE') for a in root.findall('.//RUN_ATTRIBUTE')};options=attrs.get('options','')
        files=re.findall(r'--read\d+PairFiles=(\S+)',options)
        # Require identical library prefix across I1/R1/R2 filenames.
        libs={re.sub(r'_S\d+_L\d+_[IR]\d_\d+\.fastq\.gz$','',f) for f in files}
        library=next(iter(libs)) if len(libs)==1 else ''
        return rec,{'run_accession':run,'GSM':ref.attrib.get('refname',''),'experiment_accession':ref.attrib['accession'],'submitted_library':library,'role':'TCR' if library.endswith('-TCR') else 'GEX' if library.endswith('-5GE') else 'unresolved','submitted_files':' | '.join(files)}
    xmlrows=[]
    with ThreadPoolExecutor(3) as pool:
        for rec,row in pool.map(inspect_run,todo.run_accession):retrievals.append(rec);xmlrows.append(row)
    runmap=pd.DataFrame(xmlrows).merge(e[['run_accession','fastq_ftp','fastq_bytes']],on='run_accession',validate='one_to_one');rawlinks=[]
    for gsm_id,v in z.groupby('Sample Name',observed=True):
        q=runmap[runmap.GSM.eq(gsm_id)];g=q[q.role.eq('GEX')];t=q[q.role.eq('TCR')]
        expected=set(v.Sample);assert set(g.submitted_library)==expected
        assert set(g.submitted_library.str.removesuffix('-5GE'))==set(t.submitted_library.str.removesuffix('-TCR'))
        rawlinks.append({'source_study':'GSE173303','RNA_GSM':gsm_id,'RNA_library':' | '.join(expected),'n_focal':len(v),'TCR_runs':' | '.join(t.run_accession),'TCR_libraries':' | '.join(sorted(set(t.submitted_library))),'linkage':'same GEO sample/experiment and matching original submitted library prefix','cell_receptor_matching':'not performed; raw-read processing required'})
    # GSE178991: exact batch plus library stems, never barcode-only cross-pool matching.
    e=ena['GSE178991'];z=focal[focal.source_study.eq('GSE178991')]
    for (gsm_id,sample,runs),v in z.groupby(['Sample Name','Sample','SRR'],observed=True):
        g=e[e.run_accession.isin(runs.split(':'))];titles=g.sample_title.unique();assert len(titles)==1
        title=titles[0];batch=re.search(r'batch(\d+)',title)[1];lib=re.search(r'\((.+)\)',title)[1].removesuffix('_fastqs')
        expect=lib.replace('_5primeGEX','_TEL').replace('_GEXL','_TEL')
        if expect==lib:expect=lib+'_TEL'
        candidates=e[e.sample_title.str.contains('VDJ',na=False)&e.sample_title.str.contains('batch'+batch+' ',regex=False,na=False)].copy()
        candidate_lib=candidates.sample_title.str.extract(r'\((.+)\)',expand=False).str.removesuffix('_fastqs');t=candidates[candidate_lib.eq(expect)];assert len(t)>0 and t.experiment_accession.nunique()==1
        rawlinks.append({'source_study':'GSE178991','RNA_GSM':gsm_id,'RNA_library':lib,'n_focal':len(v),'TCR_runs':' | '.join(t.run_accession),'TCR_libraries':expect,'linkage':'exact batch and GEX/TEL library name in official ENA sample titles','cell_receptor_matching':'not performed; processed 2,735,430,563-byte h5ad.gz or raw reads required'})
    for kind in ['whole','subT']:
        url=f'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE173nnn/GSE173303/suppl/GSE173303_{"whole_data" if kind=="whole" else "subT_data"}_metadata.tsv.gz';dest=check/f'{prefix}_GSE173303_{kind}_metadata.gz';retrievals.append(fetch(url,dest))
        # Metadata annotations are not receptor sequences; inspect schema only.
        columns=pd.read_csv(dest,sep='\t',nrows=0).columns.tolist();assert not any(re.search('cdr3|clonotyp|contig|productive',c,re.I) for c in columns)
    for ds,txt in {'GSE200173':'New sample-level contigs matched across 6 verified libraries; 297/582 atlas records matched, 188 strict alpha-beta pairs, zero gamma-delta pairs; focal 29/29 lack contig records','GSE173303':'New original-file-name audit verified TCR raw runs for 10 focal libraries / 58 focal records; processed metadata contains annotations but no CDR3 fields; no raw assembly performed','GSE178991':'New batch/library-name audit verified VDJ raw runs for 19 focal libraries / 41 focal records; official h5ad includes TCR annotations, 2.735 GB compressed; not downloaded'}.items():ledger.loc[ledger.source_study.eq(ds),'current_followup']=txt
    outputs=[save(ledger,table/f'{prefix}_study_ledger.csv'),save(matched,pred/f'GSE200173_{prefix}_cells.parquet'),save(pd.DataFrame(links),table/f'GSE200173_{prefix}_library_linkage.csv'),save(pd.DataFrame(quality),table/f'GSE200173_{prefix}_quality.csv'),save(runmap,table/f'GSE173303_{prefix}_runs.csv'),save(pd.DataFrame(rawlinks),table/f'{prefix}_raw_library_linkage.csv'),save(pd.DataFrame(retrievals),table/f'{prefix}_retrievals.csv')]
    text='''# Incremental PANGEA receptor-source audit

Scope: source_annotation_NK, 0.5 <= optimized_M05 score < frozen threshold, gate NK-call, without earlier paired receptor support. All original labels and model predictions are unchanged.

All 106 source studies for the 2,882 focal records had prior Series-level metadata screening. This pass reused that registry. It does not claim an exhaustive new sample-level search of every study. Earlier GSE149356/GSE165045 matches and the later GSE168163 library audit were not rerun. GSE144469, GSE167272 and GSE169657 retain their previously documented unresolved linkage limitations.

New findings:

- GSE200173: six TCR contig files were located on individual GSM pages. Exact official mRNA/TCR library suffixes plus intact barcodes link 297 of 582 atlas records; 188 have strict paired alpha-beta evidence, none have paired gamma-delta evidence. All 29 focal records have no matching contig. This establishes library coverage without establishing NK identity for missing records. One TRG contig and no TRD occur across the downloaded files. No new gamma-delta-supported focal cell was found.
- GSE173303: original submitted filenames distinguish GEX from TCR despite both being catalogued as RNA-Seq under the same GEO samples. Matching prefixes verify raw TCR libraries for 10 focal libraries and 58 focal records. The two inspected processed metadata tables contain annotations, not CDR3 sequences. Raw cell-level reconstruction remains unperformed; public FASTQ transformations require barcode/read-layout checks before use.
- GSE178991: exact batch and GEX/TEL names identify VDJ raw libraries corresponding to 19 focal GEX libraries and 41 focal records. The official sample description states that the merged h5ad includes TCR annotations. Its gzip file is 2,735,430,563 bytes; it has not been downloaded in this bounded source audit. Library correspondence is not cell-level receptor evidence.

Priority: inspect the GSE178991 processed receptor fields before large raw reassembly; inspect GSE173303 raw read/barcode layout next. Failure to find a receptor in an alpha-beta-enriched assay cannot exclude gamma-delta identity. Existing GSE200173 protein measurements may provide additional lineage information, but ADT matrices have not been inspected here. No receptor-negative cell was relabelled NK, and no new putative label was assigned.

Without new wet-lab work, a productive paired TRG/TRD sequence with validated cell/library linkage is stronger identity evidence than RNA/UMAP agreement. Suitable existing FASTQ/BAM can be inspected for receptor reconstruction; a count matrix alone cannot reconstruct CDR3. Existing cell-matched CD3/TCR-gamma-delta protein can contribute if available. Independent receptor-anchored references and technical QC can prioritize unresolved candidates, but expression-only agreement does not establish definitive identity.

Sources: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM6019642 ; https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5402919 ; https://www.nature.com/articles/s41467-022-29539-3 ; https://github.com/liulab-dfci/TRUST4 . Exact retrieval URLs and checksums are recorded in the retrieval table.

Reproduce on Woody: `cd /data/users/kaelynwen/gdT_project/gdt_lr_12model && /data/conda_envs/kaelynwen/ENTER/bin/python -c "from pathlib import Path; import runpy; print(runpy.run_path('build_21_gate_LRM05_evaluation_15organs.py')['run_gate_discordance_audit_receptor_search'](Path.cwd()))"`. Download units are atomically cached; per-library matched tables are saved before aggregate output. No model fitting or UMAP computation is performed.
'''
    report=table/f'{prefix}.md';tmp=Path(str(report)+'.tmp');tmp.write_text(text);os.replace(tmp,report);outputs.append(report)
    assert all(sha(project/p)==h for p,h in inputs.items())
    meta={'completed_utc':datetime.now(timezone.utc).isoformat(),'inputs':inputs,'source_function_sha256':hashlib.sha256(inspect.getsource(run_gate_discordance_audit_receptor_search).encode()).hexdigest(),'cohort_n':len(focal),'prior_screen_studies':len(ledger),'new_followup_studies':['GSE200173','GSE173303','GSE178991'],'no_model_change':True,'raw_reads_downloaded':False}
    path=table/f'{prefix}_manifest.json';tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(meta,indent=2));os.replace(tmp,path);outputs.append(path)
    artifacts=outputs+[project/r['path'] for r in retrievals]+list(check.glob(f'{prefix}_GSM*_cells.parquet'))
    records=[{'local_path':'/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/gdt_lr_12model/'+str(p.relative_to(project)),'Woody_path':'/data/users/kaelynwen/gdT_project/gdt_lr_12model/'+str(p.relative_to(project)),'sha256':sha(p)} for p in dict.fromkeys(artifacts)]
    save(pd.DataFrame(records),table/f'{prefix}_output_manifest.csv')
    return text


def prepare_gate_discordance_audit_GSE178991(project):
    """Acquire and inspect the official processed input, without loading expression."""
    import gzip, hashlib, json, os, shutil, subprocess
    from datetime import datetime, timezone
    import h5py
    from anndata.io import read_elem
    project = Path(project).resolve()
    check = project / 'checkpoints' / NAME
    prefix = 'gate_discordance_audit_GSE178991'
    url = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE178nnn/GSE178991/suppl/GSE178991_20201228_adata.h5ad.gz'
    compressed = check / url.rsplit('/', 1)[-1]
    expected_bytes = 2735430563
    def sha(path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()
    def save_json(value, path):
        tmp = Path(str(path) + '.tmp')
        tmp.write_text(json.dumps(value, indent=2) + '\n')
        os.replace(tmp, path)
    if not compressed.exists():
        partial = Path(str(compressed) + '.part')
        subprocess.run(['wget', '-c', '--progress=dot:giga', '-O', str(partial), url], check=True)
        assert partial.stat().st_size == expected_bytes
        os.replace(partial, compressed)
    assert compressed.stat().st_size == expected_bytes
    compressed_sha = sha(compressed)
    h5ad = compressed.with_suffix('')
    receipt = check / (prefix + '_input.json')
    old = json.loads(receipt.read_text()) if receipt.exists() else {}
    valid = (h5ad.exists() and old.get('compressed_sha256') == compressed_sha
             and old.get('h5ad_sha256') == sha(h5ad))
    if not valid:
        tmp = Path(str(h5ad) + '.tmp')
        with gzip.open(compressed, 'rb') as src, open(tmp, 'wb') as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        # Reaching EOF verifies the gzip CRC and uncompressed size.
        with h5py.File(tmp, 'r') as f:
            assert 'obs' in f and 'X' in f
        os.replace(tmp, h5ad)
    with h5py.File(h5ad, 'r') as f:
        obs = read_elem(f['obs'])
        schema = {'root_keys': list(f.keys()), 'obs_columns': list(obs.columns),
                  'n_obs': len(obs), 'obs_index_unique': bool(obs.index.is_unique),
                  'obsm_keys': list(f['obsm']) if 'obsm' in f else [],
                  'uns_keys': list(f['uns']) if 'uns' in f else []}
    assert obs.index.is_unique
    obs_path = check / (prefix + '_obs.parquet')
    tmp = Path(str(obs_path) + '.tmp')
    obs.to_parquet(tmp)
    os.replace(tmp, obs_path)
    save_json({'url': url, 'expected_compressed_bytes': expected_bytes,
               'compressed_sha256': compressed_sha, 'h5ad_sha256': sha(h5ad),
               'obs_sha256': sha(obs_path), 'schema': schema,
               'gzip_crc_verified': True, 'completed_utc': datetime.now(timezone.utc).isoformat()}, receipt)
    print(json.dumps(schema, indent=2), flush=True)
    return obs


def run_gate_discordance_audit_GSE178991(project):
    """Cell/library-resolved processed-receptor audit; never relabel atlas cells."""
    import hashlib, inspect, json, os, re, sys
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime, timezone
    import pandas as pd
    import requests
    project = Path(project).resolve()
    check = project / 'checkpoints' / NAME
    table = project / 'results/tables' / NAME
    pred = project / 'predictions' / NAME
    prefix = 'gate_discordance_audit_GSE178991'
    def sha(path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
        return h.hexdigest()
    def save(value, path):
        tmp = Path(str(path) + '.tmp')
        if path.suffix == '.parquet': value.to_parquet(tmp, index=False)
        elif path.suffix == '.csv': value.to_csv(tmp, index=False)
        elif path.suffix == '.json': tmp.write_text(json.dumps(value, indent=2) + '\n')
        else: tmp.write_text(value)
        os.replace(tmp, path)
        return path
    obs_path = check / (prefix + '_obs.parquet')
    receipt_path = check / (prefix + '_input.json')
    if not receipt_path.exists(): prepare_gate_discordance_audit_GSE178991(project)
    receipt = json.loads(receipt_path.read_text())
    assert sha(obs_path) == receipt['obs_sha256']
    obs = pd.read_parquet(obs_path)
    assert obs.index.is_unique and len(obs) == receipt['schema']['n_obs']
    required = ['sample', 'seq_batch', 'patient', 'timepoint', 'CDR3A', 'CDR3B',
                'CDR3a', 'CDR3b', 'TRAV', 'TRAJ', 'TRBV', 'TRBJ']
    assert set(required) <= set(obs.columns)
    assert obs.CDR3A.astype(str).equals(obs.CDR3a.astype(str))
    assert obs.CDR3B.astype(str).equals(obs.CDR3b.astype(str))
    source = pred / 'gate_discordance_audit_threshold_cells.parquet'
    allcells = pd.read_parquet(source)
    study = allcells[allcells.dataset.eq('PANGEA') & allcells.source_study.eq('GSE178991')].copy()
    assert study.cell_id.is_unique and len(study) == 4096
    study['receptor_search_focal'] = (study.source_annotation_NK & study.threshold_lower.eq(.5)
        & study.gate_NK_call & ~study.threshold_audit_gd_receptor_support
        & ~study.threshold_audit_ab_receptor_support)
    assert study.receptor_search_focal.sum() == 41
    def fetch(gsm):
        path = check / (prefix + '_' + gsm + '.txt')
        url = f'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gsm}&targ=self&form=text&view=full'
        if not path.exists():
            r = requests.get(url, timeout=60); r.raise_for_status()
            assert '^SAMPLE = ' + gsm in r.text
            save(r.text, path)
        text = path.read_text()
        assert '^SAMPLE = ' + gsm in text
        return gsm, text, path, url
    with ThreadPoolExecutor(max_workers=4) as pool:
        metadata = {r[0]: r[1:] for r in pool.map(fetch, sorted(study['Sample Name'].unique()))}
    # Name conversions remove only documented library suffixes and date prefixes.
    # Exact sequencing batch and GEO patient/timepoint membership are checked below.
    def sample_key(s):
        s = s.strip()
        if s.startswith('020218_'): return s.removeprefix('020218_')
        if s.endswith('_GEXL'): return s.removesuffix('_GEXL')
        if s.startswith('sample_'): return s.removeprefix('sample_').replace('_5primeGEX', '_GEX')
        return s
    code_sha = hashlib.sha256(inspect.getsource(run_gate_discordance_audit_GSE178991).encode()).hexdigest()
    source_sha = sha(source)
    parts, libraries, artifacts = [], [], [obs_path, receipt_path]
    for (gsm, sample), rows in study.groupby(['Sample Name', 'Sample'], observed=True):
        txt, metadata_path, url = metadata[gsm]
        title = re.search(r'!Sample_title = (.+)', txt)[1].strip()
        official_library = re.search(r'\((.+)\)', title)[1].removesuffix('_fastqs')
        assert official_library == sample.strip(), (gsm, sample, title)
        batch = 'batch' + re.search(r'batch(\d+)', title)[1]
        mapped_sample = sample_key(sample)
        original = obs[obs['sample'].astype(str).eq(mapped_sample)].copy()
        assert len(original) and set(original.seq_batch.astype(str)) == {batch}
        official_pairs = set((a.replace('-', ''), 'd' + b) for a, b in
            re.findall(r'patient id, time point: ([\d-]+) Day(\d+)', txt))
        observed_pairs = set(zip(original.patient.astype(str), original.timepoint.astype(str)))
        patient_conflicts = sorted(observed_pairs - official_pairs)
        # Preserve metadata inconsistencies rather than silently resolving them.
        assert official_pairs
        assert all(i.endswith('_' + mapped_sample) for i in original.index)
        z = rows.copy()
        barcode = z.cell_id.str.split('@').str[-1]
        assert barcode.str.fullmatch(r'[ACGT]{16}-1').all()
        z['receptor_search_h5ad_index'] = barcode.str.removesuffix('-1') + '_' + mapped_sample
        assert z.receptor_search_h5ad_index.is_unique
        z['receptor_search_matched_record'] = z.receptor_search_h5ad_index.isin(original.index)
        selected = obs.reindex(z.receptor_search_h5ad_index)
        for column in obs.columns:
            z['receptor_search_' + column] = selected[column].to_numpy()
        z['receptor_search_patient_timepoint_verified'] = [
            (str(a), str(b)) in official_pairs
            for a, b in zip(selected.patient, selected.timepoint)]
        z['receptor_search_patient_timepoint_conflict'] = (
            z.receptor_search_matched_record & ~z.receptor_search_patient_timepoint_verified)
        def present(column):
            s = z['receptor_search_' + column]
            return s.notna() & ~s.astype(str).str.lower().isin(['nan', 'none', '', 'na'])
        z['receptor_search_CDR3a_present'] = present('CDR3a')
        z['receptor_search_CDR3b_present'] = present('CDR3b')
        z['receptor_search_paired_ab_annotation'] = (
            present('CDR3a') & present('CDR3b') & present('TRAV') & present('TRAJ')
            & present('TRBV') & present('TRBJ'))
        z['receptor_search_any_ab_annotation'] = present('CDR3a') | present('CDR3b')
        z['receptor_search_gamma_delta_assessable'] = False
        z['receptor_search_contig_quality_available'] = False
        z['receptor_search_RNA_GSM'] = gsm
        signature = {'source_sha256': source_sha, 'obs_sha256': receipt['obs_sha256'],
                     'metadata_sha256': sha(metadata_path), 'code_sha256': code_sha,
                     'GSM': gsm, 'sample': sample, 'expected_rows': len(z),
                     'expected_columns': z.columns.tolist()}
        unit = check / (prefix + '_' + gsm + '_cells.parquet')
        unit_manifest = check / (prefix + '_' + gsm + '_complete.json')
        # Each scientifically complete library is independently durable and validated.
        previous = json.loads(unit_manifest.read_text()) if unit_manifest.exists() else {}
        if (unit.exists() and previous.get('signature') == signature
                and previous.get('output_sha256') == sha(unit)):
            cached = pd.read_parquet(unit)
            assert len(cached) == len(z) and cached.columns.tolist() == z.columns.tolist()
            assert cached.cell_id.tolist() == z.cell_id.tolist()
            z = cached
        else:
            save(z, unit)
            save({'signature': signature, 'output_sha256': sha(unit)}, unit_manifest)
        focal = z.receptor_search_focal
        libraries.append({'RNA_GSM': gsm, 'RNA_library': sample, 'h5ad_sample': mapped_sample,
            'seq_batch': batch, 'patient_timepoint_membership_verified': not patient_conflicts,
            'patient_timepoint_conflicts': json.dumps(patient_conflicts),
            'n_atlas_patient_timepoint_conflicts': int(z.receptor_search_patient_timepoint_conflict.sum()),
            'n_atlas': len(z), 'n_h5ad_matched': int(z.receptor_search_matched_record.sum()),
            'n_any_ab_annotation': int(z.receptor_search_any_ab_annotation.sum()),
            'n_paired_ab_annotation': int(z.receptor_search_paired_ab_annotation.sum()),
            'n_focal': int(focal.sum()),
            'n_focal_matched': int(z.loc[focal, 'receptor_search_matched_record'].sum()),
            'n_focal_any_ab_annotation': int(z.loc[focal, 'receptor_search_any_ab_annotation'].sum()),
            'source_url': url, 'metadata_sha256': sha(metadata_path)})
        parts.append(z); artifacts.extend([metadata_path, unit, unit_manifest])
        save(pd.DataFrame(libraries), check / (prefix + '_progress.csv'))
        print('Validated', gsm, sample, 'matched', int(z.receptor_search_matched_record.sum()),
              '/', len(z), 'focal', int(focal.sum()), flush=True)
    out = pd.concat(parts, ignore_index=True)
    assert len(out) == len(study) and out.cell_id.is_unique
    assert set(out.cell_id) == set(study.cell_id)
    # Original model scores, gate states and all pre-existing evidence remain identical.
    pd.testing.assert_frame_equal(out.set_index('cell_id')[study.columns.drop('cell_id')].sort_index(),
                                  study.set_index('cell_id').sort_index(), check_dtype=False)
    artifacts.append(save(out, pred / (prefix + '_cells.parquet')))
    artifacts.append(save(out.loc[out.receptor_search_focal], table / (prefix + '_focal_cells.csv')))
    artifacts.append(save(pd.DataFrame(libraries), table / (prefix + '_library_linkage.csv')))
    summaries = []
    for name, mask in [('GSE178991', pd.Series(True, index=out.index)),
                       ('source_annotation_NK', out.source_annotation_NK),
                       ('source_annotation_gdT', out.source_annotation_gdT),
                       ('receptor_search_focal', out.receptor_search_focal)]:
        z = out.loc[mask]
        summaries.append({'cohort': name, 'n': len(z),
            'n_h5ad_matched': int(z.receptor_search_matched_record.sum()),
            'n_unmatched': int((~z.receptor_search_matched_record).sum()),
            'n_any_ab_annotation': int(z.receptor_search_any_ab_annotation.sum()),
            'n_paired_ab_annotation': int(z.receptor_search_paired_ab_annotation.sum()),
            'n_patient_timepoint_conflicts': int(z.receptor_search_patient_timepoint_conflict.sum()),
            'n_paired_ab_annotation_with_verified_membership': int((z.receptor_search_paired_ab_annotation & z.receptor_search_patient_timepoint_verified).sum()),
            'gamma_delta_assessable': False, 'contig_quality_available': False})
    summary = pd.DataFrame(summaries)
    focal_summary = summary.loc[summary.cohort.eq('receptor_search_focal')].iloc[0]
    assert focal_summary.n_h5ad_matched == 37 and focal_summary.n_unmatched == 4
    assert focal_summary.n_any_ab_annotation == 0 and focal_summary.n_patient_timepoint_conflicts == 0
    artifacts.append(save(summary, table / (prefix + '_summary.csv')))
    quality = []
    for col in ['CDR3a', 'CDR3b', 'TRAV', 'TRAJ', 'TRBV', 'TRBJ']:
        s = obs[col].astype(str)
        quality.append({'field': col, 'n_total': len(obs),
            'n_present': int((~s.str.lower().isin(['nan', 'none', '', 'na'])).sum()),
            'n_delimited_multiple_values': int(s.str.contains(r';|,|\|').sum())})
    artifacts.append(save(pd.DataFrame(quality), table / (prefix + '_field_coverage.csv')))
    final_manifest = {'completed_utc': datetime.now(timezone.utc).isoformat(),
        'source_function_sha256': code_sha, 'input_table_sha256': source_sha,
        'input_receipt': receipt, 'python': sys.version, 'pandas': pd.__version__,
        'required_libraries': int(study['Sample Name'].nunique()), 'completed_libraries': len(parts),
        'summary': summaries, 'model_or_threshold_changes': False,
        'relabelled_cells': 0, 'raw_read_reconstruction_performed': False}
    artifacts.append(save(final_manifest, table / (prefix + '_manifest.json')))
    artifacts.append(check / (prefix + '_progress.csv'))
    summary_md = ('| ' + ' | '.join(summary.columns) + ' |\n'
        + '| ' + ' | '.join(['---'] * len(summary.columns)) + ' |\n'
        + '\n'.join('| ' + ' | '.join(map(str, row)) + ' |'
                    for row in summary.itertuples(index=False, name=None)))
    report = f'''# GSE178991 processed-receptor audit

The official merged h5ad contains {len(obs):,} processed cells. The exported receptor fields are alpha/beta CDR3 amino-acid annotations and alpha/beta V/J annotations. There are no gamma/delta receptor annotation columns, no receptor nucleotide sequences, and no per-contig productive, full_length, high_confidence or is_cell flags. Expression of TRG/TRD genes in an RNA matrix is not a reconstructed receptor sequence. This file cannot establish paired gamma/delta receptor support or its absence.

Matching uses exact GEX library names from GEO, sequencing batch, an explicit sample-name conversion, and intact 16-base barcodes (only the -1 suffix is removed). Every library is checked against GEO patient/timepoint membership; no barcode-only cross-library join is allowed. Six libraries have additional patient/timepoint combinations in the h5ad that are absent from their GEO lists: I_1/I_2/I_3 have 002017 d0 and II_1/II_2/II_3 have 002005 d28. The 129 affected atlas records are retained and explicitly flagged, not treated as membership-verified evidence. None is in the focal group. The cause of these metadata discrepancies is unresolved. The h5ad is author-filtered/demultiplexed, whereas PANGEA cells were selected separately. Missing cells must remain unmatched; their specific exclusion reasons are not established by this audit.

{summary_md}

The focal group is unchanged: source_annotation_NK, 0.5 <= optimized_M05 score < its frozen threshold, gate_NK_call, and no earlier paired receptor support. Of its 41 records, 37 match the processed h5ad and have no alpha/beta CDR3 annotation; four do not match. None gains gamma/delta identity evidence from this file. These are unresolved records, not confirmed NK cells. Alpha/beta annotations in other matched cells provide a positive matching control, but their missing contig-quality information prevents calling them quality-verified paired receptors. Original labels, model scores, gate states and earlier evidence flags are preserved.

This processed-file avenue has now been inspected. Repeating its absence-of-TCR check or adding expression/UMAP plots will not resolve these 41 identities. A distinct next analysis would inspect the already located raw GEX/VDJ read layouts and available gamma/delta sequence coverage before any receptor reconstruction; an alpha/beta-targeted library may lack such coverage. RNA-derived hypotheses may remain putative and must be kept separate from receptor-supported identities.

Source: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178991 . Every contributing GSM URL is recorded in the library-linkage table.

Reproduce on Woody: `cd /data/users/kaelynwen/gdT_project/gdt_lr_12model && /data/conda_envs/kaelynwen/ENTER/bin/python -c "from pathlib import Path; import runpy; runpy.run_path('build_21_gate_LRM05_evaluation_15organs.py')['run_gate_discordance_audit_GSE178991'](Path.cwd())"`.

Acquisition, if needed: invoke `prepare_gate_discordance_audit_GSE178991` from the same source. Compressed downloads resume using the .part file. Decompression is atomic and verifies the gzip CRC; extracted observations are hash checked. Each completed library has a cell table, configuration/input/schema signature and output hash. Interruption preserves completed libraries. The audit reuses only signature- and hash-validated units. Final aggregation requires all 42 libraries. No model or UMAP computation is repeated.
'''
    artifacts.append(save(report, table / (prefix + '.md')))
    local_root = Path('/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/gdt_lr_12model')
    remote_root = Path('/data/users/kaelynwen/gdT_project/gdt_lr_12model')
    manifest_rows = []
    for path in artifacts + [project / 'build_21_gate_LRM05_evaluation_15organs.py',
        check / 'GSE178991_20201228_adata.h5ad.gz', check / 'GSE178991_20201228_adata.h5ad']:
        rel = path.relative_to(project)
        manifest_rows.append({'relative_path': str(rel), 'local_path': str(local_root / rel),
            'Woody_path': str(remote_root / rel), 'sha256': sha(path), 'bytes': path.stat().st_size,
            'local_copy_expected': path.suffix != '.h5ad'})
    save(pd.DataFrame(manifest_rows), table / (prefix + '_output_manifest.csv'))
    print(summary.to_string(index=False), flush=True)
    return summary


def run_gate_discordance_audit_GSE178991_V8(project, n_spots=100000):
    """Bounded native-SRA feasibility probe, not receptor assembly or relabelling."""
    import collections, gzip, hashlib, inspect, io, itertools, json, os, re, subprocess, sys
    import tarfile, xml.etree.ElementTree as ET
    from datetime import datetime, timezone
    import pandas as pd
    import requests
    project = Path(project).resolve()
    check = project / 'checkpoints' / NAME
    table = project / 'results/tables' / NAME
    prefix = 'gate_discordance_audit_GSE178991_V8'
    assert 1 <= n_spots <= 100000, 'This entry point is a bounded feasibility probe.'
    artifacts = []
    def sha(p):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
        return h.hexdigest()
    def save(value, p):
        tmp = Path(str(p) + '.tmp')
        if p.suffix == '.csv': value.to_csv(tmp, index=False)
        elif p.suffix == '.json': tmp.write_text(json.dumps(value, indent=2) + '\n')
        else: tmp.write_text(value)
        os.replace(tmp, p)
        if p not in artifacts: artifacts.append(p)
        return p
    def fetch(url, p):
        if not p.exists():
            r = requests.get(url, timeout=60); r.raise_for_status(); save(r.text, p)
        if p not in artifacts: artifacts.append(p)
        return p.read_text()
    source = project / 'predictions' / NAME / 'gate_discordance_audit_GSE178991_cells.parquet'
    cells = pd.read_parquet(source)
    cells = cells[cells.Sample.astype(str).eq('V8')].copy()
    assert len(cells) == 98 and cells.receptor_search_focal.sum() == 6
    assert cells.receptor_search_matched_record.all()
    obs_path = check / 'gate_discordance_audit_GSE178991_obs.parquet'
    obs = pd.read_parquet(obs_path)
    obs = obs[obs['sample'].astype(str).eq('V8')]
    whitelist = {i.removesuffix('_V8') for i in obs.index}
    atlas_barcodes = set(cells.cell_id.str.split('@').str[-1].str.removesuffix('-1'))
    focal_barcodes = set(cells.loc[cells.receptor_search_focal, 'cell_id'].str.split('@').str[-1].str.removesuffix('-1'))
    known_cdr3 = {r.cell_id.split('@')[-1].removesuffix('-1'):
        [str(r.receptor_search_CDR3a), str(r.receptor_search_CDR3b)] for r in cells.itertuples()}
    standard_code = dict(zip(map(''.join, itertools.product('TCAG', repeat=3)),
        'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'))
    assert standard_code['ATG'] == 'M' and standard_code['TGA'] == '*'
    runs = ['SRR14925065', 'SRR14925066', 'SRR14925067', 'SRR14925068',
            'SRR14925077', 'SRR14925078', 'SRR14925079', 'SRR14925080']
    ena = pd.read_csv(check / 'gate_discordance_audit_receptor_search_GSE178991_ENA.txt', sep='\t')
    metadata, run_rows = {}, []
    for run in runs:
        ena_path = check / (prefix + '_' + run + '_ENA.xml')
        eroot = ET.fromstring(fetch('https://www.ebi.ac.uk/ena/browser/api/xml/' + run, ena_path))
        npath = check / (prefix + '_' + run + '_NCBI.xml')
        nroot = ET.fromstring(fetch('https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/run_new?acc=' + run, npath))
        rnode = nroot.find('RUN'); assert rnode.attrib['accession'] == run
        options = next(x.findtext('VALUE') for x in eroot.findall('.//RUN_ATTRIBUTE') if x.findtext('TAG') == 'options')
        assert '--readTypes=TTB' in options and 'V8' in options
        kind = 'VDJ' if 'V8_TEL' in options else 'GEX'
        src = next(x for x in rnode.findall('./SRAFiles/SRAFile') if x.attrib.get('semantic_name') == 'SRA Normalized')
        lengths = [int(x.attrib['average']) for x in rnode.findall('./Statistics/Read')]
        assert lengths == [8, 150, 150]
        exported = ena.loc[ena.run_accession.eq(run)].iloc[0]
        assert ';' not in exported.fastq_ftp
        row = {'run_accession': run, 'library': 'V8_TEL' if kind == 'VDJ' else 'V8',
               'kind': kind, 'total_spots': int(rnode.attrib['total_spots']),
               'original_read_lengths': json.dumps(lengths), 'original_read_types': 'TTB',
               'original_submitted_files': options, 'native_sra_url': src.attrib['url'],
               'native_sra_bytes': int(src.attrib['size']), 'native_sra_md5': src.attrib['md5'],
               'ENA_fastq_url': 'https://' + exported.fastq_ftp,
               'ENA_fastq_bytes': int(exported.fastq_bytes),
               'ENA_metadata_sha256': sha(ena_path), 'NCBI_metadata_sha256': sha(npath)}
        metadata[run] = row; run_rows.append(row)
        save(pd.DataFrame(run_rows), check / (prefix + '_progress.csv'))
    save(pd.DataFrame(run_rows), table / (prefix + '_run_inventory.csv'))
    # Official, isolated binary distribution. No environment installation or config editing.
    archive = check / 'sratoolkit.3.4.1-ubuntu64.tar.gz'
    if not archive.exists():
        r = requests.get('https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/3.4.1/' + archive.name, stream=True, timeout=60)
        r.raise_for_status(); tmp = Path(str(archive) + '.part')
        with open(tmp, 'wb') as f:
            for block in r.iter_content(1024 * 1024): f.write(block)
        os.replace(tmp, archive)
    assert hashlib.md5(archive.read_bytes()).hexdigest() == 'ec6e9056a2bfebcf23c6cd6e02951ef2'
    binary = check / 'sratoolkit.3.4.1-ubuntu64/bin/vdb-dump'
    if not binary.exists():
        with tarfile.open(archive) as tar: tar.extractall(check, filter='data')
    tool_version = subprocess.check_output([str(binary), '--version'], text=True).strip()
    # Existing reference is copied into the audit cache to make the probe self-contained.
    reference = check / (prefix + '_regions.fa')
    original_reference = Path('/data/users/kaelynwen/references/cellranger-vdj/refdata-cellranger-vdj-GRCh38-alts-ensembl-7.1.0/fasta/regions.fa')
    if not reference.exists(): save(original_reference.read_text(), reference)
    else: artifacts.append(reference)
    records = []
    for block in reference.read_text().split('>')[1:]:
        header, sequence = block.split('\n', 1)
        fields = header.split('|'); records.append((fields[3], fields[5], ''.join(sequence.split()).upper()))
    k = 31
    def rc(s): return s.translate(str.maketrans('ACGTN', 'TGCAN'))[::-1]
    owners = collections.defaultdict(set)
    constants = collections.defaultdict(set)
    for region, chain, sequence in records:
        for i in range(len(sequence) - k + 1):
            word = sequence[i:i+k]
            if set(word) <= set('ACGT'):
                for w in [word, rc(word)]:
                    owners[w].add(chain)
                    if region == 'C-REGION': constants[w].add(chain)
    # Screening signal only: exact constant-region matches unique by chain within
    # this receptor reference, not globally unique genomic mappings or CDR3s.
    lookup = {w: next(iter(c)) for w, c in constants.items() if len(c) == 1 and owners[w] == c}
    def chain_hits(sequence):
        starts = collections.defaultdict(list)
        for i in range(len(sequence) - k + 1):
            chain = lookup.get(sequence[i:i+k])
            if chain: starts[chain].append(i)
        return {c for c, positions in starts.items() if max(positions) - min(positions) >= k}
    summaries, percell = [], []
    for run in ['SRR14925065', 'SRR14925077']:
        meta = metadata[run]
        raw = check / (prefix + '_' + run + '_spots.json.gz')
        receipt = check / (prefix + '_' + run + '_complete.json')
        signature = {'run': run, 'url': meta['native_sra_url'], 'advertised_sra_md5': meta['native_sra_md5'],
            'n_spots': n_spots, 'first_spot': 1, 'columns': 'READ_LEN,READ_TYPE,READ_START,READ',
            'tool_sha256': sha(binary), 'tool_version': tool_version}
        old = json.loads(receipt.read_text()) if receipt.exists() else {}
        if not (raw.exists() and old.get('signature') == signature and old.get('output_sha256') == sha(raw)):
            tmp = Path(str(raw) + '.json.tmp')
            cmd = [str(binary), meta['native_sra_url'], '-R', f'1-{n_spots}', '-C', signature['columns'], '-f', 'json', '-n']
            log = check / (prefix + '_' + run + '_vdb.log')
            with open(tmp, 'w') as f, open(log, 'w') as err:
                subprocess.run(cmd, cwd=check, stdout=f, stderr=err, check=True, timeout=600)
            records_raw = json.loads(tmp.read_text()); assert len(records_raw) == n_spots
            assert [r['row_id'] for r in records_raw] == list(range(1, n_spots + 1))
            compressed_tmp = Path(str(raw) + '.tmp')
            with open(tmp, 'rb') as f, gzip.open(compressed_tmp, 'wb') as out:
                import shutil
                shutil.copyfileobj(f, out)
            os.replace(compressed_tmp, raw); tmp.unlink()
            save({'signature': signature, 'output_sha256': sha(raw), 'full_SRA_checksum_verified': False,
                  'note': 'Bounded native SRA row query, not a complete SRA download.'}, receipt)
        else:
            with gzip.open(raw, 'rt') as f: records_raw = json.load(f)
        artifacts.extend([raw, receipt])
        log_path = check / (prefix + '_' + run + '_vdb.log')
        if log_path.exists(): artifacts.append(log_path)
        ena_probe = check / (prefix + '_' + run + '_ENA_1000.fastq.gz')
        if not ena_probe.exists():
            r = requests.get(meta['ENA_fastq_url'], stream=True, timeout=60); r.raise_for_status()
            r.raw.decode_content = False
            reader = gzip.GzipFile(fileobj=r.raw)
            content = b''.join(reader.readline() for _ in range(4000)); r.close()
            tmp = Path(str(ena_probe) + '.tmp')
            with gzip.open(tmp, 'wb') as f: f.write(content)
            os.replace(tmp, ena_probe)
        artifacts.append(ena_probe)
        with gzip.open(ena_probe, 'rt') as f: lines = f.read().splitlines()
        assert len(lines) == 4000 and all(lines[i].startswith('@') for i in range(0, 4000, 4))
        compared = min(1000, n_spots)
        for i in range(compared):
            record = records_raw[i]; start = record['READ_START'][2]; length = record['READ_LEN'][2]
            assert lines[4*i+1] == record['READ'][start:start+length]
            assert lines[4*i].endswith('/3')
        counts = collections.Counter(); counts_by_barcode = collections.defaultdict(collections.Counter)
        for r in records_raw:
            assert r['READ_LEN'] == [8, 150, 150] and r['READ_TYPE'] == [0, 0, 1]
            assert r['READ_START'] == [0, 8, 158] and len(r['READ']) == 308
            r1 = r['READ'][8:158]; r2 = r['READ'][158:308]; barcode = r1[:16]
            counts['spots'] += 1
            if barcode in whitelist: counts['spots_with_h5ad_V8_barcode'] += 1
            if barcode in atlas_barcodes: counts['spots_with_atlas_barcode'] += 1
            if barcode in focal_barcodes: counts['spots_with_focal_barcode'] += 1
            if r2[:16] in whitelist: counts['R2_prefix_matching_h5ad_barcode'] += 1
            bc = counts_by_barcode[barcode]; bc['spots'] += 1
            # Remove 16-base barcode + assumed 10-base UMI before R1 screening.
            # R2 is also screened independently; no reconstruction uses this assumption.
            for chain in chain_hits(r1[26:]) | chain_hits(r2):
                counts['spots_' + chain + '_constant_screen'] += 1
                bc['spots_' + chain + '_constant_screen'] += 1
            if barcode in known_cdr3:
                translated = []
                for sequence in [r1[26:], r2]:
                    for oriented in [sequence, rc(sequence)]:
                        for frame in range(3):
                            translated.append(''.join(standard_code.get(oriented[i:i+3], 'X')
                                for i in range(frame, len(oriented) - 2, 3)))
                for chain, cdr3 in zip(['a', 'b'], known_cdr3[barcode]):
                    if cdr3 not in ['nan', 'None', ''] and any(cdr3 in peptide for peptide in translated):
                        label = 'spots_matching_processed_CDR3' + chain
                        counts[label] += 1; bc[label] += 1
        summary = {'run_accession': run, 'kind': meta['kind'], **dict(counts),
                   'ENA_first_1000_equal_native_read3': True,
                   'distinct_h5ad_V8_barcodes': len(set(counts_by_barcode) & whitelist),
                   'distinct_focal_barcodes': len(set(counts_by_barcode) & focal_barcodes),
                   'atlas_cells_matching_processed_CDR3': sum(
                       bool(v['spots_matching_processed_CDR3a'] or v['spots_matching_processed_CDR3b'])
                       for bc, v in counts_by_barcode.items() if bc in atlas_barcodes)}
        summaries.append(summary)
        for row in cells.itertuples():
            barcode = row.cell_id.split('@')[-1].removesuffix('-1')
            bc = counts_by_barcode[barcode]
            percell.append({'run_accession': run, 'kind': meta['kind'], 'cell_id': row.cell_id,
                'barcode': barcode, 'receptor_search_focal': bool(row.receptor_search_focal),
                'source_annotation': row.source_annotation,
                'processed_CDR3a': row.receptor_search_CDR3a, 'processed_CDR3b': row.receptor_search_CDR3b,
                'processed_paired_ab_annotation': bool(row.receptor_search_paired_ab_annotation),
                'sampled_spots': bc['spots'], **{c: bc[c] for c in [
                    'spots_TRA_constant_screen', 'spots_TRB_constant_screen',
                    'spots_TRG_constant_screen', 'spots_TRD_constant_screen',
                    'spots_matching_processed_CDR3a', 'spots_matching_processed_CDR3b']}})
        save(pd.DataFrame(summaries).fillna(0), check / (prefix + '_probe_progress.csv'))
        print('Completed native SRA probe', json.dumps(summary), flush=True)
    save(pd.DataFrame(summaries).fillna(0), table / (prefix + '_read_probe_summary.csv'))
    save(pd.DataFrame(percell), table / (prefix + '_cell_read_probe.csv'))
    final = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
        'tool_version': tool_version, 'source_sha256': hashlib.sha256(inspect.getsource(run_gate_discordance_audit_GSE178991_V8).encode()).hexdigest(),
        'inputs': {str(source): sha(source), str(obs_path): sha(obs_path), str(reference): sha(reference)},
        'sample_definition': f'First {n_spots} spots of one GEX and one VDJ run; not random and not full-library coverage',
        'read_screen': 'At least two non-overlapping exact 31-mers from the same constant-region chain, unique by chain within receptor reference only; either orientation; not globally unique alignment or CDR3 evidence',
        'R1_screen_trim': '16-base barcode plus assumed 10-base UMI; this is not a validated reconstruction configuration',
        'summaries': summaries, 'receptor_assembly_performed': False, 'identity_labels_changed': False,
        'model_or_threshold_changed': False}
    save(final, table / (prefix + '_manifest.json'))
    probe_cells = pd.DataFrame(percell)
    focal = probe_cells[probe_cells.receptor_search_focal]
    focal_wide = focal.pivot(index='barcode', columns='kind', values='sampled_spots').fillna(0)
    focal_md = '| V8 barcode | GEX sampled spots | VDJ sampled spots |\n| --- | ---: | ---: |\n'
    focal_md += '\n'.join(f'| {bc} | {int(r.GEX)} | {int(r.VDJ)} |' for bc, r in focal_wide.iterrows())
    report = f'''# GSE178991 V8 raw-read feasibility audit

Question: can original V8 cell barcodes be recovered, and does a bounded raw-read probe justify gamma/delta receptor reconstruction? This is a technical feasibility audit, not a cell-identity classifier or a complete receptor reconstruction.

## Inputs and native read structure

The inventory covers four V8 GEX runs (SRR14925065-SRR14925068) and four V8_TEL VDJ runs (SRR14925077-SRR14925080). No existing matching raw files were found in the inspected local project and Woody user directories before this probe. Official ENA/NCBI metadata lists the original I1/R1/R2 submissions, with technical/technical/biological read classes (TTB). All eight native runs have 8/150/150-base read lengths. Native SRA preserves all three reads. Two independently retrieved ENA FASTQ prefixes match the native third read exactly for the first 1,000 spots in each tested run. Thus the public ENA FASTQ route used here omits the first two reads, including the barcode-containing R1; it must not be used alone for cell-resolved assembly.

Native SRA inventory sizes: GEX {sum(r['native_sra_bytes'] for r in run_rows if r['kind']=='GEX'):,} bytes; VDJ {sum(r['native_sra_bytes'] for r in run_rows if r['kind']=='VDJ'):,} bytes. These complete libraries were not downloaded. The probe retrieved the first {n_spots:,} native spots of SRR14925065 and SRR14925077, preserving all reads. This is a contiguous technical probe, not a random sample or comprehensive cell coverage. The original SRA advertised checksum is recorded but a whole-file checksum was not verified for these partial queries.

## Barcode and receptor-sequence checks

The 16-base R1 prefix is compared without error correction to the 11,594 author-retained V8 cell barcodes, the unchanged 98 V8 atlas cells, and the six focal candidates. The focal group retains the previously frozen definition: source NK annotation, 0.5 <= optimized_M05 score < original threshold, gate NK call, and no earlier paired receptor support.

{pd.DataFrame(summaries).fillna(0).to_csv(index=False)}

Focal candidate coverage (spots are sequencing records, not independent molecules):

{focal_md}

The constant-region screen requires two non-overlapping exact 31-base matches supporting the same chain in either orientation. Matches are unique by chain within the saved Cell Ranger VDJ reference, not verified unique genome-wide. R1 is screened after removing the 16-base barcode and assumed 10-base UMI; R2 is also screened. This is only a sequence-screening criterion: it does not identify a rearranged V(D)J junction or a productive receptor. No full gamma/delta positive control or sensitivity benchmark is supplied by this probe. Zero TRG/TRD screen hits therefore cannot establish their absence or NK identity.

As a separate technical positive control, each atlas cell's original CDR3a/CDR3b amino-acid sequence is searched exactly across six-frame translations of its own sampled cDNA reads. A hit is a processed-sequence concordance check, not a newly assembled receptor or proof of a complete alpha/beta pair. GEX has {int(summaries[0].get('atlas_cells_matching_processed_CDR3',0))} atlas cells with such matches; VDJ has {int(summaries[1].get('atlas_cells_matching_processed_CDR3',0))}. The cell-level table records the corresponding counts and original sequences. This validates the barcode-to-receptor linkage for the positive controls while leaving gamma/delta detection sensitivity unestablished.

## Interpretation and next decision

Technical linkage is feasible through native SRA with technical reads retained. This probe has not resolved the identities of any of the six focal V8 cells or the broader 41 GSE178991 candidates. The sampled focal reads are sparse and not UMI-deduplicated. Base-quality scores were not requested in this bounded native query; full quality-aware reconstruction requires retrieving them. All model scores, gate states, source annotations and existing evidence remain unchanged.

The next bounded step should use substantially more sequence coverage from one verified native V8_TEL run to recover the known alpha/beta CDR3 controls and inspect raw TRG/TRD V/J/junction evidence, retaining R1 barcode/UMI information. Gamma/delta capability must be evaluated separately; alpha/beta control recovery is not a gamma/delta sensitivity estimate. If the library lacks suitable gamma/delta coverage, consider appropriately reconstructed 5-prime GEX reads and a receptor-supported gamma/delta positive control with comparable chemistry. Do not treat negative assembly as NK proof and do not expand to all eight complete runs before evaluating the pilot. No raw receptor assembly was launched in this audit.

Original study methods describe 5-prime GEX/VDJ, Cell Ranger 3.1.0 and demuxlet singlet filtering; they do not establish gamma/delta-specific enrichment in the inspected methods: https://insight.jci.org/articles/view/147474 . Native read provenance is recorded per run from https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/run_new and https://www.ebi.ac.uk/ena/browser/api/xml/ .

## Reproducibility and resume

Authoritative source: build_21_gate_LRM05_evaluation_15organs.py, callable run_gate_discordance_audit_GSE178991_V8. The original notebook execution is not repeated. Each run probe is saved atomically with its run URL, advertised input checksum, queried rows, requested schema, tool version/hash and output SHA256. Only validated probe files are reused. All final outputs are assembled after both probes complete. Restart command on Woody:

`cd /data/users/kaelynwen/gdT_project/gdt_lr_12model && /data/conda_envs/kaelynwen/ENTER/bin/python -c "from pathlib import Path; import runpy; runpy.run_path('build_21_gate_LRM05_evaluation_15organs.py')['run_gate_discordance_audit_GSE178991_V8'](Path.cwd())"`

The output manifest lists absolute local and Woody paths, sizes and hashes. No manuscript or Overleaf file was changed.
'''
    save(report, table / (prefix + '.md'))
    # All intermediate and final locations are explicit, including cached native probes.
    local = Path('/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/gdt_lr_12model')
    woody = Path('/data/users/kaelynwen/gdT_project/gdt_lr_12model')
    artifacts += [project / 'build_21_gate_LRM05_evaluation_15organs.py', archive]
    paths = []
    for p in sorted(set(artifacts)):
        rel = p.relative_to(project)
        paths.append({'relative_path': str(rel), 'local_path': str(local / rel),
            'Woody_path': str(woody / rel), 'bytes': p.stat().st_size, 'sha256': sha(p)})
    save(pd.DataFrame(paths), table / (prefix + '_output_manifest.csv'))
    return final


def run_gate_discordance_audit_GSE178991_V8_reconstruction(project):
    """Full single-run extraction and barcode-resolved TRUST4 reconstruction."""
    import collections, concurrent.futures, gzip, hashlib, json, os, shutil, subprocess
    from datetime import datetime, timezone
    import pandas as pd
    project = Path(project).resolve(); check = project / 'checkpoints' / NAME
    table = project / 'results/tables' / NAME
    prefix = 'gate_discordance_audit_GSE178991_V8_SRR14925077'
    def digest(p, alg='sha256'):
        h = hashlib.new(alg)
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
        return h.hexdigest()
    def js(value, path):
        tmp = Path(str(path) + '.tmp'); tmp.write_text(json.dumps(value, indent=2) + '\n'); os.replace(tmp, path)
    raw = check / 'SRR14925077'
    expected_md5 = '8b2d8b71a0a4a86e1507522ac4f25b19'
    if not raw.exists():
        partial = Path(str(raw) + '.part')
        subprocess.run(['wget', '-c', '--progress=dot:giga', '-O', str(partial),
            'https://sra-pub-run-odp.s3.amazonaws.com/sra/SRR14925077/SRR14925077'], check=True)
        assert partial.stat().st_size == 2474400807 and digest(partial, 'md5') == expected_md5
        os.replace(partial, raw)
    assert raw.stat().st_size == 2474400807 and digest(raw, 'md5') == expected_md5
    raw_sha = digest(raw)
    source = project / 'predictions' / NAME / 'gate_discordance_audit_GSE178991_cells.parquet'
    cells = pd.read_parquet(source); cells = cells[cells.Sample.astype(str).eq('V8')].copy()
    assert len(cells) == 98 and cells.receptor_search_focal.sum() == 6 and cells.cell_id.is_unique
    whitelist = {c.split('@')[-1].removesuffix('-1').encode() for c in cells.cell_id}
    assert len(whitelist) == 98
    tool = check / 'sratoolkit.3.4.1-ubuntu64/bin/fastq-dump'
    config = {'input_sha256': raw_sha, 'input_md5': expected_md5, 'cell_table_sha256': digest(source),
        'whitelist': sorted(b.decode() for b in whitelist), 'tool_sha256': digest(tool),
        'total_spots': 25057514, 'chunk_spots': 2000000,
        'read_schema': 'Unsplit 308bp spot: I1[0:8], R1[8:158], R2[158:308]; preserve Phred+33',
        'barcode_policy': 'Exact 16-base R1 prefix; no correction; no quality-based pre-exclusion',
        'output_schema': 'Paired original 150bp R1/R2 FASTQ; same SRR.spot identifier',
        'UMI_schema': 'R1[16:26]; retain raw bases and qualities'}
    config_path = check / (prefix + '_reconstruction_config.json'); js(config, config_path)
    def extract(bounds):
        lo, hi = bounds; unit = f'{prefix}_{lo:08d}_{hi:08d}'
        receipt = check / (unit + '_extraction_complete.json')
        outputs = [check / (unit + '_R1.fastq.gz'), check / (unit + '_R2.fastq.gz')]
        signature = {**config, 'lo': lo, 'hi': hi}
        old = json.loads(receipt.read_text()) if receipt.exists() else {}
        if old.get('signature') == signature and all(p.exists() and digest(p) == old.get('hashes', {}).get(p.name) for p in outputs):
            print('Reused extraction', lo, hi, 'selected', old['selected_spots'], flush=True); return old
        temp = [Path(str(p) + '.tmp') for p in outputs]
        log = check / (unit + '_extraction.log')
        cmd = [str(tool), '-Z', '-M', '1', '-N', str(lo), '-X', str(hi), str(raw)]
        counts = collections.Counter(); tso = collections.Counter(); qc = collections.Counter()
        processed = 0
        with open(log, 'wb') as err, gzip.open(temp[0], 'wb', compresslevel=3) as out1, gzip.open(temp[1], 'wb', compresslevel=3) as out2:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, cwd=check)
            try:
                stream = proc.stdout
                while True:
                    header = stream.readline()
                    if not header: break
                    sequence = stream.readline().rstrip(b'\r\n'); plus = stream.readline(); quality = stream.readline().rstrip(b'\r\n')
                    assert header.startswith(b'@SRR14925077.') and plus.startswith(b'+')
                    assert len(sequence) == len(quality) == 308
                    spot = int(header.split(None, 1)[0].split(b'.')[-1])
                    assert spot == lo + processed
                    processed += 1
                    barcode = sequence[8:24]
                    if barcode not in whitelist: continue
                    r1, r2 = sequence[8:158], sequence[158:308]
                    q1, q2 = quality[8:158], quality[158:308]
                    rid = b'@SRR14925077.' + str(spot).encode() + b'\n'
                    out1.write(rid + r1 + b'\n+\n' + q1 + b'\n')
                    out2.write(rid + r2 + b'\n+\n' + q2 + b'\n')
                    counts[barcode.decode()] += 1; tso[r1[26:39].decode()] += 1
                    qc['barcode_all_Q20'] += int(min(q1[:16]) >= 53)
                    qc['UMI_all_Q20'] += int(min(q1[16:26]) >= 53)
                assert proc.wait() == 0
            except BaseException:
                proc.kill(); proc.wait(); raise
        assert processed == hi - lo + 1
        for tmp, out in zip(temp, outputs): os.replace(tmp, out)
        result = {'signature': signature, 'processed_spots': processed, 'selected_spots': sum(counts.values()),
            'barcode_counts': dict(counts), 'TSO_sequences': dict(tso), 'quality_counts': dict(qc),
            'hashes': {p.name: digest(p) for p in outputs}, 'command': cmd,
            'completed_utc': datetime.now(timezone.utc).isoformat()}
        js(result, receipt)
        print('Completed extraction', lo, hi, 'selected', result['selected_spots'], flush=True)
        return result
    bounds = [(lo, min(lo + config['chunk_spots'] - 1, config['total_spots']))
              for lo in range(1, config['total_spots'] + 1, config['chunk_spots'])]
    units = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(extract, b): b for b in bounds}
        for future in concurrent.futures.as_completed(futures):
            units.append(future.result())
            js({'required_units': len(bounds), 'completed_units': len(units),
                'processed_spots': sum(u['processed_spots'] for u in units),
                'selected_spots': sum(u['selected_spots'] for u in units)}, check / (prefix + '_extraction_progress.json'))
    assert sum(u['processed_spots'] for u in units) == config['total_spots']
    counts = collections.Counter(); tso = collections.Counter(); qc = collections.Counter()
    for u in units:
        counts.update(u['barcode_counts']); tso.update(u['TSO_sequences']); qc.update(u['quality_counts'])
    assert tso['TTTCTTATATGGG'] / sum(counts.values()) > .8, 'Check kit structure before UMI-based assembly'
    fastqs = [check / (prefix + '_R1.fastq.gz'), check / (prefix + '_R2.fastq.gz')]
    for mate, path in enumerate(fastqs, start=1):
        tmp = Path(str(path) + '.tmp')
        with open(tmp, 'wb') as out:
            for lo, hi in bounds:
                q = check / (f'{prefix}_{lo:08d}_{hi:08d}_R{mate}.fastq.gz')
                with open(q, 'rb') as f: shutil.copyfileobj(f, out)
        os.replace(tmp, path)
    coverage = cells.copy()
    coverage['reconstruction_selected_spots'] = [counts[c.split('@')[-1].removesuffix('-1')] for c in cells.cell_id]
    coverage_path = table / (prefix + '_extraction_cells.csv')
    tmp = Path(str(coverage_path) + '.tmp'); coverage.to_csv(tmp, index=False); os.replace(tmp, coverage_path)
    js({'config': config, 'completed_units': len(units), 'selected_spots': sum(counts.values()),
        'TSO_exact_fraction': tso['TTTCTTATATGGG'] / sum(counts.values()), 'quality_counts': dict(qc),
        'fastq_sha256': {p.name: digest(p) for p in fastqs}}, table / (prefix + '_extraction_manifest.json'))
    trust = check / 'TRUST4'; ref = trust / 'human_IMGT+C.fa'
    assert ref.exists() and all(('>' + chain) in ref.read_text() for chain in ['TRAV', 'TRBV', 'TRGV', 'TRDV', 'TRGJ', 'TRDJ'])
    commit = subprocess.check_output(['git', '-C', str(trust), 'rev-parse', 'HEAD'], text=True).strip()
    assembly_prefix = prefix + '_TRUST4'
    cmd = [str(trust / 'run-trust4'), '-f', str(ref), '--ref', str(ref),
        '-1', str(fastqs[0]), '-2', str(fastqs[1]), '--barcode', str(fastqs[0]), '--UMI', str(fastqs[0]),
        '--readFormat', 'bc:0:15,um:16:25,r1:39:-1,r2:0:-1', '-t', '4',
        '--outputReadAssignment', '--od', str(check), '-o', assembly_prefix]
    signature = {'command': cmd, 'trust4_commit': commit, 'reference_sha256': digest(ref),
        'input_fastq_sha256': {p.name: digest(p) for p in fastqs},
        'binary_sha256': {name: digest(trust / name) for name in ['trust4', 'fastq-extractor', 'annotator', 'run-trust4']}}
    assembly_receipt = check / (prefix + '_assembly_complete.json')
    expected = [check / (assembly_prefix + suffix) for suffix in ['_barcode_report.tsv', '_barcode_airr.tsv', '_cdr3.out', '_annot.fa']]
    old = json.loads(assembly_receipt.read_text()) if assembly_receipt.exists() else {}
    if not (old.get('signature') == signature and all(p.exists() and digest(p) == old.get('hashes', {}).get(p.name) for p in expected)):
        log = check / (prefix + '_assembly.log')
        with open(log, 'w') as f: subprocess.run(cmd, cwd=trust, stdout=f, stderr=subprocess.STDOUT, check=True)
        assert all(p.exists() for p in expected)
        js({'signature': signature, 'hashes': {p.name: digest(p) for p in expected},
            'completed_utc': datetime.now(timezone.utc).isoformat()}, assembly_receipt)
    print('Complete native-run extraction and TRUST4 assembly', sum(counts.values()), 'selected spots', flush=True)
    return {'assembly_receipt': str(assembly_receipt), 'coverage': str(coverage_path), 'outputs': list(map(str, expected))}


def assess_gate_discordance_audit_GSE178991_V8_reconstruction(project):
    """Validate non-imputed junctions against original reads; retain uncertainty."""
    import collections, gzip, hashlib, inspect, itertools, json, os, subprocess
    from datetime import datetime, timezone
    import pandas as pd
    project = Path(project).resolve(); check = project / 'checkpoints' / NAME; table = project / 'results/tables' / NAME
    prefix = 'gate_discordance_audit_GSE178991_V8_SRR14925077'
    trust = check / 'TRUST4'; original = check / (prefix + '_TRUST4')
    noimp = check / (prefix + '_TRUST4_noImputation')
    def sha(p):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
        return h.hexdigest()
    def save(value, path):
        tmp = Path(str(path) + '.tmp')
        if path.suffix == '.csv': value.to_csv(tmp, index=False)
        elif path.suffix == '.json': tmp.write_text(json.dumps(value, indent=2) + '\n')
        else: tmp.write_text(value)
        os.replace(tmp, path); return path
    assembly = json.loads((check / (prefix + '_assembly_complete.json')).read_text())
    for name, digest in assembly['hashes'].items(): assert sha(check / name) == digest
    cmd = [str(trust / 'annotator'), '-f', str(trust / 'human_IMGT+C.fa'),
        '-a', str(original) + '_final.out', '-r', str(original) + '_assembled_reads.fa',
        '-o', str(noimp), '-t', '4', '--barcode', '--UMI', '--noImpute', '--airrAlignment']
    nonimp_files = [Path(str(noimp) + s) for s in ['_annot.fa', '_cdr3.out', '_airr_align.tsv']]
    signature = {'command': cmd, 'input_hashes': {s: sha(Path(str(original) + s)) for s in ['_final.out', '_assembled_reads.fa']},
        'reference_sha256': sha(trust / 'human_IMGT+C.fa'), 'annotator_sha256': sha(trust / 'annotator')}
    receipt = check / (prefix + '_noImputation_complete.json')
    old = json.loads(receipt.read_text()) if receipt.exists() else {}
    if not (old.get('signature') == signature and all(p.exists() and sha(p) == old.get('hashes', {}).get(p.name) for p in nonimp_files)):
        with open(nonimp_files[0], 'w') as out, open(check / (prefix + '_noImputation.log'), 'w') as err:
            subprocess.run(cmd, stdout=out, stderr=err, cwd=trust, check=True)
        save({'signature': signature, 'hashes': {p.name: sha(p) for p in nonimp_files}}, receipt)
    columns = ['consensus_id', 'index_within_consensus', 'V_gene', 'D_gene', 'J_gene', 'C_gene',
        'CDR1', 'CDR2', 'CDR3_nt', 'CDR3_score', 'TRUST4_UMI_count', 'CDR3_germline_similarity', 'complete_vdj']
    calls = pd.read_csv(nonimp_files[1], sep='\t', names=columns, keep_default_na=False)
    calls['barcode'] = calls.consensus_id.str.rsplit('_', n=1).str[0]
    calls['chain'] = calls.J_gene.str[:3].where(calls.J_gene.ne('*'), calls.C_gene.str[:3])
    calls.loc[~calls.chain.isin(['TRA','TRB','TRG','TRD','IGH','IGK','IGL']), 'chain'] = 'unresolved'
    code = dict(zip(map(''.join, itertools.product('TCAG', repeat=3)),
        'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'))
    def translate(s):
        if not s or s == '*' or len(s) % 3: return ''
        return ''.join(code.get(s[i:i+3], 'X') for i in range(0, len(s), 3))
    calls['CDR3_aa'] = calls.CDR3_nt.map(translate)
    calls['nonimputed_productive_junction'] = (calls.CDR3_score.gt(.01) & calls.V_gene.ne('*') & calls.J_gene.ne('*')
        & calls.CDR3_aa.ne('') & ~calls.CDR3_aa.str.contains(r'\*|X', regex=True))
    coverage_path = table / (prefix + '_extraction_cells.csv')
    cells = pd.read_csv(coverage_path, dtype={'receptor_search_patient': str})
    cells['barcode'] = cells.cell_id.str.split('@').str[-1].str.removesuffix('-1')
    assert set(calls.barcode) <= set(cells.barcode)
    # Keep every chain and secondary contig; do not rely on the two-chain barcode report.
    unique = calls[calls.CDR3_nt.str.fullmatch('[ACGT]+')].drop_duplicates(['barcode','chain','CDR3_nt']).copy()
    keys_by_barcode = collections.defaultdict(list)
    for row in unique.itertuples(): keys_by_barcode[row.barcode].append((row.chain, row.CDR3_nt))
    rc = lambda s: s.translate(str.maketrans('ACGT','TGCA'))[::-1]
    support = collections.defaultdict(lambda: {'spots': 0, 'umis': set(), 'hq_spots': 0, 'hq_umis': set()})
    selected_counts = collections.Counter(); raw_umis = collections.defaultdict(set)
    r1path = check / (prefix + '_R1.fastq.gz'); r2path = check / (prefix + '_R2.fastq.gz')
    with gzip.open(r1path, 'rt') as f1, gzip.open(r2path, 'rt') as f2:
        while True:
            h1, h2 = f1.readline(), f2.readline()
            if not h1:
                assert not h2; break
            r1, r2 = f1.readline().strip(), f2.readline().strip()
            plus1, plus2 = f1.readline(), f2.readline()
            q1, q2 = f1.readline().strip(), f2.readline().strip()
            assert h1 == h2 and plus1.startswith('+') and plus2.startswith('+')
            assert len(r1) == len(r2) == len(q1) == len(q2) == 150
            barcode, umi = r1[:16], r1[16:26]
            selected_counts[barcode] += 1; raw_umis[barcode].add(umi)
            barcode_umi_hq = min(map(ord, q1[:26])) >= 53 and set(umi) <= set('ACGT')
            for chain, nt in keys_by_barcode[barcode]:
                found = False; good = False
                for sequence, quality in [(r1[39:], q1[39:]), (r2, q2)]:
                    for query in [nt, rc(nt)]:
                        pos = sequence.find(query)
                        while pos >= 0:
                            found = True
                            good |= barcode_umi_hq and min(map(ord, quality[pos:pos+len(query)])) >= 53
                            pos = sequence.find(query, pos+1)
                if found:
                    s = support[(barcode, chain, nt)]; s['spots'] += 1; s['umis'].add(umi)
                    if good: s['hq_spots'] += 1; s['hq_umis'].add(umi)
    assert sum(selected_counts.values()) == int(cells.reconstruction_selected_spots.sum()) == 150976
    for label in ['raw_exact_spots','raw_exact_distinct_UMIs','raw_Q20_spots','raw_Q20_distinct_UMIs']:
        calls[label] = 0
    for i, row in calls.iterrows():
        s = support[(row.barcode, row.chain, row.CDR3_nt)]
        calls.loc[i, ['raw_exact_spots','raw_exact_distinct_UMIs','raw_Q20_spots','raw_Q20_distinct_UMIs']] = [s['spots'],len(s['umis']),s['hq_spots'],len(s['hq_umis'])]
    calls['nonimputed_junction_with_2_Q20_UMIs'] = calls.nonimputed_productive_junction & calls.raw_Q20_distinct_UMIs.ge(2)
    shared = calls[calls.raw_Q20_distinct_UMIs.ge(1)].groupby(['chain','CDR3_nt']).barcode.nunique()
    calls['n_barcodes_sharing_Q20_junction'] = [int(shared.get((r.chain,r.CDR3_nt),0)) for r in calls.itertuples()]
    # A shared junction is a review flag, not automatic contamination: real clones also share CDR3.
    for chain in ['TRA','TRB','TRG','TRD']:
        z = calls[calls.chain.eq(chain) & calls.nonimputed_junction_with_2_Q20_UMIs]
        counts = z.groupby('barcode').CDR3_nt.nunique()
        cells['reconstruction_'+chain+'_junctions_2_Q20_UMIs'] = cells.barcode.map(counts).fillna(0).astype(int)
    cells['reconstruction_paired_ab_2_Q20_UMIs'] = cells.reconstruction_TRA_junctions_2_Q20_UMIs.gt(0) & cells.reconstruction_TRB_junctions_2_Q20_UMIs.gt(0)
    cells['reconstruction_paired_gd_2_Q20_UMIs'] = cells.reconstruction_TRG_junctions_2_Q20_UMIs.gt(0) & cells.reconstruction_TRD_junctions_2_Q20_UMIs.gt(0)
    for chain, original_col in [('TRA','receptor_search_CDR3a'),('TRB','receptor_search_CDR3b')]:
        flags = []; flags2 = []
        for row in cells.itertuples():
            known = str(getattr(row, original_col))
            z = calls[calls.barcode.eq(row.barcode) & calls.chain.eq(chain) & calls.CDR3_aa.eq(known) & calls.nonimputed_productive_junction]
            flags.append(bool((z.raw_Q20_distinct_UMIs >= 1).any()))
            flags2.append(bool(z.nonimputed_junction_with_2_Q20_UMIs.any()))
        cells['reconstruction_matches_processed_'+chain] = flags
        cells['reconstruction_matches_processed_'+chain+'_2_UMIs'] = flags2
    cells['reconstruction_mixed_ab_gd'] = (cells.reconstruction_TRA_junctions_2_Q20_UMIs.gt(0) | cells.reconstruction_TRB_junctions_2_Q20_UMIs.gt(0)) & (cells.reconstruction_TRG_junctions_2_Q20_UMIs.gt(0) | cells.reconstruction_TRD_junctions_2_Q20_UMIs.gt(0))
    save(calls, table / (prefix + '_junction_evidence.csv'))
    save(cells, table / (prefix + '_reconstruction_cells.csv'))
    save(cells[cells.receptor_search_focal], table / (prefix + '_focal_reconstruction.csv'))
    summaries = []
    for name, z in [('V8', cells), ('receptor_search_focal',cells[cells.receptor_search_focal]),
                    ('processed_paired_ab_annotation',cells[cells.receptor_search_paired_ab_annotation])]:
        summaries.append({'cohort':name,'n':len(z),'selected_spots':int(z.reconstruction_selected_spots.sum()),
            'paired_ab_2_Q20_UMIs':int(z.reconstruction_paired_ab_2_Q20_UMIs.sum()),
            'paired_gd_2_Q20_UMIs':int(z.reconstruction_paired_gd_2_Q20_UMIs.sum()),
            'matches_processed_TRA':int(z.reconstruction_matches_processed_TRA.sum()),
            'matches_processed_TRB':int(z.reconstruction_matches_processed_TRB.sum()),
            'matches_both_processed_chains':int((z.reconstruction_matches_processed_TRA & z.reconstruction_matches_processed_TRB).sum())})
    save(pd.DataFrame(summaries), table / (prefix + '_reconstruction_summary.csv'))
    # Same-cell comparison at the original frozen child threshold only.
    threshold_path = project / 'predictions' / NAME / 'gate_discordance_audit_threshold_cells.parquet'
    allcells = pd.read_parquet(threshold_path); comparisons=[]
    for dataset, z in allcells.groupby('dataset', observed=True):
        selected = z.p_gdT_given_T.ge(.9394744727488467)
        gd = z.threshold_audit_gd_receptor_support; ab = z.threshold_audit_ab_receptor_support
        comparisons.append({'dataset':dataset,'threshold':.9394744727488467,'gd_reference_n':int(gd.sum()),
            'ab_reference_n':int(ab.sum()),'optimized_M05_gd_detected':int((selected & gd).sum()),
            'gate_optimized_M05_gd_detected':int((selected & gd & z.gate_T_eligible).sum()),
            'optimized_M05_ab_called_gd':int((selected & ab).sum()),
            'gate_optimized_M05_ab_called_gd':int((selected & ab & z.gate_T_eligible).sum()),
            'gd_lost_gate_NK':int((selected & gd & z.gate_NK_call).sum()),
            'gd_lost_gate_abstain':int((selected & gd & z.gate_abstain).sum())})
    save(pd.DataFrame(comparisons), table / (prefix + '_model_comparison.csv'))
    summary = {'completed_utc':datetime.now(timezone.utc).isoformat(),'cohorts':summaries,
        'n_nonimputed_calls':len(calls), 'calls_by_J_C_chain':calls.chain.value_counts().to_dict(),
        'source_function_sha256':hashlib.sha256(inspect.getsource(assess_gate_discordance_audit_GSE178991_V8_reconstruction).encode()).hexdigest(),
        'input_hashes':{str(p):sha(p) for p in [coverage_path, threshold_path, r1path, r2path, *nonimp_files]},
        'quality_rule':'Non-imputed in-frame CDR3 with V and J calls, no stop/unknown amino acids, exact full-junction read matches, >=2 distinct raw UMIs with barcode+UMI+junction all Q20; same-cell pairing; raw UMI counts are not error-corrected',
        'no_identity_or_model_changes':True,'model_recommendation':'Use ungated optimized_M05 for recall-oriented candidate discovery; retain gate as review information, not an automatic veto. No universal accuracy superiority established.'}
    save(summary, table / (prefix + '_assessment_manifest.json'))
    print(json.dumps(summary['cohorts'],indent=2),flush=True)
    print(cells.loc[cells.receptor_search_focal, ['barcode','reconstruction_selected_spots']+[c for c in cells if c.endswith('_junctions_2_Q20_UMIs')]].to_string(index=False),flush=True)
    return summary


def report_gate_discordance_audit_GSE178991_V8_reconstruction(project):
    """Publish the completed evidence audit and scoped model recommendation locally."""
    import hashlib, json, os, platform, subprocess
    from datetime import datetime, timezone
    import pandas as pd
    project = Path(project).resolve(); check = project / 'checkpoints' / NAME; table = project / 'results/tables' / NAME
    prefix = 'gate_discordance_audit_GSE178991_V8_SRR14925077'
    def sha(p):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
        return h.hexdigest()
    def save(value, p):
        tmp = Path(str(p)+'.tmp')
        if p.suffix == '.csv': value.to_csv(tmp,index=False)
        elif p.suffix == '.json': tmp.write_text(json.dumps(value,indent=2)+'\n')
        else: tmp.write_text(value)
        os.replace(tmp,p)
    def markdown(df):
        return ('| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+
            '\n'.join('| '+' | '.join(map(str,row))+' |' for row in df.itertuples(index=False,name=None)))
    cells = pd.read_csv(table/(prefix+'_reconstruction_cells.csv'))
    calls = pd.read_csv(table/(prefix+'_junction_evidence.csv'))
    assessment = json.loads((table/(prefix+'_assessment_manifest.json')).read_text())
    for p,digest in assessment['input_hashes'].items(): assert sha(Path(p)) == digest
    focal = cells[cells.receptor_search_focal]
    assert len(cells)==98 and len(focal)==6 and focal.reconstruction_selected_spots.sum()==5801
    focal_calls = calls[calls.barcode.isin(focal.barcode) & calls.nonimputed_productive_junction]
    assert focal_calls.raw_exact_distinct_UMIs.max()==1
    assert cells.reconstruction_paired_gd_2_Q20_UMIs.sum()==0
    summary = pd.read_csv(table/(prefix+'_reconstruction_summary.csv'))
    comparisons = pd.read_csv(table/(prefix+'_model_comparison.csv'))
    assert comparisons.loc[comparisons.dataset.eq('SPL'),'optimized_M05_gd_detected'].iloc[0]==227
    assert comparisons.loc[comparisons.dataset.eq('SPL'),'gate_optimized_M05_gd_detected'].iloc[0]==205
    qc = focal_calls.drop_duplicates(['barcode','chain','CDR3_nt']).groupby(['barcode','chain']).agg(
        n_junctions=('CDR3_nt','size'), max_raw_distinct_UMIs=('raw_exact_distinct_UMIs','max'),
        max_Q20_distinct_UMIs=('raw_Q20_distinct_UMIs','max')).reset_index()
    save(qc,table/(prefix+'_focal_low_support_calls.csv'))
    control = cells[cells.receptor_search_paired_ab_annotation][['cell_id','reconstruction_selected_spots',
        'receptor_search_CDR3a','receptor_search_CDR3b','reconstruction_matches_processed_TRA',
        'reconstruction_matches_processed_TRB','reconstruction_paired_ab_2_Q20_UMIs']]
    save(control,table/(prefix+'_processed_control_recovery.csv'))
    trust = check/'TRUST4'
    commit = subprocess.check_output(['git','-C',str(trust),'rev-parse','HEAD'],text=True).strip()
    archive = check/(prefix+'_TRUST4_source.tar.gz')
    if not archive.exists():
        subprocess.run(['git','-C',str(trust),'archive','--format=tar.gz','-o',str(archive),'HEAD'],check=True)
    extraction = json.loads((table/(prefix+'_extraction_manifest.json')).read_text())
    report=f'''# GSE178991 V8 single-run receptor reconstruction and model choice

## Recommendation

Use ungated optimized_M05 as the main candidate-discovery model for the stated goal of retaining gamma-delta T cells. Keep the frozen T/NK gate scores and T/NK/abstain states as review information, without automatically removing candidates. This is a task-specific recommendation, not proof that ungated LR has universally better accuracy. No deployed model, threshold, existing atlas label or putative overlay was changed. CE-M05 was not the intervention tested by this reconstruction.

## Complete run, cohort and inputs

SRR14925077 is one complete V8_TEL VDJ run, not the entire four-run V8_TEL library. Its 2,474,400,807-byte native SRA file passed official MD5 8b2d8b71a0a4a86e1507522ac4f25b19. All 25,057,514 spots were scanned; native I1/R1/R2 and original Phred+33 qualities were retained. Exact 16-base R1 barcodes selected the unchanged 98 V8 PANGEA atlas cells, including six focal source-NK candidates with 0.5 <= optimized_M05 score < 0.9394744727488467 and a gate NK call. The other 35 GSE178991 focal candidates are not in V8 and were not reconstructed here.

Thirteen atomically completed extraction units retained 150,976 paired spots. The six focal cells contribute 5,801 spots. Exact barcode matching may miss reads with barcode errors; no correction was applied. R1 positions 16:26 were used as the 10-base UMI, and the 13-base TSO at 26:39 matched TTTCTTATATGGG in {extraction['TSO_exact_fraction']:.2%} of selected reads. Assembly uses R1[39:] and the full R2, while raw original R1/R2 and qualities are preserved. These are sequencing records, not counts of independent RNA molecules.

{markdown(focal[['barcode','reconstruction_selected_spots','p_gdT_given_T','p_T']])}

## Reconstruction and safeguards

TRUST4 source revision: {commit}. The saved human_IMGT+C.fa reference includes alpha, beta, gamma and delta V/J sequences. Each cell is assembled separately by its barcode within the combined run. Both the standard output and a second annotation with --noImpute are retained. Evidence evaluation uses all non-imputed CDR3 calls, including secondary contigs, rather than only TRUST4's selected two-chain barcode report. The default barcode report may impute sequences; it is not treated as validated identity evidence.

Every reported junction is checked against the original selected read pairs. Evidence tables retain exact read matches, distinct raw UMI sequences, and support where barcode, UMI and full junction bases are all Q20 or better. The stricter review rule requires a non-imputed, in-frame CDR3 with V/J calls, no stop or unknown amino acids and at least two such distinct UMI sequences per chain. This is a conservative audit rule, not an externally calibrated diagnostic cutoff. UMI error correction was not performed. Full-variable-region assembly status is retained separately from complete CDR3 status. Shared junctions across barcodes are flagged; this alone cannot distinguish a true clone from contamination. Mixed chains require review.

## Results

{markdown(summary)}

All seven original paired alpha-beta control cells recover the exact previously annotated beta CDR3, and six recover the exact previously annotated alpha CDR3, with at least one Q20-supported raw UMI for each recovered junction. Three of these seven controls meet the stricter two-UMI-per-chain pairing rule. Alpha-beta control recovery validates technical linkage and reconstruction, but does not validate gamma-delta detection sensitivity in this enrichment protocol.

None of the six focal candidates has a supported gamma-delta pair. Their alpha/beta-like calls each have at most one distinct raw UMI even before Q20 filtering; therefore the negative gamma-delta result is not being created solely by the two-UMI quality rule. The sparse alpha/beta calls are retained as low-support conflicting information, not sufficient for relabelling the cells alpha-beta T or NK. Ambient sequences, barcode cross-talk, multiplets and low coverage remain possible explanations. No candidate is assigned a new identity.

No TRG/TRD CDR3 pair was recovered anywhere in these 98 atlas cells. One non-focal short contig has a TRDV1 V assignment together with TRAJ27; the alpha J assignment does not establish a delta chain. A V-gene name alone is not gamma-delta receptor evidence. This run has no established gamma-delta-positive control, so absent gamma/delta assembly cannot demonstrate true NK identity or validate the gate's exclusions. Other sequencing runs and GEX reads were not reconstructed in this bounded analysis.

## Why the recommendation favors ungated LR for discovery

The comparison below reuses frozen predictions at the original optimized_M05 threshold on identical cells within each dataset; the new V8 reconstruction is not silently substituted into these old reference cohorts.

{markdown(comparisons)}

In the existing SPL receptor-reference cohort, ungated optimized_M05 detects 227/378 gamma-delta-supported cells (60.05%), whereas hard gate plus optimized_M05 detects 205/378 (54.23%). The additional loss is 22 cells (9.69% of those originally detected): eight are gate NK calls and fourteen are abstentions. The alpha-beta-reference calls decrease from one to zero. This measures the trade-off in the available receptor-reference cohort, not NK-specific false-positive performance; alpha-beta reference cells are not an independently validated NK negative set. Processed receptor evidence and possible multiplets retain the earlier quality limitations.

PANGEA receptor coverage is incomplete and selected, and 15-organ has no gamma-delta receptor reference in this evaluation. Author NK annotations do not independently validate true NK specificity. The V8 candidates remain unresolved after the present run, so their removal cannot be counted as demonstrated false-positive correction. These limitations prevent a claim that hard gating improves overall atlas accuracy.

For recall-oriented discovery, retain optimized_M05 candidates with gate status attached. Use gated results as a separate conservative/sensitivity analysis rather than a replacement for the main candidate set. This recommendation does not prove that every retained NK-like candidate is gamma-delta T, and does not make 0.5 a validated deployment threshold. Any high-recall operating threshold must be selected inside donor-separated training/development data and checked on a new independent evaluation cohort.

## Reproducibility and restart

Python: {platform.python_version()}; pandas: {pd.__version__}; SRA Toolkit 3.4.1. TRUST4 and reference hashes, commands and input hashes are in the manifests. Build used existing zlib headers/libraries from /data/conda_envs/kaelynwen/ENTER, without a system-package installation. Official TRUST4 source is archived; build command inside its source directory:

`make -j 4 trust4 fastq-extractor annotator CXXFLAGS="-O3 -Wall -g -I/data/conda_envs/kaelynwen/ENTER/include" LINKFLAGS="-L/data/conda_envs/kaelynwen/ENTER/lib -Wl,-rpath,/data/conda_envs/kaelynwen/ENTER/lib -lpthread -lz"`

Authoritative source: build_21_gate_LRM05_evaluation_15organs.py. The added entry points are callable separately; the prior notebook Run All was not repeated. On Woody, restart with:

`cd /data/users/kaelynwen/gdT_project/gdt_lr_12model && /data/conda_envs/kaelynwen/ENTER/bin/python -c "from pathlib import Path; import runpy; m=runpy.run_path('build_21_gate_LRM05_evaluation_15organs.py'); p=Path.cwd(); m['run_gate_discordance_audit_GSE178991_V8_reconstruction'](p); m['assess_gate_discordance_audit_GSE178991_V8_reconstruction'](p); m['report_gate_discordance_audit_GSE178991_V8_reconstruction'](p)"`

The native download can resume from SRR14925077.part. Each extraction unit verifies input/configuration/tool/cohort signatures and output hashes before reuse; aggregation requires all 13 units. Assembly and non-imputed annotation have separate completion receipts. Read-level evidence is recalculated from validated extracted FASTQ inputs. The native 2.47 GB SRA is retained on Woody; local delivery contains the selected reads, assembly outputs, evidence, code archive, report and manifests. The output manifest explicitly records intended absolute local/Woody paths and local-copy scope. No manuscript or Overleaf file was changed.

Method references: https://github.com/liulab-dfci/TRUST4 and https://doi.org/10.1038/s41592-021-01142-2 . Raw-run provenance: https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/run_new?acc=SRR14925077 .
'''
    save(report,table/(prefix+'_reconstruction.md'))
    paths = sorted(set(list(check.glob(prefix+'*'))+list(table.glob(prefix+'*'))+[project/'build_21_gate_LRM05_evaluation_15organs.py',check/'SRR14925077']))
    local=Path('/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/gdt_lr_12model');remote=Path('/data/users/kaelynwen/gdT_project/gdt_lr_12model')
    rows=[]
    for p in paths:
        if not p.is_file() or p.name.endswith('.tmp') or p.name.endswith('_reconstruction_output_manifest.csv'):continue
        rel=p.relative_to(project)
        rows.append({'relative_path':str(rel),'local_path':str(local/rel),'Woody_path':str(remote/rel),
            'bytes':p.stat().st_size,'sha256':sha(p),'local_copy_expected':p.name!='SRR14925077'})
    save(pd.DataFrame(rows),table/(prefix+'_reconstruction_output_manifest.csv'))
    print('Report and',len(rows),'manifested files completed',flush=True)
    return str(table/(prefix+'_reconstruction.md'))


def build():
    cells=[]
    for part in (CURRENT_CANDIDATES+RUNTIME+DISCORDANCE+RECEPTOR_AUDIT+SOURCE_AUDIT+P8_AUDIT+ATLAS_AUDIT+THRESHOLD_AUDIT+THRESHOLD_UMAP).split('# %% ')[1:]:
        kind,source=part.split('\n',1)
        cells.append((nbformat.v4.new_markdown_cell if kind=='markdown' else nbformat.v4.new_code_cell)(source.strip()))
    notebook=nbformat.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'authoritative_source':Path(__file__).name})
    path=Path(__file__).resolve().parent/'notebooks'/f'{NAME}.ipynb'
    if path.exists():
        previous=nbformat.read(path,as_version=4)
        old={(cell.cell_type,cell.source):cell for cell in previous.cells}
        for cell in notebook.cells:
            prior=old.get((cell.cell_type,cell.source))
            if prior is not None:
                cell['id']=prior['id']
            if prior is not None and cell.cell_type=='code':
                cell['outputs']=prior.get('outputs',[])
                cell['execution_count']=prior.get('execution_count')
                cell['metadata']=prior.get('metadata',{})
        notebook.metadata['execution_note']='Unchanged evaluation cells retain prior outputs; the self-contained gate-discordance section can run independently from verified saved inputs.'
    nbformat.write(notebook,path)
    print(path)

if __name__=='__main__': build()
