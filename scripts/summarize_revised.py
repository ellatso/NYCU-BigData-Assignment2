"""Create slide-ready evidence only from a real revised run, with scope checks."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from asg2.provenance import verify_data

def run(a):
    out=Path(a.out);meta=verify_data(a.data)
    design=json.loads((out/'design.json').read_text())
    if design['scope']!='observed_official_gharchive':raise ValueError('Synthetic test scores cannot become slide evidence')
    df=pd.read_csv(out/'metrics.csv');pooled=df[df.split.eq('pooled_observed_development')].copy()
    cal=df[df.split.eq('calibration_mean')][['model','score']].rename(columns={'score':'calibration_score'})
    combined=pooled.merge(cal,on='model',how='left')
    combined.to_csv(out/'slide_evidence.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4.7));ax.barh(combined.model,combined.score*100,color='#344394')
    ax.set_xlabel('Observed development Score (%)');ax.invert_yaxis();ax.set_xlim(0,100)
    ax.set_title('Matched data and dates; F1 has zero grade weight')
    fig.tight_layout();fig.savefig(out/'ablation.png',dpi=180);plt.close(fig)
    lines=['# Revised measured results','', 'Source: https://www.gharchive.org/','',
           'Official builder commit: '+meta['official_commit'],
           'Data period: '+' to '.join(design['source_period_utc']),
           '', 'These are observed historical outcomes, not official test accuracy or confirmed complete GitHub activity.','',
           combined[['model','n_features','calibration_score','macro_f1','micro_f1','score']].to_csv(index=False),
           '', 'See coverage.csv, time_diagnostics.json and reproduction_checks.json (if the independent check was executed).']
    (out/'MEASURED_RESULTS.md').write_text('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',default='artifacts_revised');run(p.parse_args())
