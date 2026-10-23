"""Q1: every declared FK, plus the actual two-user as-of entity graph."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from graphviz import Digraph
from asg2.core import TABLES,PK,FKS,TIME,COLS,stamp

def main(data,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    g=Digraph('schema',format='svg');g.attr(rankdir='LR',fontname='Arial');g.attr('node',shape='plain',fontname='Arial',fontsize='11')
    for name in TABLES:
        fields=[(PK[name], f'{PK[name]} (PK)')]
        fields += [(c, f'{c} (FK)') for c in FKS[name]]
        if name=='Users':fields += [('stable', 'github_user_id (stable ID)')]
        label='<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">'
        label+=f'<TR><TD BGCOLOR="#d8e7f0"><B>{name}</B></TD></TR>'
        for port,text in fields:label+=f'<TR><TD PORT="{port}" ALIGN="LEFT">{text}</TD></TR>'
        label+='</TABLE>>'
        g.node(name,label)
    for table,cols in FKS.items():
        for column,target in cols.items():g.edge(table+':'+column,target+':'+PK[target],label=column,fontsize='8')
    g.save(str(out/'schema.dot'))
    try:g.render(str(out/'schema'),cleanup=True)
    except Exception as e:print('DOT saved; SVG needs the Graphviz dot executable:',e)
    base=Path(data)/'data/db'
    ts={}
    end=stamp('2026-09-17T00:00:00Z');start=stamp('2026-09-15T00:00:00Z')
    for table in TABLES:
        columns=list(COLS[table])
        if table=='PullRequests':columns+=['merged_by_id','merged_at']
        d=pd.read_parquet(base/f'{table}.parquet',columns=columns)
        d=d[d[TIME[table]]<end].copy()
        if table=='PullRequests':d.loc[d.merged_at.isna()|(d.merged_at>=end),'merged_by_id']=pd.NA
        ts[table]=d.set_index(PK[table],drop=False)
    users=ts['Users'];seeds=set(map(int,users.loc[users.github_user_id.isin([22996444,28983449]),'user_id']))
    if len(seeds)!=2:raise ValueError('Required users missing from this snapshot')
    chosen={("Users",u) for u in seeds};in_window=set()
    actors={'PullRequests':'author_id','Issues':'author_id','IssueComments':'author_id','PRReviews':'reviewer_id','PRReviewComments':'author_id','Pushes':'user_id'}
    for name,actor in actors.items():
        d=ts[name];sel=d[(d[TIME[name]]>=start)&d[actor].isin(seeds)]
        for i in sel.index:chosen.add((name,int(i)));in_window.add((name,int(i)))
    # Include FK endpoints recursively; never add unrelated events in shared repos.
    todo=list(chosen)
    while todo:
        table,i=todo.pop();row=ts[table].loc[i]
        for c,target in FKS[table].items():
            if c not in row or pd.isna(row[c]):continue
            j=int(row[c]);key=(target,j)
            if j in ts[target].index and key not in chosen:
                chosen.add(key);todo.append(key)
    h=Digraph('entities',format='svg');h.attr(rankdir='TB',fontname='Arial',label='2026-09-15 00:00 to 09-17 00:00 UTC\nColored activity rows are in the window; gray nodes are necessary FK context.',labelloc='t');h.attr('node',shape='box',fontname='Arial',fontsize='10')
    for table,i in sorted(chosen):
        row=ts[table].loc[i];label=f'{table} {i}'
        if table=='Users':label+=f'\nGitHub ID {int(row.github_user_id)}'
        key=f'{table}__{i}';fill='#b5d8ed' if (table,i) in in_window else '#eeeeee'
        if table=='Users' and i in seeds:fill='#f5d992'
        h.node(key,label,style='filled',fillcolor=fill)
    edges=0
    for table,i in sorted(chosen):
        row=ts[table].loc[i]
        for c,target in FKS[table].items():
            if c not in row or pd.isna(row[c]):continue
            j=int(row[c])
            if (target,j) in chosen:h.edge(f'{table}__{i}',f'{target}__{j}',label=c,fontsize='8');edges+=1
    h.save(str(out/'entity_graph.dot'))
    try:h.render(str(out/'entity_graph'),cleanup=True)
    except Exception as e:print('DOT saved; SVG needs Graphviz:',e)
    (out/'graph_manifest.json').write_text(json.dumps({'schema_tables':8,'schema_fk_edges':sum(map(len,FKS.values())),'entity_nodes':len(chosen),'entity_edges':edges,'activity_rows_in_window':len(in_window),'start_utc':str(start),'end_utc_exclusive':str(end),'seed_github_ids':[22996444,28983449],'context_policy':'necessary foreign-key endpoints may predate the selected activity window; no future rows or unrelated repo events'},indent=2))
    print('Graph manifest:',(out/'graph_manifest.json').read_text())

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',default='artifacts/graphs');a=p.parse_args();main(a.data,a.out)
