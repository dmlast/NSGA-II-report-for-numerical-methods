"""Generate visual comparisons between a simple NSGA-style baseline and NSGA-II."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.nsga2_core import (
    NSGAConfig, assign_rank_and_distance, fast_non_dominated_sort, get_first_front,
    nsga2, zdt1, zdt2, zdt3, true_front_zdt1, true_front_zdt2, true_front_zdt3,
    convergence_metric, diversity_metric, make_offspring
)

ASSETS = ROOT / 'assets'


def evaluate(objectives, X):
    return np.column_stack([f(X) for f in objectives])


def crowding_like_selection(F, fronts, N):
    # Same last-front logic as NSGA-II for plotting stable baseline, but without elitist parent+child merge.
    ranks, dist, fronts2 = assign_rank_and_distance(F)
    idx=[]
    for front in fronts2:
        if len(idx)+len(front)<=N:
            idx.extend(front)
        else:
            idx.extend(sorted(front, key=lambda i: -dist[i])[:N-len(idx)])
            break
    return np.array(idx), ranks, dist, fronts2


def nsga1_simple(objectives, bounds, cfg):
    """A compact NSGA-1994-like baseline: rank-based selection, no elitist union."""
    rng=np.random.default_rng(cfg.seed)
    N=cfg.pop_size
    bounds=np.asarray(bounds,float)
    pop=rng.uniform(bounds[:,0], bounds[:,1], size=(N,bounds.shape[0]))
    F=evaluate(objectives, pop)
    ranks, distances, fronts=assign_rank_and_distance(F)
    hist={'objectives':[F.copy()], 'fronts':[fronts]}
    for _ in range(cfg.n_gen):
        # offspring from current population; no P∪Q elitist survival step
        Q=make_offspring(pop, ranks, distances, bounds, cfg, rng)
        FQ=evaluate(objectives, Q)
        # old NSGA did not guarantee elites; keep the new generation only
        pop, F = Q, FQ
        ranks, distances, fronts=assign_rank_and_distance(F)
        hist['objectives'].append(F.copy()); hist['fronts'].append(fronts)
    return hist


def plot_compare(name, objs_bounds, true_front, fname):
    objs,bounds=objs_bounds
    cfg=NSGAConfig(pop_size=100,n_gen=200,seed=9)
    h1=nsga1_simple(objs,bounds,cfg)
    h2=nsga2(objs,bounds,cfg)
    f1=get_first_front(h1,-1)
    f2=get_first_front(h2,-1)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),sharex=False,sharey=False)
    for ax,front,title,color in [(axes[0],f1,'NSGA (без элитизма)', '#d98e32'), (axes[1],f2,'NSGA-II', '#2f75b5')]:
        ax.plot(true_front[:,0], true_front[:,1], lw=2, color='#9a9a9a', label='истинный фронт')
        ax.scatter(front[:,0], front[:,1], s=20, alpha=.85, color=color, label='найденный фронт')
        ax.set_title(title)
        ax.set_xlabel('$f_1$'); ax.set_ylabel('$f_2$')
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle(f'{name}: одинаковый бюджет, N=100, T=200', y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS/fname, bbox_inches='tight')
    plt.close(fig)
    return h1,h2


def plot_metrics(results):
    names=list(results.keys())
    x=np.arange(len(names)); w=.36
    conv_nsga=[]; conv_n2=[]; div_nsga=[]; div_n2=[]
    for name,(h1,h2,tf) in results.items():
        conv_nsga.append(convergence_metric(get_first_front(h1,-1),tf))
        conv_n2.append(convergence_metric(get_first_front(h2,-1),tf))
        div_nsga.append(diversity_metric(get_first_front(h1,-1)))
        div_n2.append(diversity_metric(get_first_front(h2,-1)))
    fig,ax=plt.subplots(figsize=(8.5,3.6))
    ax.bar(x-w/2, conv_nsga, w, label='NSGA')
    ax.bar(x+w/2, conv_n2, w, label='NSGA-II')
    ax.set_xticks(x, names); ax.set_ylabel('$\\Upsilon$ (меньше лучше)')
    ax.set_title('Близость к истинному фронту')
    ax.grid(axis='y', alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(ASSETS/'compare_metrics.png', bbox_inches='tight'); plt.close(fig)


def main():
    results={}
    for name,ob,tf,fname in [
        ('ZDT1', zdt1(30), true_front_zdt1(500), 'compare_zdt1.png'),
        ('ZDT3', zdt3(30), true_front_zdt3(500), 'compare_zdt3.png'),
        ('ZDT2', zdt2(30), true_front_zdt2(500), 'compare_zdt2.png'),
    ]:
        h1,h2=plot_compare(name,ob,tf,fname)
        results[name]=(h1,h2,tf)
    plot_metrics(results)
    print('comparison assets written')

if __name__=='__main__':
    main()
