"""Estimate the four joint outcomes at the current and final decisions."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from candidate_paths import prepare
from search import CACHE, OUT, TIMES

rows = []
for path in sorted(CACHE.glob('*.npz')):
    s = prepare(path)
    train = s['split'] == 'train'
    for number, name in [(327, '현재 득표 차와 상위 두 후보의 정답 확률로 16회 예상 이득 추정'),
                         (328, '현재 득표 차와 상위 두 후보의 정답 확률 및 최근 변화로 16회 예상 이득 추정'),
                         (329, '직전 제출 후보와 기존 경쟁 후보 및 새 답의 득표 변화로 예상 이득 추정'),
                         (330, '후보별 득표 변화와 새 답을 낸 풀이의 확신도로 예상 이득 추정')]:
        stop = np.ones_like(s['hit'], dtype=bool)
        for j, t in enumerate(TIMES[:-1]):
            features = [s['margin'][:, j], s['p'][:, j], s['gap'][:, j]]
            if number == 328:
                features.extend([s[k][:, j] for k in ('leader_change', 'runner_change')])
            if number in (329,330):
                features.extend(s[k][:,j] for k in ('previous_change','entrant_share','rival_gain'))
            if number == 330:
                features.append(s['entrant_conf'][:,j])
            x = np.column_stack(features)
            scale = StandardScaler().fit(x[train])
            x = scale.transform(x)
            y = 2*s['hit'][:, j].astype(int) + s['hit'][:, -1].astype(int)
            # One pseudo-observation per outcome keeps rare classes estimable.
            model = LogisticRegression(C=.25,solver='saga',random_state=327,max_iter=20000).fit(
                np.vstack([x[train], np.zeros((4, x.shape[1]))]), np.r_[y[train], np.arange(4)])
            prob = model.predict_proba(x)
            assert np.isfinite(prob).all() and np.allclose(prob.sum(axis=1), 1)
            stop[:, j] = s['valid'][:, j] & (prob[:, 1] <= prob[:, 2])
        end = stop.argmax(axis=1)
        ix = np.arange(len(end))
        for split in ('dev', 'test'):
            mask = s['split'] == split
            h = s['hit'][ix, end][mask]
            tokens = s['tok'][ix, end][mask]
            full = s['tok'][mask, -1]
            rows.append(dict(split=split, cell=path.stem, id=number, kind=name, correct=h.sum(),
                             total=len(h), accuracy=h.mean(), solutions=TIMES[end[mask]].mean(),
                             tokens=tokens.mean(), full_tokens=full.mean(), token_ratio=tokens.sum()/full.sum()))
data = pd.DataFrame(rows)
data.to_csv(OUT/'candidate_gain_cells.csv',index=False)
print(data.groupby(['id','split'])[['accuracy','solutions']].mean())
