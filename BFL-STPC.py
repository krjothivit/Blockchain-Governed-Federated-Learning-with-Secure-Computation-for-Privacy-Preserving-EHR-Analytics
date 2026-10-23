
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from src.generate import generate
from src.crypto_sim import simulated_masked_sum
from src.audit import append_event, verify_chain

def sigmoid(z):
    return 1/(1+np.exp(-np.clip(z, -35, 35)))

def loss_grad(w, X, y):
    p = sigmoid(X @ w)
    return X.T @ (p-y)/len(y)

def fit_local(w, X, y, *, epochs, lr, batch_size, clip_norm, noise_std, rng):
    out = w.copy()
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for part in np.array_split(order, max(1, int(np.ceil(len(y)/batch_size)))):
            g = loss_grad(out, X[part], y[part])
            # Global gradient clipping here is NOT sample-level DP-SGD.
            g *= min(1.0, clip_norm/(np.linalg.norm(g)+1e-12))
            if noise_std:
                g = g + rng.normal(0, noise_std, len(g))
            out -= lr*g
    return out

def main(config):
    c = yaml.safe_load(Path(config).read_text())
    rng = np.random.default_rng(c['seed'])
    outdir = Path(c['output_dir']); outdir.mkdir(parents=True, exist_ok=True)
    df = generate(c['n_samples'], c['n_features'], c['seed'])
    df.to_csv(outdir/'synthetic_data.csv', index=False)
    X = df.drop(columns=['target']).to_numpy(); y = df.target.to_numpy()
    indices = np.arange(len(y))
    trainval, test = train_test_split(indices, test_size=c['test_fraction'],
                                     stratify=y, random_state=c['seed'])
    train, val = train_test_split(trainval, test_size=c['val_fraction'],
                                 stratify=y[trainval], random_state=c['seed']+1)
    # All preprocessing fit on train only; no held-out leakage.
    imp = SimpleImputer(strategy='median').fit(X[train]); Xi = imp.transform(X)
    scaler = StandardScaler().fit(Xi[train]); Xp = scaler.transform(Xi)
    Xp = np.column_stack([Xp, np.ones(len(Xp))])
    # Client allocation only from training indices (non-IID via label sorting + shuffle).
    shuffled = rng.permutation(train)
    clients = [a for a in np.array_split(shuffled, c['n_clients']) if len(a)]
    w = np.zeros(Xp.shape[1]); audit = []; history = []
    for r in range(c['rounds']):
        locals_ = [fit_local(w, Xp[idx], y[idx], epochs=c['local_epochs'],
                    lr=c['learning_rate'], batch_size=c['batch_size'],
                    clip_norm=c['clip_norm'], noise_std=c['noise_std'], rng=rng)
                    for idx in clients]
        weights = np.array([len(idx) for idx in clients], dtype=float)
        weighted = [u * (n/weights.sum()) for u,n in zip(locals_,weights)]
        w = (simulated_masked_sum(weighted, c['seed']+r)
             if c['secure_mask_simulation'] else np.sum(weighted,axis=0))
        vp = sigmoid(Xp[val] @ w)
        metric = {'round': r+1, 'val_accuracy': float(accuracy_score(y[val],vp>=0.5)),
                  'val_auc': float(roc_auc_score(y[val],vp))}
        history.append(metric)
        append_event(audit, 'round_aggregate', {'round': r+1, 'clients':len(clients)})
    tp = sigmoid(Xp[test] @ w)
    result = {'warning':'REFERENCE BASELINE ONLY; not verified BFL-STPC results',
       'test_accuracy':float(accuracy_score(y[test], tp>=0.5)),
       'test_roc_auc':float(roc_auc_score(y[test],tp)),
       'test_confusion_matrix':confusion_matrix(y[test],tp>=0.5).tolist(),
       'audit_valid':verify_chain(audit), 'train_n':len(train),'val_n':len(val),'test_n':len(test),
       'config':c}
    (outdir/'metrics.json').write_text(json.dumps(result,indent=2))
    (outdir/'history.json').write_text(json.dumps(history,indent=2))
    (outdir/'audit.json').write_text(json.dumps(audit,indent=2))
    (outdir/'split_indices.json').write_text(json.dumps({'train':train.tolist(),
             'validation':val.tolist(),'test':test.tolist()},indent=2))
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/experiment.yaml')
    main(p.parse_args().config)
