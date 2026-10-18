"""Independent direct calls to pinned author factories/notebook operations.

Protocol adaptations are F-only float32 constant dropping and typed K+1 unknowns.
No adapter model, loss, preprocessing, selection or prediction helper is used.
"""
import copy
import hashlib

import numpy as np


def direct(F, yF, G, yG, Q, config, seed):
    import torch
    import tabm
    import rtdl_num_embeddings as numemb
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.metrics import mean_squared_error
    schema = config.feature_schema
    names = [n for n in schema['columns'] if n not in schema['categorical']]
    raw = F[names].to_numpy(dtype=np.float32)
    keep = ~(raw == raw[0]).all(0)
    dropped = {name: float(raw[0, i]) for i, name in enumerate(names) if not keep[i]}
    numeric = np.ascontiguousarray(raw[:, keep])
    noise = np.random.default_rng(0).normal(0., 1e-5, numeric.shape).astype(numeric.dtype)
    qt = QuantileTransformer(n_quantiles=max(min(len(F)//30, 1000), 10),
        output_distribution='normal', subsample=10**9).fit(numeric+noise)
    Fnum, Gnum, Qnum = [qt.transform(np.ascontiguousarray(x[names].to_numpy(dtype=np.float32)[:, keep]))
        for x in (F, G, Q)]
    vocab, catnames = [], [n for n in schema['columns'] if n in schema['categorical']]
    # Fixtures intentionally use strings; no adapter category helper defines this reference.
    for name in catnames:
        vocab.append(sorted({'str:'+x for x in F[name]}))
    def categories(frame):
        return np.array([[next((i for i, x in enumerate(v) if x == 'str:'+frame.iloc[j][name]), len(v))
            for name, v in zip(catnames, vocab)] for j in range(len(frame))], np.int64).reshape(len(frame), len(vocab))
    Fcat, Gcat, Qcat = [categories(x) for x in (F, G, Q)]
    yF, yG = np.asarray(yF, np.float32), np.asarray(yG, np.float32)
    mu, scale = yF.mean().item(), yF.std().item()
    constant = bool((yF == yF[0]).all())
    if constant: scale = 1.
    train_target = torch.from_numpy((yF-mu)/scale)
    p = config.resolved()
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        embedding = None if config.variant == 'plain' else numemb.PiecewiseLinearEmbeddings(
            numemb.compute_bins(torch.from_numpy(Fnum), n_bins=p['n_bins']),
            d_embedding=p['d_embedding'], activation=False, version='B')
        model = tabm.TabM.make(n_num_features=Fnum.shape[1], cat_cardinalities=[len(v)+1 for v in vocab],
            d_out=1, num_embeddings=embedding, n_blocks=p['n_blocks'], d_block=p['d_block'],
            dropout=p['dropout'], activation='ReLU', k=32, arch_type='tabm',
            start_scaling_init=p['start_scaling_init'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
        def apply(num, cat, rows):
            return model(torch.from_numpy(num)[rows], torch.from_numpy(cat)[rows] if cat.shape[1] else None).squeeze(-1).float()
        def predict(num, cat):
            model.eval()
            with torch.no_grad():
                members = torch.cat([apply(num, cat, rows) for rows in torch.arange(len(num)).split(8096)]).numpy()
            return (members*scale+mu).mean(1)
        history, orders, best_weights = [], [], None
        best_score, selected, remaining, updates = -float('inf'), -1, 16, 0
        for epoch in range(config.max_epochs):
            order = torch.randperm(len(F))
            losses, norms = 0., []
            for rows in order.split(256):
                model.train()
                optimizer.zero_grad()
                yp = apply(Fnum, Fcat, rows)
                loss = torch.nn.functional.mse_loss(yp.flatten(0, 1), train_target[rows].repeat_interleave(32))
                loss.backward()
                norms.append(float(torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.)))
                optimizer.step()
                losses += loss.item()*len(rows)
                updates += 1
            score = -(mean_squared_error(yG, predict(Gnum, Gcat))**.5)
            history.append(dict(epoch=epoch, F_online_member_mse=losses/len(F), G_rmse=float(-score),
                applied_updates=updates, row_order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest(),
                gradient_norm_min=min(norms), gradient_norm_max=max(norms)))
            orders.append(order.numpy().copy())
            if score > best_score:
                best_score, selected, remaining = score, epoch, 16
                best_weights = copy.deepcopy(model.state_dict())
            else:
                remaining -= 1
            if remaining < 0: break
        model.load_state_dict(best_weights)
        prediction = predict(Qnum, Qcat)
    features = dict(schema=schema, vocabularies=vocab, numeric_keep_mask=keep.tolist(),
        dropped_numeric_values=dropped, constant_detection_dtype='float32',
        quantiles=qt.quantiles_.tolist(), references=qt.references_.tolist(), fit_rows=len(F),
        quantile_random_state=None, noise_seed=0, noise_std=1e-5, input_dtype='float32',
        noisy_fit_dtype='float32', output_distribution='normal')
    target = dict(mean=mu, scale=scale, constant_float32_F=constant,
        input_dtype='float32', reduction_dtype='float32')
    return prediction, history, np.stack(orders), dict(features=features, target=target)
