import numpy as np
from scipy.spatial.distance import mahalanobis
from numpy.linalg import norm
import pandas as pd

def smape_idao(pred, ans):
    pred = pred.values
    ans = ans.values
    return np.mean(np.abs(pred - ans) / (np.abs(pred) + np.abs(ans)))

def idao_score(pred, ans):
    return 100 * (1 - smape_idao(pred, ans))

def smape_new_vector_norm(pred, ans, av_score=True):
    assert set(ans.columns) == {'y', 'Vy', 'x', 'z', 'Vx', 'sat_id', 'Vz'}
    for c in ["x", "y", "z", "Vx", "Vy", "Vz"]:
        assert c in pred.columns
    assert (ans.index == pred.index).all()
    scores = []
    if not av_score:
        r_losses = v_losses = []
    for sat_id in pd.unique(ans["sat_id"]):
        idxs = (ans["sat_id"] == sat_id).values
        p = pred.loc[idxs, ["x", "y", "z", "Vx", "Vy", "Vz"]].values.astype("float")
        a = ans.loc[idxs, ["x", "y", "z", "Vx", "Vy", "Vz"]].values.astype("float")
        loss = p - a
        # coordinates
        r_loss = norm(loss[:, :3], axis=1) / (norm(p[:, :3], axis=1) + norm(a[:, :3], axis=1))
        r_loss = np.mean(r_loss)
        # velocities
        v_loss = norm(loss[:, 3:], axis=1) / (norm(p[:, 3:], axis=1) + norm(a[:, 3:], axis=1))     
        v_loss = np.mean(v_loss)
        # score
        scores.append(r_loss + v_loss)
        if not av_score:
            r_losses.append(r_loss)            
            v_losses.append(v_loss)
    if av_score:
        return np.mean(scores)
    else:
        return np.array(scores), np.array(r_losses), np.array(v_losses)

def mahalanobis_distance(pred, ans):
    # TODO: не ясно откуда берется ковариационная матрица
    scores = []
    for sat_id in pd.unique(ans["sat_id"]):
        idxs = (ans["sat_id"] == sat_id).values
        p = pred.loc[idxs, ["x", "y", "z", "Vx", "Vy", "Vz"]]
        a = ans.loc[idxs, ["x", "y", "z", "Vx", "Vy", "Vz"]]
        # coordinates
        r_loss = []
        for coord in ['x', 'y', 'z']:
            p_coord = p[coord].values.astype("float")
            a_coord = a[coord].values.astype("float")
            cov = np.cov(np.stack([p_coord, a_coord], axis=1))
            r_loss.append(mahalanobis(p_coord, a_coord, cov))
        # score
        scores.append(r_loss)
    return np.mean(scores)