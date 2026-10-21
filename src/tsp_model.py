"""
tsp_model - AlphaFold-inspired sparse pair model ("PairFormer-lite") that predicts
LKH candidate edges and node penalties (pi) for Euclidean TSP.

Representation (as in AlphaFold's Evoformer):
  * single (node) representation  h_i            (N, dn)
  * pair (edge) representation    z_ij on kNN    (N, K, de), directed edges i -> NB[i, a]
Block = pair-biased node attention -> node transition -> node->edge product update ->
        reverse-edge mixing -> triangle multiplicative update (outgoing, incoming) ->
        [optional triangle attention] -> edge transition.
Triangles are the directed kNN triangles {i->j, i->k, j->k}; the same list serves the
"outgoing" (z_ij += sum_k a_ik * b_jk) and "incoming" (z_jk += sum_i a_ij * b_ik) updates.
All input features are invariant to translation, rotation, scale and node order.
Heads: edge logits (top-k -> CANDIDATE_FILE) and scale-equivariant node penalties
pi_i = s_i * f_i (-> PI_FILE). Optional AlphaFold-style recycling.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

PRECISION = 100  # LKH default


# ------------------------------------------------------------------ graph + features
def knn_np(xy, K):
    from scipy.spatial import cKDTree
    n = len(xy)
    d, idx = cKDTree(xy).query(xy, k=K + 1)
    self_mask = idx == np.arange(n)[:, None]
    no_self = ~self_mask.any(1)
    self_mask[no_self, -1] = True
    keep = ~self_mask
    return idx[keep].reshape(n, K), d[keep].reshape(n, K)


def extend_graph(xyn, NB, ens_succ, extra):
    """TourFold v2 graph: kNN-K plus up to `extra` neighbours that the tour ensemble uses but kNN-K lacks
    (long edges, e.g. between clusters). They replace the farthest kNN slots, so every row still has K entries;
    the most frequently used ensemble edges come first. Rows are re-sorted by distance.
    Returns NB (n,K), D (n,K) and a flag (n,K) marking the ensemble-only edges."""
    n, K = NB.shape
    succ = np.asarray(ens_succ, dtype=np.int64)
    Mt = succ.shape[1]
    pred = np.empty_like(succ)
    ar = np.arange(n)
    for m in range(Mt):
        pred[succ[:, m], m] = ar
    cand = np.concatenate([succ, pred], 1)                              # (n, 2M) ensemble neighbours
    cand = np.where((cand[:, :, None] == NB[:, None, :]).any(-1), -1, cand)
    S = np.sort(cand, 1)
    first = np.ones_like(S, dtype=bool)
    first[:, 1:] = S[:, 1:] != S[:, :-1]
    valid = (S >= 0) & first
    cnt = (S[:, :, None] == S[:, None, :]).sum(-1)
    dist = np.linalg.norm(xyn[np.maximum(S, 0)] - xyn[:, None, :], axis=-1)
    score = np.where(valid, cnt * 10.0 - dist, -np.inf)                 # frequency first, then distance
    order = np.argsort(-score, 1, kind="stable")[:, :extra]
    chosen = np.take_along_axis(S, order, 1)
    nv = np.isfinite(np.take_along_axis(score, order, 1)).sum(1)       # ensemble-only neighbours used
    pos = np.arange(K)[None, :]
    keep = pos < (K - nv)[:, None]
    tail = np.take_along_axis(chosen, np.clip(pos - (K - nv)[:, None], 0, extra - 1), 1)
    NB2 = np.where(keep, NB, tail)
    D2 = np.linalg.norm(xyn[NB2] - xyn[:, None, :], axis=-1)
    o = np.argsort(D2, 1, kind="stable")
    return (np.take_along_axis(NB2, o, 1), np.take_along_axis(D2, o, 1),
            np.take_along_axis(~keep, o, 1))


def prepare_instance(xy, K, tour=None, pi=None, precision=PRECISION, ens_succ=None, ens_cost=None, extra=0):
    """Normalize coordinates, build kNN graph, labels and pi targets (numpy).
    extra > 0 (TourFold v2): the graph also gets up to `extra` ensemble edges per node (see extend_graph)."""
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    if n <= K:
        raise ValueError(f"n={n} must exceed K={K}")
    mn = xy.min(0)
    span = float((xy.max(0) - mn).max()) or 1.0
    xyn = (xy - mn) / span
    NB, D = knn_np(xyn, K)
    s = np.maximum(D[:, : max(1, K // 2)].mean(1), 1e-9)
    ens_only = None
    if extra and ens_succ is not None:
        NB, D, ens_only = extend_graph(xyn, NB, ens_succ, int(extra))
    out = {"n": n, "span": span, "xy": xyn.astype(np.float32), "NB": NB.astype(np.int64),
           "D": D.astype(np.float32), "s": s.astype(np.float32),
           "s_glob": np.full(n, np.median(s), dtype=np.float32)}
    if ens_only is not None:
        out["ens_only"] = ens_only.astype(np.float32)
    if tour is not None:
        t = np.asarray(tour)
        succ = np.empty(n, np.int64)
        pred = np.empty(n, np.int64)
        succ[t] = np.roll(t, -1)
        pred[t] = np.roll(t, 1)
        out["y"] = (NB == succ[:, None]) | (NB == pred[:, None])
    if pi is not None:
        out["pi_t"] = (np.asarray(pi, dtype=np.float64) / (precision * span * s)).astype(np.float32)
    if ens_succ is not None:
        out.update(ensemble_features(xyn, NB, s, ens_succ, ens_cost))
    return out


def ensemble_features(xyn, NB, s, ens_succ, ens_cost=None):
    """TourFold 'tour alignment' (the MSA analogue): per tour m and node i, how tour m connects i.
    Everything is symmetric in the two tour neighbours, so a reversed tour gives the same features."""
    succ = np.asarray(ens_succ, dtype=np.int64)
    n, M = succ.shape
    K = NB.shape[1]
    pred = np.empty_like(succ)
    ar = np.arange(n)
    for m in range(M):
        pred[succ[:, m], m] = ar
    eq_s = NB[:, :, None] == succ[:, None, :]
    eq_p = NB[:, :, None] == pred[:, None, :]
    pos_s = np.where(eq_s.any(1), eq_s.argmax(1), K)
    pos_p = np.where(eq_p.any(1), eq_p.argmax(1), K)
    ds = np.log(np.linalg.norm(xyn[succ] - xyn[:, None, :], axis=-1) / s[:, None] + 1e-4)
    dp = np.log(np.linalg.norm(xyn[pred] - xyn[:, None, :], axis=-1) / s[:, None] + 1e-4)
    fs = ((succ[:, None, :] == succ[:, :, None]) | (pred[:, None, :] == succ[:, :, None])).mean(-1)
    fp = ((succ[:, None, :] == pred[:, :, None]) | (pred[:, None, :] == pred[:, :, None])).mean(-1)
    cost = np.ones(M) if ens_cost is None else np.asarray(ens_cost, dtype=np.float64)
    q = (cost - cost.min()) / max(cost.mean(), 1e-9) * 100.0
    msa_f = np.stack([np.minimum(ds, dp), np.maximum(ds, dp), np.broadcast_to(q[None, :], (n, M)),
                      np.minimum(fs, fp), np.maximum(fs, fp)], -1)
    hit = eq_s | eq_p
    top = np.argsort(cost, kind="stable")[: max(1, M // 2)]
    ef_ens = np.stack([hit.mean(-1), hit[:, :, top].mean(-1)], -1)
    return {"msa_f": msa_f.transpose(1, 0, 2).astype(np.float32),
            "msa_ps": pos_s.T.astype(np.int64), "msa_pp": pos_p.T.astype(np.int64),
            "ens_succ": succ.T.copy(), "ens_pred": pred.T.copy(),
            "ef_ens": ef_ens.astype(np.float32)}


def collate(items):
    """Concatenate prepared instances into one disjoint graph."""
    off, N = [], 0
    for it in items:
        off.append(N)
        N += it["n"]
    cat = lambda k: np.concatenate([it[k] for it in items], 0)  # noqa: E731
    b = {"xy": torch.from_numpy(cat("xy")), "D": torch.from_numpy(cat("D")),
         "s": torch.from_numpy(cat("s")), "s_glob": torch.from_numpy(cat("s_glob")),
         "NB": torch.from_numpy(np.concatenate([it["NB"] + o for it, o in zip(items, off)], 0)),
         "bidx": torch.from_numpy(np.concatenate([np.full(it["n"], i) for i, it in enumerate(items)])),
         "offsets": torch.tensor(off + [N]), "span": torch.tensor([it["span"] for it in items]),
         "B": len(items)}
    if all("y" in it for it in items):
        b["y"] = torch.from_numpy(cat("y"))
    if all("pi_t" in it for it in items):
        b["pi_t"] = torch.from_numpy(cat("pi_t"))
    if all("ens_only" in it for it in items):
        b["ens_only"] = torch.from_numpy(cat("ens_only"))
    if all("msa_f" in it for it in items):
        Ms = {it["msa_f"].shape[0] for it in items}
        if len(Ms) != 1:
            raise ValueError(f"all instances in a batch need the same ensemble size, got {Ms}")
        b["msa_f"] = torch.from_numpy(np.concatenate([it["msa_f"] for it in items], 1))
        b["msa_ps"] = torch.from_numpy(np.concatenate([it["msa_ps"] for it in items], 1))
        b["msa_pp"] = torch.from_numpy(np.concatenate([it["msa_pp"] for it in items], 1))
        b["ens_succ"] = torch.from_numpy(np.concatenate([it["ens_succ"] + o for it, o in zip(items, off)], 1))
        b["ens_pred"] = torch.from_numpy(np.concatenate([it["ens_pred"] + o for it, o in zip(items, off)], 1))
        b["ef_ens"] = torch.from_numpy(cat("ef_ens"))
    return b


def to_device(b, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


@torch.no_grad()
def graph_context(b, k_tri=None, chunk_nodes=40000):
    """Invariant features, reverse-edge index and directed triangles (all on device).
    k_tri: the third node k of a triangle must be among the first k_tri neighbours of both i and j
    (fewer, more local triangles -> less memory)."""
    xy, NB, D, s, sg = b["xy"], b["NB"], b["D"], b["s"], b["s_glob"]
    N, K = NB.shape
    kt = K if not k_tri else min(int(k_tri), K)
    ar = torch.arange(N, device=NB.device)
    eq = NB[NB] == ar[:, None, None]                     # (N,K,K): NB[NB[i,a], c] == i
    mutual = eq.any(-1)
    rev_pos = eq.float().argmax(-1)
    rank = torch.arange(K, device=NB.device, dtype=xy.dtype).expand(N, K) / K
    rev_rank = torch.where(mutual, rev_pos.to(xy.dtype) / K, torch.full_like(rank, 1.5))
    vec = xy[NB] - xy[:, None, :]
    ang = torch.atan2(vec[..., 1], vec[..., 0])
    rel = ang - ang[:, :1]
    d1 = D / s[:, None]
    ef = torch.stack([d1, D / s[NB], D / sg[:, None], torch.log(d1 + 1e-4), rank, mutual.to(xy.dtype),
                      rev_rank, torch.cos(rel), torch.sin(rel).abs()], -1)
    if "ef_ens" in b:
        ef = torch.cat([ef, b["ef_ens"]], -1)
    if "ens_only" in b:
        ef = torch.cat([ef, b["ens_only"][..., None].to(ef.dtype)], -1)
    cen = xy[NB].mean(1)
    indeg = torch.bincount(NB.reshape(-1), minlength=N).to(xy.dtype) / K
    nf = torch.stack([s / sg, D[:, 0] / s, D[:, -1] / s, (cen - xy).norm(dim=-1) / s, indeg], -1)
    tris = []
    step = max(1, chunk_nodes // max(1, K * kt * kt // 4096))
    for st in range(0, N, step):
        nb = NB[st:st + step]
        M = NB[nb][:, :, None, :kt] == nb[:, None, :kt, None]   # (n,a,b,c): NB[NB[i,a],c] == NB[i,b]
        idx = M.nonzero()
        i = idx[:, 0] + st
        a, bb, c = idx[:, 1], idx[:, 2], idx[:, 3]
        tris.append(torch.stack([i * K + a, i * K + bb, NB[i, a] * K + c]))
    tri = torch.cat(tris, 1) if tris else torch.zeros(3, 0, dtype=torch.long, device=NB.device)
    return {"node_feat": nf, "edge_feat": ef, "mutual": mutual, "rev_pos": rev_pos, "tri": tri}


# ---------------------------------------------------------------------------- modules
class Transition(nn.Module):
    def __init__(self, d, mult=2):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.l1 = nn.Linear(d, d * mult)
        self.l2 = nn.Linear(d * mult, d)

    def forward(self, x):
        return self.l2(F.gelu(self.l1(self.ln(x))))


class NodeAttention(nn.Module):
    """Attention of node i over its K neighbours, biased by the pair representation."""

    def __init__(self, dn, de, heads=4):
        super().__init__()
        self.h, self.dh = heads, dn // heads
        self.ln, self.lnz = nn.LayerNorm(dn), nn.LayerNorm(de)
        self.q, self.kv = nn.Linear(dn, dn), nn.Linear(dn, 2 * dn)
        self.bias = nn.Linear(de, heads)
        self.g, self.o = nn.Linear(dn, dn), nn.Linear(dn, dn)

    def forward(self, h, z, NB):
        N, K = NB.shape
        x = self.ln(h)
        q = self.q(x).view(N, self.h, self.dh)
        k, v = self.kv(x)[NB].split(self.h * self.dh, -1)
        k, v = k.view(N, K, self.h, self.dh), v.view(N, K, self.h, self.dh)
        logit = torch.einsum("nhd,nkhd->nhk", q, k) / math.sqrt(self.dh)
        logit = logit + self.bias(self.lnz(z)).permute(0, 2, 1)
        att = logit.float().softmax(-1).to(v.dtype)
        out = torch.einsum("nhk,nkhd->nhd", att, v).reshape(N, -1)
        return torch.sigmoid(self.g(x)) * self.o(out)


class EdgeFromNodes(nn.Module):
    def __init__(self, dn, de):
        super().__init__()
        self.ln = nn.LayerNorm(dn)
        self.l, self.r, self.o = nn.Linear(dn, de), nn.Linear(dn, de), nn.Linear(de, de)

    def forward(self, h, NB):
        x = self.ln(h)
        return self.o(self.l(x)[:, None, :] * self.r(x)[NB])


class ReverseMix(nn.Module):
    def __init__(self, de):
        super().__init__()
        self.ln, self.l = nn.LayerNorm(de), nn.Linear(de, de)

    def forward(self, z, NB, rev_pos, mutual):
        zr = z[NB, rev_pos]                                    # z_ji for edge i->j
        return self.l(self.ln(zr)) * mutual[..., None].to(z.dtype)


class TriangleMultiplication(nn.Module):
    """AlphaFold Alg. 11/12 on sparse directed kNN triangles."""

    def __init__(self, de, dh, mode):
        super().__init__()
        self.mode = mode
        self.ln = nn.LayerNorm(de)
        self.a, self.ag = nn.Linear(de, dh), nn.Linear(de, dh)
        self.b, self.bg = nn.Linear(de, dh), nn.Linear(de, dh)
        self.g = nn.Linear(de, de)
        self.ln_o, self.o = nn.LayerNorm(dh), nn.Linear(dh, de)

    def forward(self, z, tri):
        N, K, de = z.shape
        x = self.ln(z).reshape(N * K, de)
        a = torch.sigmoid(self.ag(x)) * self.a(x)
        b = torch.sigmoid(self.bg(x)) * self.b(x)
        e_ij, e_ik, e_jk = tri
        tgt, s1, s2 = (e_ij, e_ik, e_jk) if self.mode == "out" else (e_jk, e_ij, e_ik)
        prod = (a[s1] * b[s2]).float()
        m = torch.zeros(N * K, prod.shape[1], device=z.device, dtype=torch.float32).index_add_(0, tgt, prod)
        cnt = torch.bincount(tgt, minlength=N * K).clamp_min(1).float().sqrt()
        m = (m / cnt[:, None]).to(x.dtype)
        out = torch.sigmoid(self.g(x)) * self.o(self.ln_o(m))
        return out.view(N, K, de)


class TriangleAttention(nn.Module):
    """AlphaFold Alg. 13 (starting node) on sparse triangles: edge i->j attends over k with
    keys/values from i->k and a bias from j->k."""

    def __init__(self, de, heads=4):
        super().__init__()
        self.h = heads
        self.dh = max(8, de // heads)
        self.ln = nn.LayerNorm(de)
        self.q, self.k, self.v = (nn.Linear(de, self.h * self.dh) for _ in range(3))
        self.bias, self.g = nn.Linear(de, heads), nn.Linear(de, self.h * self.dh)
        self.o = nn.Linear(self.h * self.dh, de)

    def forward(self, z, tri):
        N, K, de = z.shape
        E = N * K
        x = self.ln(z).reshape(E, de)
        e_ij, e_ik, e_jk = tri
        q = self.q(x)[e_ij].view(-1, self.h, self.dh)
        k = self.k(x)[e_ik].view(-1, self.h, self.dh)
        logit = ((q * k).sum(-1) / math.sqrt(self.dh) + self.bias(x)[e_jk]).float()   # (T,h)
        mx = torch.full((E, self.h), -1e30, device=z.device).scatter_reduce(
            0, e_ij[:, None].expand(-1, self.h), logit, reduce="amax", include_self=True)
        w = torch.exp(logit - mx[e_ij])
        den = torch.zeros(E, self.h, device=z.device).index_add_(0, e_ij, w).clamp_min(1e-9)
        w = w / den[e_ij]
        v = self.v(x)[e_ik].view(-1, self.h, self.dh).float()
        out = torch.zeros(E, self.h, self.dh, device=z.device).index_add_(0, e_ij, w[..., None] * v)
        out = (torch.sigmoid(self.g(x)) * out.reshape(E, -1).to(x.dtype))
        return self.o(out).view(N, K, de)


class Block(nn.Module):
    def __init__(self, dn, de, dtri, heads, tri_attn):
        super().__init__()
        self.node_att = NodeAttention(dn, de, heads)
        self.node_tr = Transition(dn)
        self.e_from_n = EdgeFromNodes(dn, de)
        self.rev = ReverseMix(de)
        self.tri_out = TriangleMultiplication(de, dtri, "out")
        self.tri_in = TriangleMultiplication(de, dtri, "in")
        self.tri_att = TriangleAttention(de, heads) if tri_attn else None
        self.edge_tr = Transition(de)

    def forward(self, h, z, NB, rev_pos, mutual, tri):
        h = h + self.node_att(h, z, NB)
        h = h + self.node_tr(h)
        z = z + self.e_from_n(h, NB)
        z = z + self.rev(z, NB, rev_pos, mutual)
        z = z + self.tri_out(z, tri)
        z = z + self.tri_in(z, tri)
        if self.tri_att is not None:
            z = z + self.tri_att(z, tri)
        z = z + self.edge_tr(z)
        return h, z


class PairFormerTSP(nn.Module):
    def __init__(self, dn=128, de=64, dtri=32, layers=8, heads=4, tri_attn=False, recycles=0,
                 grad_ckpt=False, k_tri=12, node_in=5, edge_in=9):
        super().__init__()
        self.cfg = dict(dn=dn, de=de, dtri=dtri, layers=layers, heads=heads, tri_attn=tri_attn,
                        recycles=recycles, grad_ckpt=grad_ckpt, k_tri=k_tri, node_in=node_in, edge_in=edge_in)
        self.node_in = nn.Sequential(nn.Linear(node_in, dn), nn.GELU(), nn.Linear(dn, dn))
        self.edge_in = nn.Sequential(nn.Linear(edge_in, de), nn.GELU(), nn.Linear(de, de))
        self.rec_h = nn.Sequential(nn.LayerNorm(dn), nn.Linear(dn, dn))
        self.rec_z = nn.Sequential(nn.LayerNorm(de), nn.Linear(de, de))
        for m in (self.rec_h[1], self.rec_z[1]):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.blocks = nn.ModuleList(Block(dn, de, dtri, heads, tri_attn) for _ in range(layers))
        self.edge_head = nn.Sequential(nn.LayerNorm(de), nn.Linear(de, de), nn.GELU(), nn.Linear(de, 1))
        self.pi_head = nn.Sequential(nn.LayerNorm(dn), nn.Linear(dn, dn), nn.GELU(), nn.Linear(dn, 1))
        self.grad_ckpt = grad_ckpt
        self.recycles = recycles

    def trunk(self, ctx, NB, h_prev=None, z_prev=None):
        h = self.node_in(ctx["node_feat"])
        z = self.edge_in(ctx["edge_feat"])
        if h_prev is not None:
            h = h + self.rec_h(h_prev)
            z = z + self.rec_z(z_prev)
        for blk in self.blocks:
            if self.grad_ckpt and self.training:
                h, z = checkpoint(blk, h, z, NB, ctx["rev_pos"], ctx["mutual"], ctx["tri"], use_reentrant=False)
            else:
                h, z = blk(h, z, NB, ctx["rev_pos"], ctx["mutual"], ctx["tri"])
        return h, z

    def forward(self, ctx, NB, n_recycle=None):
        n_rec = self.recycles if n_recycle is None else n_recycle
        h_prev = z_prev = None
        if n_rec > 0:
            with torch.no_grad():
                for _ in range(n_rec):
                    h_prev, z_prev = self.trunk(ctx, NB, h_prev, z_prev)
                    h_prev, z_prev = h_prev.detach(), z_prev.detach()
        h, z = self.trunk(ctx, NB, h_prev, z_prev)
        logit = self.edge_head(z).squeeze(-1).float()                  # (N,K) directed
        mutual, rev_pos = ctx["mutual"], ctx["rev_pos"]
        logit_rev = logit[NB, rev_pos]
        sym = torch.where(mutual, 0.5 * (logit + logit_rev), logit)
        f = self.pi_head(h).squeeze(-1).float()
        return {"edge_logit": sym, "pi_f": f}


# ------------------------------------------------------------------------- losses
def scatter_mean(x, idx, B):
    s = torch.zeros(B, device=x.device, dtype=x.dtype).index_add_(0, idx, x)
    c = torch.zeros(B, device=x.device, dtype=x.dtype).index_add_(0, idx, torch.ones_like(x))
    return s / c.clamp_min(1)


def compute_losses(out, b, w_bce=0.5, w_pi=1.0):
    logit = out["edge_logit"]
    y = b["y"].float()
    K = y.shape[1]
    has = y.sum(1) > 0
    logp = F.log_softmax(logit, 1)
    ce = -((y / y.sum(1, keepdim=True).clamp_min(1)) * logp).sum(1)[has].mean()
    bce = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor((K - 2) / 2, device=y.device))
    loss = ce + w_bce * bce
    logs = {"ce": ce.item(), "bce": bce.item()}
    if "pi_t" in b and w_pi > 0:
        s, bidx, B = b["s"], b["bidx"], b["B"]
        pi_hat, pi_tgt = s * out["pi_f"], s * b["pi_t"]
        d = (pi_hat - scatter_mean(pi_hat, bidx, B)[bidx]) - (pi_tgt - scatter_mean(pi_tgt, bidx, B)[bidx])
        mse = (d / s).pow(2).mean()
        loss = loss + w_pi * mse
        logs["pi_mse"] = mse.item()
    logs["loss"] = loss.item()
    return loss, logs


# ---------------------------------------------------------------------- inference
def autocast_dtype(device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


@torch.no_grad()
def predict(model, xy, K, device, k_out=5, n_recycle=None):
    """Return LKH-ready candidates (n,k_out), alpha ranks, integer pi and timings."""
    t0 = time.perf_counter()
    prep = prepare_instance(xy, K)
    b = to_device(collate([prep]), device)
    t1 = time.perf_counter()
    ctx = graph_context(b, k_tri=model.cfg.get("k_tri"))
    dt = autocast_dtype(device)
    model.eval()
    with torch.autocast(device_type=device.type, dtype=dt, enabled=dt is not None):
        out = model(ctx, b["NB"], n_recycle=n_recycle)
    logit = out["edge_logit"].float()
    top = logit.topk(k_out, dim=1).indices
    cand = torch.gather(b["NB"], 1, top).cpu().numpy()
    pi_hat = (b["s"] * out["pi_f"]).double()
    pi_hat = pi_hat - pi_hat.mean()
    pi_int = torch.round(pi_hat * PRECISION * prep["span"]).long().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    alpha = np.tile(np.arange(k_out, dtype=np.int64) * 100, (len(cand), 1))
    return {"cand": cand, "alpha": alpha, "pi": pi_int, "logit": logit.cpu().numpy(),
            "NB": prep["NB"], "t_prep": t1 - t0, "t_model": t2 - t1, "t_total": t2 - t0}


# =============================================================================== TourFold
# The ensemble of cheap local-optimum tours plays the role of AlphaFold's MSA:
#   column attention  = for one node, compare how the M tours connect it
#   tour convolution  = context along each tour (successor + predecessor, shared weights)
#   row attention     = for one tour and node, look at the node's kNN neighbours, biased by the pair rep
#   outer-product mean= co-variation across tours  ->  pair (edge) representation
# followed by the pair stack with triangle updates (Block above).

class MSAColumnAttention(nn.Module):
    def __init__(self, dm, heads=4):
        super().__init__()
        self.h, self.dh = heads, max(4, dm // heads)
        self.ln = nn.LayerNorm(dm)
        self.qkv = nn.Linear(dm, 3 * self.h * self.dh)
        self.g, self.o = nn.Linear(dm, self.h * self.dh), nn.Linear(self.h * self.dh, dm)

    def forward(self, msa):
        M, N, _ = msa.shape
        x = self.ln(msa)
        q, k, v = self.qkv(x).view(M, N, 3, self.h, self.dh).unbind(2)
        logit = torch.einsum("mnhd,lnhd->nhml", q, k) / math.sqrt(self.dh)
        att = logit.float().softmax(-1).to(v.dtype)
        out = torch.einsum("nhml,lnhd->mnhd", att, v).reshape(M, N, -1)
        return self.o(torch.sigmoid(self.g(x)) * out)


class MSATourConv(nn.Module):
    """msa[m,i] += f(msa[m,i], msa[m,succ_m(i)] + msa[m,pred_m(i)]) - direction-free context along each tour."""

    def __init__(self, dm):
        super().__init__()
        self.ln = nn.LayerNorm(dm)
        self.self_w, self.nbr_w = nn.Linear(dm, dm), nn.Linear(dm, dm)
        self.o = nn.Linear(dm, dm)

    def forward(self, msa, succ, pred):
        M, N, dm = msa.shape
        x = self.ln(msa)
        nb = torch.gather(x, 1, succ[..., None].expand(M, N, dm)) + torch.gather(x, 1, pred[..., None].expand(M, N, dm))
        return self.o(F.gelu(self.self_w(x) + self.nbr_w(nb)))


class MSARowAttention(nn.Module):
    """For tour m and node i: attention over i's kNN neighbours j, biased by the pair representation z_ij."""

    def __init__(self, dm, de, heads=4):
        super().__init__()
        self.h, self.dh = heads, max(4, dm // heads)
        self.ln, self.lnz = nn.LayerNorm(dm), nn.LayerNorm(de)
        self.q, self.kv = nn.Linear(dm, self.h * self.dh), nn.Linear(dm, 2 * self.h * self.dh)
        self.bias = nn.Linear(de, heads)
        self.g, self.o = nn.Linear(dm, self.h * self.dh), nn.Linear(self.h * self.dh, dm)

    def forward(self, msa, z, NB):
        M, N, _ = msa.shape
        K = NB.shape[1]
        x = self.ln(msa)
        q = self.q(x).view(M, N, self.h, self.dh)
        k, v = self.kv(x)[:, NB].split(self.h * self.dh, -1)               # (M,N,K,h*dh)
        k, v = k.view(M, N, K, self.h, self.dh), v.view(M, N, K, self.h, self.dh)
        logit = torch.einsum("mnhd,mnkhd->mnhk", q, k) / math.sqrt(self.dh)
        logit = logit + self.bias(self.lnz(z)).permute(0, 2, 1)[None]
        att = logit.float().softmax(-1).to(v.dtype)
        out = torch.einsum("mnhk,mnkhd->mnhd", att, v).reshape(M, N, -1)
        return self.o(torch.sigmoid(self.g(x)) * out)


class OuterProductMean(nn.Module):
    """z_ij += W * mean_m (a[m,i] (x) b[m,j]) on kNN edges: co-variation of how the tours treat i and j."""

    def __init__(self, dm, de, c=8):
        super().__init__()
        self.c = c
        self.ln = nn.LayerNorm(dm)
        self.a, self.b = nn.Linear(dm, c), nn.Linear(dm, c)
        self.o = nn.Linear(c * c, de)

    def forward(self, msa, NB):
        M, N, _ = msa.shape
        K = NB.shape[1]
        x = self.ln(msa)
        a, b = self.a(x), self.b(x)[:, NB]                                     # (M,N,c), (M,N,K,c)
        opm = torch.einsum("mnc,mnkd->nkcd", a.float(), b.float()) / M
        return self.o(opm.reshape(N, K, self.c * self.c).to(x.dtype))


class EvoBlock(nn.Module):
    def __init__(self, dn, de, dm, dtri, heads, tri_attn, opm_c=8):
        super().__init__()
        self.col = MSAColumnAttention(dm, heads)
        self.conv = MSATourConv(dm)
        self.row = MSARowAttention(dm, de, heads)
        self.msa_tr = Transition(dm)
        self.opm = OuterProductMean(dm, de, opm_c)
        self.pair = Block(dn, de, dtri, heads, tri_attn)

    def forward(self, msa, h, z, NB, rev_pos, mutual, tri, succ, pred):
        msa = msa + self.col(msa)
        msa = msa + self.conv(msa, succ, pred)
        msa = msa + self.row(msa, z, NB)
        msa = msa + self.msa_tr(msa)
        z = z + self.opm(msa, NB)
        h, z = self.pair(h, z, NB, rev_pos, mutual, tri)
        return msa, h, z


class TourFoldNet(nn.Module):
    """use_ensemble=False gives the single-instance ablation (same pair stack, no tour alignment)."""

    def __init__(self, dn=128, de=64, dm=32, dtri=32, msa_layers=2, pair_layers=6, heads=4,
                 tri_attn=False, use_ensemble=True, k_tri=12, K=16, opm_c=8, grad_ckpt=False,
                 node_in=5, edge_in=9, msa_in=5, graph_extra=0):
        super().__init__()
        self.cfg = dict(dn=dn, de=de, dm=dm, dtri=dtri, msa_layers=msa_layers, pair_layers=pair_layers,
                        heads=heads, tri_attn=tri_attn, use_ensemble=use_ensemble, k_tri=k_tri, K=K,
                        opm_c=opm_c, grad_ckpt=grad_ckpt, node_in=node_in, edge_in=edge_in, msa_in=msa_in,
                        graph_extra=int(graph_extra if use_ensemble else 0))
        self.use_ensemble = use_ensemble
        self.node_in = nn.Sequential(nn.Linear(node_in, dn), nn.GELU(), nn.Linear(dn, dn))
        e_in = edge_in + (2 if use_ensemble else 0) + (1 if (use_ensemble and graph_extra) else 0)
        self.edge_in = nn.Sequential(nn.Linear(e_in, de), nn.GELU(),
                                     nn.Linear(de, de))
        if use_ensemble:
            self.msa_in = nn.Sequential(nn.Linear(msa_in, dm), nn.GELU(), nn.Linear(dm, dm))
            self.pos_emb = nn.Embedding(K + 1, dm)
            self.evo = nn.ModuleList(EvoBlock(dn, de, dm, dtri, heads, tri_attn, opm_c) for _ in range(msa_layers))
        self.blocks = nn.ModuleList(Block(dn, de, dtri, heads, tri_attn) for _ in range(pair_layers))
        self.edge_head = nn.Sequential(nn.LayerNorm(de), nn.Linear(de, de), nn.GELU(), nn.Linear(de, 1))
        self.pi_head = nn.Sequential(nn.LayerNorm(dn), nn.Linear(dn, dn), nn.GELU(), nn.Linear(dn, 1))
        self.conf_head = nn.Sequential(nn.LayerNorm(dn), nn.Linear(dn, dn), nn.GELU(), nn.Linear(dn, 1))
        self.grad_ckpt = grad_ckpt

    def _run(self, fn, *args):
        if self.grad_ckpt and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, ctx, b, n_recycle=None):
        NB = b["NB"]
        h = self.node_in(ctx["node_feat"])
        z = self.edge_in(ctx["edge_feat"])
        if self.use_ensemble:
            msa = self.msa_in(b["msa_f"]) + self.pos_emb(b["msa_ps"]) + self.pos_emb(b["msa_pp"])
            for blk in self.evo:
                msa, h, z = self._run(blk, msa, h, z, NB, ctx["rev_pos"], ctx["mutual"], ctx["tri"],
                                      b["ens_succ"], b["ens_pred"])
        for blk in self.blocks:
            h, z = self._run(blk, h, z, NB, ctx["rev_pos"], ctx["mutual"], ctx["tri"])
        logit = self.edge_head(z).squeeze(-1).float()
        mutual, rev_pos = ctx["mutual"], ctx["rev_pos"]
        sym = torch.where(mutual, 0.5 * (logit + logit[NB, rev_pos]), logit)
        return {"edge_logit": sym, "pi_f": self.pi_head(h).squeeze(-1).float(),
                "conf_logit": self.conf_head(h).squeeze(-1).float()}


def tourfold_losses(out, b, w_bce=0.5, w_pi=1.0, w_conf=0.2):
    loss, logs = compute_losses(out, b, w_bce, w_pi)
    if "conf_logit" in out and w_conf > 0:
        top2 = out["edge_logit"].detach().topk(2, dim=1).indices
        target = torch.gather(b["y"], 1, top2).all(1).float()
        lc = F.binary_cross_entropy_with_logits(out["conf_logit"], target)
        loss = loss + w_conf * lc
        logs["conf_bce"] = lc.item()
        logs["loss"] = loss.item()
    return loss, logs


def load_model(path, device):
    """Rebuild a TourFoldNet from a training checkpoint (its stored config) and load the weights."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ck["cfg"])
    cfg["grad_ckpt"] = False
    model = TourFoldNet(**cfg).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), ck


def rank_neighbours(NB, logit):
    """All graph neighbours of every node, best first by logit; equal logits (bf16 output) keep the graph's
    distance order, so the ranking is deterministic."""
    return torch.gather(NB, 1, torch.sort(logit, dim=1, descending=True, stable=True).indices)


@torch.no_grad()
def predict_tourfold(model, xy, K, device, ens_succ=None, ens_cost=None, k_out=5):
    """Candidates, integer pi, per-edge probabilities and per-node confidence for one instance.
    'rank' lists all K graph neighbours of each node, best first (see rank_neighbours)."""
    t0 = time.perf_counter()
    use = model.cfg.get("use_ensemble", False)
    prep = prepare_instance(xy, K, ens_succ=ens_succ if use else None, ens_cost=ens_cost if use else None,
                            extra=model.cfg.get("graph_extra", 0) if use else 0)
    b = to_device(collate([prep]), device)
    t1 = time.perf_counter()
    ctx = graph_context(b, k_tri=model.cfg.get("k_tri"))
    dt = autocast_dtype(device)
    model.eval()
    with torch.autocast(device_type=device.type, dtype=dt, enabled=dt is not None):
        out = model(ctx, b)
    logit = out["edge_logit"].float()
    top = logit.topk(k_out, dim=1).indices
    cand = torch.gather(b["NB"], 1, top).cpu().numpy()
    rank = rank_neighbours(b["NB"], logit).cpu().numpy()
    pi_hat = (b["s"] * out["pi_f"]).double()
    pi_hat = pi_hat - pi_hat.mean()
    pi_int = torch.round(pi_hat * PRECISION * prep["span"]).long().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    alpha = np.tile(np.arange(k_out, dtype=np.int64) * 100, (len(cand), 1))
    return {"cand": cand, "rank": rank, "alpha": alpha, "pi": pi_int, "prob": torch.sigmoid(logit).cpu().numpy(),
            "conf": torch.sigmoid(out["conf_logit"]).cpu().numpy(), "NB": prep["NB"], "D": prep["D"],
            "t_prep": t1 - t0, "t_model": t2 - t1, "t_total": t2 - t0}
