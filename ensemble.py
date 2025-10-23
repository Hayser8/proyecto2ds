import os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Config y rutas
# -------------------------------
ROOT = Path(".").resolve()
DATA_CLEAN = ROOT / "data" / "clean"
ARTIFACTS  = ROOT / "artifacts_ens"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# Hiper-parámetros de calibración del BTD (ajústalos si ya mediste otros)
GAMMA  = 0.80
LAMBDA = 1.143
# Si no tenemos tau_hat a mano, usa el último que reportaste o uno ~0.49
DEFAULT_TAU = 0.490347

np.set_printoptions(suppress=True, floatmode="fixed", precision=6)


# -------------------------------
# Utilidades
# -------------------------------
def info(msg): print(f"[INFO] {msg}")

def find_first(paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None

def ensure_prob_matrix(df, prefix=""):
    """
    Normaliza columnas del modelo a p_A, p_B, p_TIE.
    Admite varios nombres (p.ej. winner_*_prob, logits -> softmax, etc.).
    Si sólo hay binario A/B, construye p_TIE = 1 - max(pA, pB).
    """
    cols = {c.lower(): c for c in df.columns}

    # 1) Probabilidades claras
    candA = [c for c in cols if c in {f"{prefix}p_a", "p_a", "winner_model_a", "winner_model_a_prob"}]
    candB = [c for c in cols if c in {f"{prefix}p_b", "p_b", "winner_model_b", "winner_model_b_prob"}]
    candT = [c for c in cols if c in {f"{prefix}p_tie", "p_tie", "winner_tie", "winner_tie_prob"}]

    out = pd.DataFrame(index=df.index)

    if candA and candB and candT:
        out["p_A"] = df[cols[candA[0]]].astype(float)
        out["p_B"] = df[cols[candB[0]]].astype(float)
        out["p_TIE"] = df[cols[candT[0]]].astype(float)

    elif candA and candB:
        # binario → crear TIE
        pA = df[cols[candA[0]]].astype(float).to_numpy()
        pB = df[cols[candB[0]]].astype(float).to_numpy()
        pT = 1.0 - np.maximum(pA, pB)
        out["p_A"], out["p_B"], out["p_TIE"] = pA, pB, pT

    else:
        # ¿hay logits? intenta detectar "logit_a/logit_b/logit_tie"
        logA = [c for c in cols if c in {f"{prefix}logit_a","logit_a"}]
        logB = [c for c in cols if c in {f"{prefix}logit_b","logit_b"}]
        logT = [c for c in cols if c in {f"{prefix}logit_tie","logit_tie"}]
        if logA and logB and logT:
            Z = df[[cols[logA[0]], cols[logB[0]], cols[logT[0]]]].to_numpy(dtype=float)
            e = np.exp(Z - Z.max(axis=1, keepdims=True))
            P = e / e.sum(axis=1, keepdims=True)
            out["p_A"], out["p_B"], out["p_TIE"] = P[:,0], P[:,1], P[:,2]
        else:
            raise ValueError("No pude inferir columnas de probas. Asegúrate de tener p_A/p_B/(p_TIE) o logits.")

    # Clipping + renormalización
    P = out[["p_A","p_B","p_TIE"]].to_numpy(dtype=float)
    P = np.clip(P, 1e-9, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    out = pd.DataFrame(P, columns=["p_A","p_B","p_TIE"], index=df.index)
    return out


# -------------------------------
# BTD: carga o reconstruye predicciones en VAL/TEST
# -------------------------------
def load_or_build_btd_preds(split: str):
    """
    split ∈ {"val","test"}
    Intenta leer data/clean/btd_preds_{split}.parquet.
    Si no existe, reconstruye a partir de:
      - data/clean/{split}15.parquet
      - data/clean/btd_strengths.csv
      - DEFAULT_TAU / (GAMMA, LAMBDA)
    """
    assert split in {"val","test"}
    pred_path = DATA_CLEAN / f"btd_preds_{split}.parquet"
    if pred_path.exists():
        info(f"BTD {split}: usando {pred_path}")
        DF = pd.read_parquet(pred_path)
        return DF[["id","p_A","p_B","p_TIE"]].copy()

    # Reconstrucción
    info(f"BTD {split}: reconstruyendo probabilidades…")
    # 1) strengths
    strengths = find_first([
        DATA_CLEAN / "btd_strengths.csv",
        ROOT / "artifacts_btd" / "btd_strengths.csv",
        ROOT / "btd_strengths.csv",
    ])
    if strengths is None:
        raise FileNotFoundError("No encontré btd_strengths.csv")

    df_strengths = pd.read_csv(strengths)
    assert {"model","beta"}.issubset(df_strengths.columns), "btd_strengths.csv debe tener 'model','beta'"
    beta_map = dict(zip(df_strengths["model"].astype(str), df_strengths["beta"].astype(float)))

    # 2) split data
    split_file = DATA_CLEAN / (f"{split}15.parquet" if split in {"val","test"} else f"{split}.parquet")
    if not split_file.exists():
        raise FileNotFoundError(f"No encontré {split_file}")

    df = pd.read_parquet(split_file)
    need = {"id","model_a","model_b"}
    if not need.issubset(df.columns):
        raise ValueError(f"{split_file} debe tener columnas {need}")

    tau = DEFAULT_TAU
    # 3) predicción con Davidson calibrado
    def predict_btd_by_names(model_a_names, model_b_names, beta_map, tau, gamma=1.0, lamb=1.0):
        bA = np.array([beta_map.get(str(a), 0.0) for a in model_a_names], dtype=float) * gamma
        bB = np.array([beta_map.get(str(b), 0.0) for b in model_b_names], dtype=float) * gamma
        pi = np.exp(bA); pj = np.exp(bB)
        tau_eff = tau * lamb
        denom = pi + pj + 2.0 * tau_eff * np.sqrt(pi * pj)
        pA = pi / denom
        pB = pj / denom
        pT = 1.0 - pA - pB
        P = np.vstack([pA, pB, pT]).T
        P = np.clip(P, 1e-12, 1.0)
        P /= P.sum(axis=1, keepdims=True)
        return P

    P = predict_btd_by_names(
        df["model_a"].astype(str).tolist(),
        df["model_b"].astype(str).tolist(),
        beta_map, tau, GAMMA, LAMBDA
    )

    out = pd.DataFrame({"id": df["id"].values, "p_A": P[:,0], "p_B": P[:,1], "p_TIE": P[:,2]})
    # guarda para la próxima
    out.to_parquet(pred_path, index=False)
    info(f"BTD {split}: guardado {pred_path}")
    return out


# -------------------------------
# XENC: localizar predicciones en VAL/TEST (flexible)
# -------------------------------
def load_xenc_preds(split: str):
    """
    split ∈ {"val","test"}
    Busca archivos típicos con predicciones del cross-encoder y normaliza a p_A/p_B/p_TIE.
    """
    assert split in {"val","test"}
    # Rutas candidatas según tu screenshot
    candidates = [
        ROOT / "traineddatamod1" / f"{split}_predictions.parquet",
        ROOT / "traineddatamod1" / f"{split}_eval_join.parquet",
        ROOT / "outputs" / "deberta_xenc" / f"{split}_predictions.parquet",
        ROOT / "outputs_ce_distilroberta" / f"{split}_predictions.parquet",
        ROOT / "reports" / f"{split}_predictions.parquet",
        ROOT / "reports" / f"{split}_eval_join.parquet",
        ROOT / "data" / f"{split}_predictions.parquet",
    ]
    p = find_first(candidates)
    if p is None:
        raise FileNotFoundError(
            f"No encontré predicciones del xenc para {split}. "
            f"Busca algo tipo '*{split}_predictions.parquet' en traineddatamod1/ o outputs/*/"
        )

    df = pd.read_parquet(p)
    if "id" not in df.columns:
        # algunos pipelines llaman 'pair_id' o similar
        for k in ["pair_id","row_id","example_id"]:
            if k in df.columns:
                df = df.rename(columns={k:"id"})
                break
        if "id" not in df.columns:
            raise ValueError(f"{p} no tiene columna id")

    P = ensure_prob_matrix(df)
    out = pd.concat([df[["id"]].reset_index(drop=True), P.reset_index(drop=True)], axis=1)
    info(f"XENC {split}: usando {p}")
    return out


# -------------------------------
# Métricas en validación
# -------------------------------
def load_val_labels():
    y = pd.read_parquet(DATA_CLEAN / "val15.parquet")[["id","winner_model_a","winner_model_b","winner_tie"]]
    y["label"] = np.select(
        [y.winner_model_a.eq(1), y.winner_model_b.eq(1), y.winner_tie.eq(1)],
        ["A","B","TIE"]
    )
    return y[["id","label"]]


def evaluate_probs(df_probs, y_true):
    df = df_probs.merge(y_true, on="id", how="inner")
    idx = df[["p_A","p_B","p_TIE"]].to_numpy().argmax(1)
    pred = np.array(["A","B","TIE"])[idx]
    f1m = f1_score(df["label"], pred, average="macro")
    acc = accuracy_score(df["label"], pred)
    return f1m, acc, len(df)


# -------------------------------
# BLENDING (optimiza w en VAL)
# -------------------------------
def run_blending(btd_val, xenc_val, btd_test, xenc_test):
    y_val = load_val_labels()
    VAL = btd_val.merge(xenc_val, on="id", suffixes=("_btd","_x"))
    TEST = btd_test.merge(xenc_test, on="id", suffixes=("_btd","_x"))

    def blend(Pb, Px, w):
        P = w*Px + (1.0-w)*Pb
        P = np.clip(P, 1e-9, 1.0)
        P = P / P.sum(axis=1, keepdims=True)
        return P

    Pb = VAL[["p_A_btd","p_B_btd","p_TIE_btd"]].to_numpy()
    Px = VAL[["p_A_x","p_B_x","p_TIE_x"]].to_numpy()

    grid = np.linspace(0.00, 1.00, 41)
    best = None
    for w in grid:
        P = blend(Pb, Px, w)
        f1m, acc, n = evaluate_probs(pd.DataFrame({"id": VAL["id"], "p_A":P[:,0],"p_B":P[:,1],"p_TIE":P[:,2]}), y_val)
        cand = (f1m, acc, w)
        if best is None or cand > best:
            best = cand
    f1m, acc, W = best
    info(f"BLEND: mejor w={W:.3f}  |  VAL F1-macro={f1m:.4f} Acc={acc:.4f}")

    # aplicar a TEST
    PbT = TEST[["p_A_btd","p_B_btd","p_TIE_btd"]].to_numpy()
    PxT = TEST[["p_A_x","p_B_x","p_TIE_x"]].to_numpy()
    PT  = blend(PbT, PxT, W)
    sub = pd.DataFrame({"id": TEST["id"], "winner_model_a": PT[:,0], "winner_model_b": PT[:,1], "winner_tie": PT[:,2]})
    out = ARTIFACTS / "submit_blend.csv"
    sub.to_csv(out, index=False)
    info(f"BLEND: guardado {out}")
    return W, (f1m, acc)


# -------------------------------
# STACKING (meta-modelo ligero en VAL)
# -------------------------------
def safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1-eps)
    return np.log(p/(1-p))

def run_stacking(btd_val, xenc_val, btd_test, xenc_test):
    y_val = load_val_labels()
    VAL = btd_val.merge(xenc_val, on="id", suffixes=("_btd","_x")).merge(y_val, on="id")
    TEST = btd_test.merge(xenc_test, on="id", suffixes=("_btd","_x"))

    Xv = VAL[["p_A_btd","p_B_btd","p_TIE_btd","p_A_x","p_B_x","p_TIE_x"]].to_numpy()
    # Variante con logits (a veces mejora):
    # Xv = np.column_stack([safe_logit(VAL[c].to_numpy()) for c in
    #                      ["p_A_btd","p_B_btd","p_TIE_btd","p_A_x","p_B_x","p_TIE_x"]])

    y_map = {"A":0,"B":1,"TIE":2}
    yv = VAL["label"].map(y_map).to_numpy()

    meta = LogisticRegression(max_iter=200, multi_class="multinomial", C=2.0)
    meta.fit(Xv, yv)

    Pv = meta.predict_proba(Xv)
    pred_v = Pv.argmax(1)
    f1m = f1_score(yv, pred_v, average="macro")
    acc = accuracy_score(yv, pred_v)
    info(f"STACK: VAL F1-macro={f1m:.4f} Acc={acc:.4f}")

    Xt = TEST[["p_A_btd","p_B_btd","p_TIE_btd","p_A_x","p_B_x","p_TIE_x"]].to_numpy()
    Pt = meta.predict_proba(Xt)
    sub = pd.DataFrame({"id": TEST["id"], "winner_model_a": Pt[:,0], "winner_model_b": Pt[:,1], "winner_tie": Pt[:,2]})
    out = ARTIFACTS / "submit_stack.csv"
    sub.to_csv(out, index=False)
    info(f"STACK: guardado {out}")
    return (f1m, acc)


# -------------------------------
# MAIN
# -------------------------------
def main():
    # 1) BTD preds (val/test)
    btd_val = load_or_build_btd_preds("val")
    btd_test = load_or_build_btd_preds("test")

    # 2) XENC preds (val/test)
    xenc_val = load_xenc_preds("val")
    xenc_test = load_xenc_preds("test")

    # 3) Sanidad de IDs
    for name, A, B in [
        ("VAL", btd_val, xenc_val),
        ("TEST", btd_test, xenc_test),
    ]:
        inter = set(A["id"]) & set(B["id"])
        if len(inter) == 0:
            raise RuntimeError(f"{name}: no hay IDs comunes entre BTD y XENC")
        if len(inter) < len(A) or len(inter) < len(B):
            info(f"{name}: warning, hay IDs que no cruzan (usando intersección).")
        A.set_index("id", inplace=True); B.set_index("id", inplace=True)
        merged = A.join(B, how="inner", lsuffix="_btd", rsuffix="_x").reset_index()
        if name == "VAL":
            merged.to_parquet(ARTIFACTS / "val_join_for_ens.parquet", index=False)
        else:
            merged.to_parquet(ARTIFACTS / "test_join_for_ens.parquet", index=False)
        # devolver a dfs originales
        A.reset_index(inplace=True); B.reset_index(inplace=True)

    # 4) BLENDING
    W, (f1b, accb) = run_blending(btd_val, xenc_val, btd_test, xenc_test)

    # 5) STACKING
    f1s, accs = run_stacking(btd_val, xenc_val, btd_test, xenc_test)

    info("=== RESUMEN ===")
    info(f"BLEND: w*XENC + (1-w)*BTD con w={W:.3f} | VAL F1-macro={f1b:.4f} Acc={accb:.4f}")
    info(f"STACK: LogisticRegression(multinomial)         | VAL F1-macro={f1s:.4f} Acc={accs:.4f}")
    info(f"Submissions: {ARTIFACTS/'submit_blend.csv'}  y  {ARTIFACTS/'submit_stack.csv'}")

if __name__ == "__main__":
    main()
