from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

import optimize_rice_strict399_N4_go_ppi_threeway as graph
import train_rice_N6_four_feature_variants_fixed_split as four
import train_rice_E0_Nge6_common6751_fixed80_10_10 as library


ROOT = Path(
    "E:/CodexMoved/Desktop/\u6c34\u7a3b/"
    "cross_species_ath_rice_common_features_models"
)
SOURCE_ROOT = Path(
    "E:/CodexMoved/Desktop/\u6c34\u7a3b/rice_mutant_sources"
)
STRICT = (
    SOURCE_ROOT
    / "oryzabase_all_essentiality_20260620"
    / "oryzabase_gene_specific_strict_binary_labels.tsv"
)
TOS17 = (
    SOURCE_ROOT
    / "processed_rap_es"
    / "tos17_rap_gene_ES_classification.tsv"
)
OUT = ROOT / "rice_strict399_N4_new80_10_10_OOF_bootstrap"
SPLIT_SEED = 20260621
CV_SEED = 20260622
BOOTSTRAP_SEED = 20260623
N_BOOTSTRAP = 10000


def split_class(values: np.ndarray, seed: int) -> dict[str, set[str]]:
    values = np.asarray(sorted(set(values)), dtype=object)
    rng = np.random.RandomState(seed)
    rng.shuffle(values)
    n_test = int(round(len(values) * 0.10))
    n_validation = int(round(len(values) * 0.10))
    return {
        "test": set(values[:n_test]),
        "validation": set(values[n_test : n_test + n_validation]),
        "train": set(values[n_test + n_validation :]),
    }


def build_dataset(meta: pd.DataFrame):
    strict = pd.read_csv(STRICT, sep="\t", dtype=str).fillna("")
    essential = set(
        strict.loc[
            strict["curated_final_classification"].eq("essential"),
            "rap_gene_id",
        ]
    )
    tos17 = pd.read_csv(TOS17, sep="\t", dtype=str).fillna("")
    e = pd.to_numeric(tos17["essential_evidence_count_E"], errors="coerce").fillna(0)
    n = pd.to_numeric(tos17["nonessential_evidence_count_N"], errors="coerce").fillna(0)
    es = pd.to_numeric(
        tos17["essentiality_score_ES_E2_over_T2"], errors="coerce"
    )
    nonessential = set(
        tos17.loc[e.eq(0) & n.ge(4) & es.lt(0.1), "rap_gene_id"]
    ) - essential
    labels = pd.DataFrame(
        [{"gene_id": g, "label": 1} for g in sorted(essential)]
        + [{"gene_id": g, "label": 0} for g in sorted(nonessential)]
    )
    meta_index = meta[["gene_id"]].copy()
    meta_index["gene_id"] = meta_index["gene_id"].astype(str)
    meta_index["matrix_row"] = np.arange(len(meta_index), dtype=int)
    joined = labels.merge(meta_index, on="gene_id", how="left", validate="one_to_one")
    missing = joined[joined["matrix_row"].isna()].copy()
    joined = joined[joined["matrix_row"].notna()].copy()
    joined["matrix_row"] = joined["matrix_row"].astype(int)
    splits = {
        label: split_class(
            joined.loc[joined["label"].eq(label), "gene_id"].to_numpy(),
            SPLIT_SEED + label * 101,
        )
        for label in [0, 1]
    }
    parts = {}
    for split_name in ["train", "validation", "test"]:
        genes = splits[0][split_name] | splits[1][split_name]
        part = joined[joined["gene_id"].isin(genes)].copy()
        part["split"] = split_name
        parts[split_name] = part.reset_index(drop=True)
    return labels, joined, missing, parts


def ranking(y, p):
    return {
        "auc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
    }


def threshold_row(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    return {
        "threshold": float(threshold),
        "sensitivity": float(sn),
        "specificity": float(sp),
        "min_sn_sp": float(min(sn, sp)),
        "youden_index": float(sn + sp - 1),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def threshold_table(y, p):
    candidates = np.unique(np.r_[0, 1, p, np.nextafter(p, np.inf)])
    return pd.DataFrame([threshold_row(y, p, t) for t in candidates])


def select_threshold(y, p, rule):
    table = threshold_table(y, p)
    if rule == "dual_constraint":
        feasible = table[
            table["sensitivity"].ge(0.8) & table["specificity"].ge(0.8)
        ].copy()
        if not feasible.empty:
            ranked = feasible.sort_values(
                ["youden_index", "min_sn_sp", "f1", "threshold"],
                ascending=[False, False, False, False],
            )
            note = "SN>=0.8 and SP>=0.8; maximum Youden"
        else:
            ranked = table.sort_values(
                ["min_sn_sp", "youden_index", "f1", "threshold"],
                ascending=False,
            )
            note = "fallback: maximum min(SN,SP)"
    elif rule == "sn_priority":
        feasible = table[table["sensitivity"].ge(0.8)].copy()
        if not feasible.empty:
            ranked = feasible.sort_values(
                ["specificity", "sensitivity", "youden_index", "f1", "threshold"],
                ascending=False,
            )
            note = "SN>=0.8; maximum SP"
        else:
            ranked = table.sort_values(
                ["sensitivity", "specificity", "youden_index", "f1", "threshold"],
                ascending=False,
            )
            note = "fallback: maximum SN then SP"
    else:
        raise ValueError(rule)
    selected = ranked.iloc[0].to_dict()
    selected["rule"] = rule
    selected["rule_note"] = note
    selected["feasible"] = bool(not feasible.empty)
    return selected, table


def go_probability(train_genes, train_y, target_genes, terms, smoothing):
    term_sum, term_count = defaultdict(float), defaultdict(int)
    prior = float(np.mean(train_y))
    for gene, label in zip(train_genes, train_y):
        for term in terms.get(gene, ()):
            term_sum[term] += float(label)
            term_count[term] += 1
    result = []
    for gene in target_genes:
        values, weights = [], []
        for term in terms.get(gene, ()):
            count = term_count.get(term, 0)
            if count:
                values.append(
                    (term_sum[term] + smoothing * prior) / (count + smoothing)
                )
                weights.append(np.log1p(count))
        result.append(np.average(values, weights=weights) if values else prior)
    return np.asarray(result)


def make_transition(n_genes, edges, threshold):
    rows, cols, data = [], [], []
    for (i, j), score in edges.items():
        if score < threshold:
            continue
        weight = score / 1000
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([weight, weight])
    adjacency = sparse.csr_matrix(
        (data, (rows, cols)), shape=(n_genes, n_genes), dtype=float
    )
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.zeros_like(degree)
    inverse[degree > 0] = 1 / degree[degree > 0]
    return sparse.diags(inverse) @ adjacency


def propagate(transition, seed_rows, seed_labels, alpha):
    prior = float(np.mean(seed_labels))
    seed = np.full(transition.shape[0], prior)
    seed[seed_rows] = seed_labels
    value = seed.copy()
    for _ in range(200):
        updated = alpha * transition.dot(value) + (1 - alpha) * seed
        updated[seed_rows] = seed_labels
        if np.max(np.abs(updated - value)) < 1e-9:
            return updated
        value = updated
    return value


def expert_oof_and_targets(
    train, validation, test, base_oof, base_val, base_test
):
    train_genes = train["gene_id"].tolist()
    val_genes = validation["gene_id"].tolist()
    test_genes = test["gene_id"].tolist()
    y = train["label"].to_numpy(int)
    folds = list(
        StratifiedKFold(5, shuffle=True, random_state=CV_SEED).split(
            np.zeros(len(y)), y
        )
    )
    terms = graph.load_go_terms()
    go_rows, go_store = [], {}
    for smoothing in [0.5, 1, 3, 10, 30]:
        raw_oof = np.zeros(len(y))
        for fit_idx, hold_idx in folds:
            raw_oof[hold_idx] = go_probability(
                [train_genes[i] for i in fit_idx],
                y[fit_idx],
                [train_genes[i] for i in hold_idx],
                terms,
                smoothing,
            )
        raw_val = go_probability(train_genes, y, val_genes, terms, smoothing)
        raw_test = go_probability(train_genes, y, test_genes, terms, smoothing)
        for weight in np.linspace(0, 1, 41):
            p_oof = (1 - weight) * base_oof + weight * raw_oof
            key = f"go_s{smoothing:g}_w{weight:.3f}"
            go_rows.append({"candidate": key, **ranking(y, p_oof)})
            go_store[key] = (
                p_oof,
                (1 - weight) * base_val + weight * raw_val,
                (1 - weight) * base_test + weight * raw_test,
            )
    go_table = pd.DataFrame(go_rows).sort_values(["auc", "auprc"], ascending=False)
    go_key = go_table.iloc[0]["candidate"]

    all_genes = train_genes + val_genes + test_genes
    index = {gene: i for i, gene in enumerate(all_genes)}
    edges, alias_manifest = graph.load_string_edges(all_genes)
    train_rows = np.array([index[g] for g in train_genes])
    val_rows = np.array([index[g] for g in val_genes])
    test_rows = np.array([index[g] for g in test_genes])
    ppi_rows, ppi_store = [], {}
    for edge_threshold in [150, 300, 400, 500, 700, 900]:
        transition = make_transition(len(all_genes), edges, edge_threshold)
        for alpha in [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]:
            raw_oof = np.zeros(len(y))
            for fit_idx, hold_idx in folds:
                values = propagate(transition, train_rows[fit_idx], y[fit_idx], alpha)
                raw_oof[hold_idx] = values[train_rows[hold_idx]]
            values = propagate(transition, train_rows, y, alpha)
            raw_val, raw_test = values[val_rows], values[test_rows]
            for weight in np.linspace(0, 1, 41):
                p_oof = (1 - weight) * base_oof + weight * raw_oof
                key = f"ppi_t{edge_threshold}_a{alpha:g}_w{weight:.3f}"
                ppi_rows.append({"candidate": key, **ranking(y, p_oof)})
                ppi_store[key] = (
                    p_oof,
                    (1 - weight) * base_val + weight * raw_val,
                    (1 - weight) * base_test + weight * raw_test,
                )
    ppi_table = pd.DataFrame(ppi_rows).sort_values(["auc", "auprc"], ascending=False)
    ppi_key = ppi_table.iloc[0]["candidate"]
    return (
        go_store[go_key],
        ppi_store[ppi_key],
        go_table,
        ppi_table,
        alias_manifest,
        len(edges),
    )


def calibrated_fusion_oof(y, x_oof, x_val, x_test):
    folds = list(
        StratifiedKFold(5, shuffle=True, random_state=CV_SEED + 1).split(
            np.zeros(len(y)), y
        )
    )
    rows, store = [], {}
    for c_value in [0.03, 0.1, 0.3, 1, 3, 10, 30, 100]:
        for class_weight in [None, "balanced"]:
            oof = np.zeros(len(y))
            for fit_idx, hold_idx in folds:
                model = LogisticRegression(
                    C=c_value,
                    class_weight=class_weight,
                    max_iter=10000,
                    random_state=CV_SEED,
                ).fit(x_oof[fit_idx], y[fit_idx])
                oof[hold_idx] = model.predict_proba(x_oof[hold_idx])[:, 1]
            rows.append(
                {
                    "candidate": f"C{c_value:g}_{class_weight or 'none'}",
                    "C": c_value,
                    "class_weight": class_weight or "none",
                    **ranking(y, oof),
                }
            )
            store[rows[-1]["candidate"]] = oof
    table = pd.DataFrame(rows).sort_values(["auc", "auprc"], ascending=False)
    selected = table.iloc[0].to_dict()
    model = LogisticRegression(
        C=selected["C"],
        class_weight=None
        if selected["class_weight"] == "none"
        else selected["class_weight"],
        max_iter=10000,
        random_state=CV_SEED,
    ).fit(x_oof, y)
    return (
        store[selected["candidate"]],
        model.predict_proba(x_val)[:, 1],
        model.predict_proba(x_test)[:, 1],
        model,
        table,
        selected,
    )


def bootstrap_ci(y, p, threshold, n=N_BOOTSTRAP):
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    metrics = {k: [] for k in ["auc", "auprc", "sensitivity", "specificity"]}
    for iteration in range(n):
        idx = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        yy, pp = y[idx], p[idx]
        row = threshold_row(yy, pp, threshold)
        metrics["auc"].append(roc_auc_score(yy, pp))
        metrics["auprc"].append(average_precision_score(yy, pp))
        metrics["sensitivity"].append(row["sensitivity"])
        metrics["specificity"].append(row["specificity"])
        if (iteration + 1) % 2000 == 0:
            print(f"bootstrap {iteration + 1}/{n}", flush=True)
    return {
        key: {
            "estimate": float(
                ranking(y, p)[key]
                if key in {"auc", "auprc"}
                else threshold_row(y, p, threshold)[key]
            ),
            "ci95_lower": float(np.percentile(values, 2.5)),
            "ci95_upper": float(np.percentile(values, 97.5)),
        }
        for key, values in metrics.items()
    }


def evaluate_rules(name, y_oof, p_oof, y_val, p_val, y_test, p_test):
    rows, details = [], {}
    for rule in ["dual_constraint", "sn_priority"]:
        selected, table = select_threshold(y_oof, p_oof, rule)
        threshold = selected["threshold"]
        table.to_csv(OUT / f"{name}_{rule}_OOF_threshold_search.tsv", sep="\t", index=False)
        val_row = threshold_row(y_val, p_val, threshold)
        test_row = threshold_row(y_test, p_test, threshold)
        ci = bootstrap_ci(y_test, p_test, threshold)
        rows.append(
            {
                "model": name,
                "threshold_rule": rule,
                "selected_threshold_from_train_OOF": threshold,
                **{f"OOF_{k}": v for k, v in selected.items() if not isinstance(v, str)},
                **{f"validation_{k}": v for k, v in val_row.items()},
                **{f"test_{k}": v for k, v in test_row.items()},
                "test_auc": ranking(y_test, p_test)["auc"],
                "test_auprc": ranking(y_test, p_test)["auprc"],
            }
        )
        details[rule] = {"OOF_selection": selected, "bootstrap_95ci": ci}
    return rows, details


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    variants, meta, prior_mapping, prior_summary, coverage, status = four.load_matrices()
    matrix, feature_names, n_bio = variants["full6755_plus_ath_prior"]
    raw_labels, matched, missing, parts = build_dataset(meta)
    raw_labels.to_csv(OUT / "raw_strict399_Tos17_N4_labels.tsv", sep="\t", index=False)
    matched.to_csv(OUT / "feature_matched_labels.tsv", sep="\t", index=False)
    missing.to_csv(OUT / "missing_features.tsv", sep="\t", index=False)
    for name, part in parts.items():
        part.to_csv(OUT / f"{name}_labels.tsv", sep="\t", index=False)
    prior_mapping.to_csv(OUT / "ath_prior_mapping.tsv", sep="\t", index=False)

    train, validation, test = parts["train"], parts["validation"], parts["test"]
    y_train = train["label"].to_numpy(np.int8)
    train_rows = train["matrix_row"].to_numpy(int)
    val_rows = validation["matrix_row"].to_numpy(int)
    test_rows = test["matrix_row"].to_numpy(int)
    (
        source_predictions,
        target_predictions,
        fold_rows,
        meta_model,
        deployment_models,
        meta_names,
    ) = library.fit_library(
        matrix[train_rows],
        y_train,
        np.ones(len(train), dtype=np.float32),
        {
            "validation": matrix[val_rows],
            "test": matrix[test_rows],
        },
        n_bio,
    )
    base_method_table = pd.DataFrame(
        [
            {"method": method, **ranking(y_train, probability)}
            for method, probability in source_predictions.items()
        ]
    ).sort_values(["auc", "auprc"], ascending=False)
    base_method = base_method_table.iloc[0]["method"]
    base_oof = source_predictions[base_method]
    base_val = target_predictions["validation"][base_method]
    base_test = target_predictions["test"][base_method]
    np.savez_compressed(
        OUT / "checkpoint_base_predictions.npz",
        base_oof=base_oof,
        base_val=base_val,
        base_test=base_test,
    )
    print(f"saved base OOF checkpoint; method={base_method}", flush=True)

    print("building GO and PPI OOF experts", flush=True)
    go, ppi, go_table, ppi_table, alias_manifest, edge_count = expert_oof_and_targets(
        train, validation, test, base_oof, base_val, base_test
    )
    go_oof, go_val, go_test = go
    ppi_oof, ppi_val, ppi_test = ppi
    np.savez_compressed(
        OUT / "checkpoint_graph_predictions.npz",
        go_oof=go_oof,
        go_val=go_val,
        go_test=go_test,
        ppi_oof=ppi_oof,
        ppi_val=ppi_val,
        ppi_test=ppi_test,
    )
    print("saved GO/PPI OOF checkpoint", flush=True)
    print("fitting OOF calibration stack", flush=True)
    fusion = calibrated_fusion_oof(
        y_train,
        np.column_stack([base_oof, go_oof, ppi_oof]),
        np.column_stack([base_val, go_val, ppi_val]),
        np.column_stack([base_test, go_test, ppi_test]),
    )
    fusion_oof, fusion_val, fusion_test, fusion_model, calibration_table, calibration_selected = fusion
    np.savez_compressed(
        OUT / "checkpoint_fusion_predictions.npz",
        fusion_oof=fusion_oof,
        fusion_val=fusion_val,
        fusion_test=fusion_test,
    )
    print("saved fusion OOF checkpoint", flush=True)

    base_method_table.to_csv(OUT / "base_OOF_method_selection.tsv", sep="\t", index=False)
    go_table.to_csv(OUT / "GO_OOF_expert_selection.tsv", sep="\t", index=False)
    ppi_table.to_csv(OUT / "PPI_OOF_expert_selection.tsv", sep="\t", index=False)
    calibration_table.to_csv(OUT / "fusion_OOF_calibration_selection.tsv", sep="\t", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "base_OOF_fold_scores.tsv", sep="\t", index=False)

    rows, detail = [], {}
    for name, probabilities in {
        "base6755": (base_oof, base_val, base_test),
        "GO_PPI_calibrated_fusion": (fusion_oof, fusion_val, fusion_test),
    }.items():
        result_rows, result_detail = evaluate_rules(
            name,
            y_train,
            probabilities[0],
            validation["label"].to_numpy(int),
            probabilities[1],
            test["label"].to_numpy(int),
            probabilities[2],
        )
        rows.extend(result_rows)
        detail[name] = result_detail
        pd.DataFrame(
            {
                "gene_id": test["gene_id"],
                "label": test["label"],
                "probability": probabilities[2],
            }
        ).to_csv(OUT / f"{name}_locked_test_predictions.tsv", sep="\t", index=False)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "OOF_threshold_rules_test_comparison.tsv", sep="\t", index=False)
    joblib.dump(
        {
            "base_method": base_method,
            "base_meta_model": meta_model,
            "base_deployment_models": deployment_models,
            "base_meta_feature_names": meta_names,
            "fusion_model": fusion_model,
            "feature_names": feature_names,
            "n_bio": n_bio,
        },
        OUT / "models.joblib",
        compress=3,
    )
    summary = {
        "design": "new stratified 80/10/10 split; thresholds selected only from 5-fold training OOF probabilities",
        "split_seed": SPLIT_SEED,
        "counts": {
            name: {
                "total": len(part),
                "essential": int(part["label"].sum()),
                "nonessential": int(part["label"].eq(0).sum()),
            }
            for name, part in parts.items()
        },
        "base_selected_OOF_method": base_method,
        "fusion_calibration_selected": calibration_selected,
        "threshold_rules": detail,
        "bootstrap_replicates": N_BOOTSTRAP,
        "string_edges": edge_count,
        "string_alias_manifest": alias_manifest,
        "prior_summary": prior_summary,
        "feature_coverage": coverage,
        "feature_status": status,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
