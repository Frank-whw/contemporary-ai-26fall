"""当代人工智能实验一：10 类新闻文本分类。

在 data-Project1/ 下运行：uv run python main.py
日志、表格、曲线写入 outputs/，测试集预测写入 predictions.csv。
"""

import copy
import json
import logging
import platform
import sys
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
from matplotlib.figure import Figure
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

SEED = 42
REPEAT_SEEDS = (1, 2)
SMALL_GAP = 0.01  # 验证 macro-F1 差距低于此值视为可能是噪声，追加 REPEAT_SEEDS 重复
VAL_SIZE = 0.2
VEC_HP = dict(lowercase=True, min_df=2)
REPS = ("count", "tf_l2", "tfidf")
LR_MAX_ITER = 1000
C_GRID = (0.1, 1.0, 10.0)
NB_ALPHAS = (0.01, 0.1, 1.0)
WIDTHS = (100, 200, 300)
MLP_HP = dict(
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=200,
    learning_rate_init=1e-3,
    shuffle=True,
)
EPOCHS = 50
CHECK_EPOCHS = 10
TIGHT = {"LR": dict(max_iter=20000, tol=1e-8), "SVC": dict(tol=1e-5)}
CLASSES = np.arange(10)

OUT = Path("outputs")
log = logging.getLogger("exp1")


def load_data():
    train = pd.read_csv("train_data.csv")
    test = pd.read_csv("test_data_unlabeled.csv")
    return (
        train["text"].astype(str).to_numpy(),
        train["target"].to_numpy(),
        test["text"].astype(str).to_numpy(),
    )


def strip_header(doc):
    _, sep, body = doc.partition("\n\n")
    return body if sep else doc


def build_features(fit_docs, *other_docs):
    """词表与 IDF 只由 fit_docs 估计；三种表示共享同一词表，维度一致。"""
    vec = CountVectorizer(**VEC_HP)
    counts = [vec.fit_transform(fit_docs)] + [vec.transform(d) for d in other_docs]
    l2 = TfidfTransformer(use_idf=False).fit(counts[0])
    idf = TfidfTransformer().fit(counts[0])
    reps = {
        "count": counts,
        "tf_l2": [l2.transform(c) for c in counts],
        "tfidf": [idf.transform(c) for c in counts],
    }
    return reps, vec, idf


def make_model(kind, hp, **kw):
    if kind == "MNB":
        return MultinomialNB(alpha=hp, **kw)
    if kind == "LR":
        return LogisticRegression(**{"C": hp, "max_iter": LR_MAX_ITER, **kw})
    if kind == "SVC":
        return SVC(kernel="linear", C=hp, **kw)
    raise ValueError(kind)


def new_mlp(width, seed):
    # MLP._fit 每次调用都会 check_random_state(random_state)：传 int 时第 2 轮起每个 partial_fit 用同一打乱顺序；
    # 传 RandomState 实例则随机流连续、每轮重新打乱，Adam 状态也保留（对照见 outputs/checkpoint_check.csv）。
    return MLPClassifier(
        hidden_layer_sizes=(width,), random_state=np.random.RandomState(seed), **MLP_HP
    )


def n_params(model):
    if isinstance(model, MultinomialNB):
        arrays = [model.feature_log_prob_, model.class_log_prior_]
    elif isinstance(model, MLPClassifier):
        arrays = model.coefs_ + model.intercepts_
    else:
        arrays = [model.coef_, model.intercept_]
    return int(sum(np.prod(a.shape) for a in arrays))


def scores(y, pred, prefix):
    return {
        f"{prefix}_macro_f1": f1_score(y, pred, average="macro"),
        f"{prefix}_acc": accuracy_score(y, pred),
    }


def label(cand):
    kind, hp, rep, docs = cand
    name = {
        "MNB": f"MNB(alpha={hp:g})",
        "LR": f"LR(C={hp:g})",
        "SVC": f"SVC-linear(C={hp:g})",
        "MLP": f"MLP(h={hp})",
    }[kind]
    return f"{name}|{rep}|{docs}"


def to_jsonable(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


class Experiment:
    def __init__(self):
        docs, self.y, test_docs = load_data()
        self.texts = {"raw": docs, "nohdr": np.array([strip_header(d) for d in docs])}
        self.test_texts = {
            "raw": test_docs,
            "nohdr": np.array([strip_header(d) for d in test_docs]),
        }
        self.splits, self.feats, self.runs, self.val_preds = {}, {}, {}, {}
        self.curves, self.comparisons, self.decisions = [], [], []
        self.info = {}

    def split(self, seed):
        if seed not in self.splits:
            idx = np.arange(len(self.y))
            self.splits[seed] = train_test_split(
                idx, test_size=VAL_SIZE, stratify=self.y, random_state=seed
            )
        return self.splits[seed]

    def features(self, seed, docs):
        if (seed, docs) not in self.feats:
            tr, va = self.split(seed)
            reps, vec, idf = build_features(self.texts[docs][tr], self.texts[docs][va])
            self.feats[seed, docs] = dict(
                reps=reps, vec=vec, idf=idf, dim=len(vec.vocabulary_), ytr=self.y[tr], yva=self.y[va]
            )
            log.info(
                "features seed=%d docs=%s: train=%d val=%d dim=%d",
                seed, docs, len(tr), len(va), len(vec.vocabulary_),
            )
        return self.feats[seed, docs]

    def describe_data(self):
        tr, va = self.split(SEED)
        counts = pd.DataFrame(
            {
                "train_fold": np.bincount(self.y[tr], minlength=10),
                "val_fold": np.bincount(self.y[va], minlength=10),
            }
        )
        counts.index.name = "class"
        counts.to_csv(OUT / "split_counts.csv")
        f = self.features(SEED, "raw")
        cnt, tfidf = f["reps"]["count"][0], f["reps"]["tfidf"][0]
        tokens = np.asarray(cnt.sum(axis=1)).ravel()
        nnz = tfidf.getnnz(axis=1)
        vocab = f["vec"].get_feature_names_out()
        term_total = np.asarray(cnt.sum(axis=0)).ravel()
        top = np.argsort(term_total)[::-1][:5]
        idf = f["idf"].idf_
        train_raw = self.texts["raw"][tr]
        present = (cnt > 0).tocsc()
        coo = cnt.tocoo(copy=True)
        biggest = np.argsort(coo.data)[::-1][:3]
        extreme = [
            dict(
                term=vocab[coo.col[k]],
                count_in_doc=int(coo.data[k]),
                doc_tokens=int(tokens[coo.row[k]]),
                doc_class=int(f["ytr"][coo.row[k]]),
                term_total_count=int(term_total[coo.col[k]]),
                docs_containing_term=int(present[:, coo.col[k]].nnz),
                classes_of_docs_containing_term=np.bincount(
                    f["ytr"][present[:, coo.col[k]].indices], minlength=10
                ).tolist(),
            )
            for k in biggest
        ]
        row_norm = np.sqrt(np.asarray(cnt.multiply(cnt).sum(axis=1)).ravel())
        diff = np.vstack(
            [np.asarray(tfidf[f["ytr"] == k].mean(axis=0)).ravel() for k in CLASSES]
        ) - np.asarray(tfidf.mean(axis=0)).ravel()
        class_terms = {int(k): vocab[np.argsort(diff[k])[::-1][:6]].tolist() for k in CLASSES}
        self.info["data"] = dict(
            n_train_csv=len(self.y),
            n_test=len(self.test_texts["raw"]),
            n_train_fold=len(tr),
            n_val_fold=len(va),
            train_fold_class_counts=counts["train_fold"].tolist(),
            val_fold_class_counts=counts["val_fold"].tolist(),
            dim_seed42=f["dim"],
            tokens_per_doc_train_fold=dict(
                min=int(tokens.min()), median=float(np.median(tokens)), max=int(tokens.max())
            ),
            nnz_per_doc_train_fold=dict(mean=float(nnz.mean()), median=float(np.median(nnz))),
            density_train_fold=float(tfidf.nnz / np.prod(tfidf.shape)),
            top5_terms_train_fold=[
                dict(
                    term=vocab[i],
                    token_share=float(term_total[i] / term_total.sum()),
                    idf=float(idf[i]),
                )
                for i in top
            ],
            idf_median=float(np.median(idf)),
            idf_max=float(idf.max()),
            largest_single_doc_counts_train_fold=extreme,
            count_row_l2_norm_train_fold=dict(
                min=float(row_norm.min()), median=float(np.median(row_norm)), max=float(row_norm.max())
            ),
            class_top_terms_train_fold=class_terms,
            train_fold_docs_without_blank_line=int(sum("\n\n" not in d for d in train_raw)),
        )
        log.info("data: %s", json.dumps(self.info["data"], ensure_ascii=False))

    def check_checkpointing(self):
        """验证逐 epoch partial_fit 记录方式与一次不间断 fit 的训练轨迹是否一致。"""
        f = self.features(SEED, "raw")
        X_tr, X_va = f["reps"]["tfidf"]
        ytr, yva = f["ytr"], f["yva"]
        w = WIDTHS[0]
        ref, other_seed = (
            MLPClassifier(hidden_layer_sizes=(w,), random_state=s, max_iter=CHECK_EPOCHS, **MLP_HP)
            for s in (SEED, SEED + 1)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            ref.fit(X_tr, ytr)
            other_seed.fit(X_tr, ytr)

            pf_rs = new_mlp(w, SEED)
            for _ in range(CHECK_EPOCHS):
                pf_rs.partial_fit(X_tr, ytr, classes=CLASSES)

            pf_int = MLPClassifier(hidden_layer_sizes=(w,), random_state=SEED, **MLP_HP)
            for _ in range(CHECK_EPOCHS):
                pf_int.partial_fit(X_tr, ytr, classes=CLASSES)

            ws = MLPClassifier(
                hidden_layer_sizes=(w,), random_state=SEED, warm_start=True, max_iter=1, **MLP_HP
            )
            for _ in range(CHECK_EPOCHS):
                ws.fit(X_tr, ytr)

        rows = []
        variants = {
            f"fit(max_iter={CHECK_EPOCHS}, random_state=42) [reference]": ref,
            "partial_fit + RandomState(42) [used]": pf_rs,
            "partial_fit + random_state=42 (int)": pf_int,
            "warm_start=True + fit(max_iter=1)": ws,
            f"fit(max_iter={CHECK_EPOCHS}, random_state=43) [seed-noise scale]": other_seed,
        }
        for name, m in variants.items():
            diff = np.abs(np.array(m.loss_curve_) - np.array(ref.loss_curve_))
            rows.append(
                dict(
                    variant=name,
                    width=w,
                    epochs=len(m.loss_curve_),
                    epoch1_loss_diff_vs_ref=float(diff[0]),
                    max_abs_loss_diff_vs_ref=float(diff.max()),
                    final_train_loss=m.loss_curve_[-1],
                    **scores(yva, m.predict(X_va), "val"),
                )
            )
        df = pd.DataFrame(rows)
        df.to_csv(OUT / "checkpoint_check.csv", index=False)
        log.info("checkpoint check:\n%s", df.to_string(index=False))
        # fit 在上一轮排列上继续打乱，partial_fit 每次从 arange 打乱：随机流相同、排列不同，故只要求首个 epoch 逐位一致
        assert rows[1]["epoch1_loss_diff_vs_ref"] == 0.0, "partial_fit 初始化/首轮与 fit 不一致"

    def run(self, cand, seed, variant="default"):
        key = (cand, seed, variant)
        if key not in self.runs:
            fit = self._fit_mlp if cand[0] == "MLP" else self._fit_closed
            row = dict(
                model=label(cand),
                kind=cand[0],
                hp=cand[1],
                rep=cand[2],
                docs=cand[3],
                seed=seed,
                variant=variant,
                **fit(cand, seed, variant),
            )
            self.runs[key] = row
            log.info(
                "run %-32s seed=%d %-7s | train F1 %.4f | val F1 %.4f acc %.4f | params %d | %.1fs%s",
                row["model"], seed, variant, row["train_macro_f1"], row["val_macro_f1"],
                row["val_acc"], row["n_params"], row["fit_s"],
                f" | best_epoch {row['best_epoch']} last-epoch val F1 {row['last_val_macro_f1']:.4f}"
                if cand[0] == "MLP" else "",
            )
        return self.runs[key]

    def _fit_closed(self, cand, seed, variant):
        kind, hp, rep, docs = cand
        f = self.features(seed, docs)
        X_tr, X_va = f["reps"][rep]
        model = make_model(kind, hp, **(TIGHT[kind] if variant == "tight" else {}))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            t0 = time.perf_counter()
            model.fit(X_tr, f["ytr"])
            fit_s = time.perf_counter() - t0
        pred = model.predict(X_va)
        self.val_preds[cand, seed, variant] = pred
        return dict(
            dim=f["dim"],
            **scores(f["ytr"], model.predict(X_tr), "train"),
            **scores(f["yva"], pred, "val"),
            n_params=n_params(model),
            fit_s=fit_s,
            n_iter=int(np.max(model.n_iter_)) if hasattr(model, "n_iter_") else None,
            converged=not any(issubclass(w.category, ConvergenceWarning) for w in caught),
            n_support=int(model.n_support_.sum()) if kind == "SVC" else None,
        )

    def _fit_mlp(self, cand, seed, variant):
        _, width, rep, docs = cand
        f = self.features(seed, docs)
        X_tr, X_va = f["reps"][rep]
        ytr, yva = f["ytr"], f["yva"]
        model = new_mlp(width, seed)
        best_f1, best_ep, best_w, fit_s, curve = -1.0, 0, None, 0.0, []
        for ep in range(1, EPOCHS + 1):
            t0 = time.perf_counter()
            model.partial_fit(X_tr, ytr, classes=CLASSES)
            fit_s += time.perf_counter() - t0
            p_tr, p_va = model.predict_proba(X_tr), model.predict_proba(X_va)
            rec = dict(
                run="split",
                width=width,
                seed=seed,
                docs=docs,
                epoch=ep,
                train_loss=model.loss_curve_[-1],
                train_logloss=log_loss(ytr, p_tr, labels=CLASSES),
                **scores(ytr, p_tr.argmax(1), "train"),
                val_logloss=log_loss(yva, p_va, labels=CLASSES),
                **scores(yva, p_va.argmax(1), "val"),
            )
            curve.append(rec)
            if rec["val_macro_f1"] > best_f1:
                best_f1, best_ep = rec["val_macro_f1"], ep
                best_w = copy.deepcopy((model.coefs_, model.intercepts_))
        assert list(model.classes_) == list(CLASSES)
        assert model.loss_curve_ == [r["train_loss"] for r in curve]
        self.curves += curve
        model.coefs_, model.intercepts_ = best_w
        pred = model.predict(X_va)
        assert np.isclose(f1_score(yva, pred, average="macro"), best_f1)
        self.val_preds[cand, seed, variant] = pred
        best, last = curve[best_ep - 1], curve[-1]
        return dict(
            dim=f["dim"],
            train_macro_f1=best["train_macro_f1"],
            train_acc=best["train_acc"],
            val_macro_f1=best["val_macro_f1"],
            val_acc=best["val_acc"],
            n_params=n_params(model),
            fit_s=fit_s,
            n_iter=len(model.loss_curve_),
            converged=None,
            best_epoch=best_ep,
            train_loss_at_best=best["train_loss"],
            val_logloss_at_best=best["val_logloss"],
            last_val_macro_f1=last["val_macro_f1"],
            last_val_acc=last["val_acc"],
            last_train_loss=last["train_loss"],
            last_val_logloss=last["val_logloss"],
            min_val_logloss=min(r["val_logloss"] for r in curve),
            min_val_logloss_epoch=int(np.argmin([r["val_logloss"] for r in curve]) + 1),
        )

    def compare(self, exp, cands, use_complexity=False):
        """先用 SEED 比较；若第一、二名差距 < SMALL_GAP，则差距内的候选追加 REPEAT_SEEDS，按均值排序。
        use_complexity 时采用一倍标准误规则：均值不低于 (最佳均值 - 最佳标准误) 的候选中取参数最少者。"""
        first = {c: self.run(c, SEED)["val_macro_f1"] for c in cands}
        order = sorted(cands, key=first.get, reverse=True)
        top = first[order[0]]
        close = [c for c in order if top - first[c] < SMALL_GAP]
        repeated = len(close) > 1
        summary = {}
        for c in cands:
            seeds = (SEED, *REPEAT_SEEDS) if repeated and c in close else (SEED,)
            rs = [self.run(c, s) for s in seeds]
            f1 = np.array([r["val_macro_f1"] for r in rs])
            acc = np.array([r["val_acc"] for r in rs])
            summary[c] = dict(
                exp=exp,
                model=label(c),
                seeds=" ".join(map(str, seeds)),
                n_seeds=len(seeds),
                val_macro_f1_seed42=first[c],
                val_macro_f1_mean=f1.mean(),
                val_macro_f1_std=f1.std(ddof=1) if len(f1) > 1 else np.nan,
                val_acc_mean=acc.mean(),
                val_macro_f1_by_seed=" ".join(f"{v:.4f}" for v in f1),
                n_params=rs[0]["n_params"],
            )
        best = max(close, key=lambda c: summary[c]["val_macro_f1_mean"])
        chosen, rule, threshold = best, "max val macro-F1 (mean over seeds if repeated)", None
        if use_complexity and repeated:
            s = summary[best]
            se = s["val_macro_f1_std"] / np.sqrt(s["n_seeds"])
            threshold = s["val_macro_f1_mean"] - se
            ok = [c for c in close if summary[c]["val_macro_f1_mean"] >= threshold]
            chosen = min(ok, key=lambda c: summary[c]["n_params"])
            rule = f"1-SE rule: mean >= {s['val_macro_f1_mean']:.4f} - {se:.4f}, then fewest params"
        for c in cands:
            summary[c]["chosen"] = c == chosen
        self.comparisons += summary.values()
        decision = dict(
            exp=exp,
            gap_top2_seed42=top - first[order[1]],
            repeated=repeated,
            repeated_models=[label(c) for c in close] if repeated else [],
            best_by_mean=label(best),
            chosen=label(chosen),
            rule=rule,
            threshold=threshold,
        )
        self.decisions.append(decision)
        log.info(
            "compare [%s]\n%s\n-> %s",
            exp,
            pd.DataFrame(list(summary.values())).drop(columns="exp").to_string(index=False),
            json.dumps(decision, ensure_ascii=False, default=to_jsonable),
        )
        return chosen

    def q1_robustness(self):
        best = {}
        for rep in REPS:
            rows = [self.run(("LR", c, rep, "raw"), SEED) for c in C_GRID]
            top = max(rows, key=lambda r: r["val_macro_f1"])
            best[rep] = dict(best_C=top["hp"], val_macro_f1=top["val_macro_f1"])
        self.info["q1_lr_best_C_per_rep_seed42"] = best
        log.info("Q1 robustness (LR best C per representation, seed 42): %s", best)

    def optimization_check(self, cand):
        base = self.run(cand, SEED)
        tight = self.run(cand, SEED, "tight")
        agree = float(
            np.mean(self.val_preds[cand, SEED, "default"] == self.val_preds[cand, SEED, "tight"])
        )
        out = dict(
            tight_hp=TIGHT[cand[0]],
            val_macro_f1_default=base["val_macro_f1"],
            val_macro_f1_tight=tight["val_macro_f1"],
            n_iter_default=base["n_iter"],
            n_iter_tight=tight["n_iter"],
            val_pred_agreement=agree,
        )
        self.info.setdefault("optimization_check", {})[label(cand)] = out
        log.info("optimization check %s: %s", label(cand), out)

    def mlp_summary(self):
        rows = [
            r for (c, _, v), r in self.runs.items() if c[0] == "MLP" and c[3] == "raw" and v == "default"
        ]
        cols = [
            "hp", "seed", "best_epoch", "val_macro_f1", "last_val_macro_f1", "train_macro_f1",
            "train_loss_at_best", "last_train_loss", "min_val_logloss_epoch", "min_val_logloss",
            "last_val_logloss", "n_params",
        ]
        df = pd.DataFrame(rows)[cols].rename(columns={"hp": "width", "val_macro_f1": "best_val_macro_f1"})
        df["best_minus_last"] = df["best_val_macro_f1"] - df["last_val_macro_f1"]
        df["val_logloss_rise_after_min"] = df["last_val_logloss"] - df["min_val_logloss"]
        df = df.sort_values(["width", "seed"])
        df.to_csv(OUT / "mlp_summary.csv", index=False)
        means = df.groupby("width")[
            ["best_epoch", "best_val_macro_f1", "last_val_macro_f1", "best_minus_last"]
        ].mean()
        self.info["mlp_width_means"] = means.reset_index().to_dict("records")
        log.info("MLP best vs last epoch:\n%s\nmeans:\n%s", df.to_string(index=False), means.to_string())

    def paired(self, final, others):
        """同一种子下 final 与其他候选的验证 macro-F1 差；MLP 另给出用最后一轮代替最佳迭代时的差。"""
        out = []
        for o in others:
            seeds = sorted(
                s for (c, s, v) in self.runs if c == final and v == "default" and (o, s, "default") in self.runs
            )
            fr = [self.runs[final, s, "default"] for s in seeds]
            orr = [self.runs[o, s, "default"] for s in seeds]
            row = dict(
                final=label(final),
                other=label(o),
                seeds=seeds,
                diff_by_seed=[a["val_macro_f1"] - b["val_macro_f1"] for a, b in zip(fr, orr)],
            )
            row["mean_diff"] = float(np.mean(row["diff_by_seed"]))
            if final[0] == "MLP":
                row["diff_by_seed_using_last_epoch"] = [
                    a["last_val_macro_f1"] - b["val_macro_f1"] for a, b in zip(fr, orr)
                ]
                row["mean_diff_using_last_epoch"] = float(np.mean(row["diff_by_seed_using_last_epoch"]))
            out.append(row)
        self.info["final_vs_others_paired"] = out
        log.info("paired differences: %s", json.dumps(out, default=to_jsonable))

    def derived(self, lr, final):
        """报告里引用的差值、比值等派生数，统一在此计算并记录。"""

        def run(c, s=SEED):
            return self.runs[c, s, "default"]

        out = {}
        for kind in ("LR", "MNB"):
            v = {r: run((kind, 1.0, r, "raw"))["val_macro_f1"] for r in REPS}
            out[f"q1_{kind}_hp1_delta_pp"] = dict(
                tfidf_minus_count=100 * (v["tfidf"] - v["count"]),
                tf_l2_minus_count=100 * (v["tf_l2"] - v["count"]),
                tfidf_minus_tf_l2=100 * (v["tfidf"] - v["tf_l2"]),
            )
        best = self.info["q1_lr_best_C_per_rep_seed42"]
        out["q1_LR_bestC_tfidf_minus_count_pp"] = 100 * (
            best["tfidf"]["val_macro_f1"] - best["count"]["val_macro_f1"]
        )
        c_runs = [run(("LR", c, "tfidf", "raw"))["val_macro_f1"] for c in C_GRID]
        out["q2_LR_val_f1_range_over_C_pp"] = 100 * (max(c_runs) - min(c_runs))
        out["final_over_lr_param_ratio"] = run(final)["n_params"] / run(lr)["n_params"]
        out["final_over_lr_fit_time_ratio_seed42"] = run(final)["fit_s"] / run(lr)["fit_s"]
        if final[0] == "MLP":
            c = pd.DataFrame(self.curves)
            c = c[(c["run"] == "split") & (c["docs"] == "raw")]
            out["mlp_min_train_macro_f1_at_epoch5"] = c.loc[c["epoch"] == 5, "train_macro_f1"].min()
            s42 = c[c["seed"] == SEED]
            out["mlp_seed42_train_loss_epoch5_by_width"] = (
                s42[s42["epoch"] == 5].set_index("width")["train_loss"].to_dict()
            )
            g = s42[s42["width"] == final[1]]
            out["final_width_seed42_train_loss_epoch1_epoch50"] = [
                g["train_loss"].iloc[0], g["train_loss"].iloc[-1]
            ]
            tail = g[g["epoch"] >= 6]["val_macro_f1"]
            out["final_width_seed42_val_f1_range_epoch_ge6"] = [tail.min(), tail.max()]
        self.info["derived"] = out
        log.info("derived numbers: %s", json.dumps(out, default=to_jsonable))

    def per_class(self, cands, final):
        f = self.features(SEED, "raw")
        yva = f["yva"]
        table = pd.DataFrame({"class": CLASSES, "support": np.bincount(yva, minlength=10)})
        for c in cands:
            p, r, f1, _ = precision_recall_fscore_support(
                yva, self.val_preds[c, SEED, "default"], labels=CLASSES, zero_division=0
            )
            name = label(c).split("|")[0]
            table[f"{name}_f1"], table[f"{name}_recall"] = f1, r
        table.to_csv(OUT / "per_class.csv", index=False)
        cm = confusion_matrix(yva, self.val_preds[final, SEED, "default"], labels=CLASSES)
        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).rename_axis("true\\pred").to_csv(
            OUT / "confusion_final_seed42.csv"
        )
        off = [(int(i), int(j), int(cm[i, j])) for i in CLASSES for j in CLASSES if i != j and cm[i, j]]
        off.sort(key=lambda t: -t[2])
        self.info["top_confusions_final_seed42"] = [
            dict(true=i, pred=j, count=n) for i, j, n in off[:6]
        ]
        log.info("per-class (seed 42 val):\n%s", table.to_string(index=False))
        log.info("top confusions (true, pred, count): %s", off[:6])

    def final_fit(self, cand):
        kind, hp, rep, docs = cand
        reps, vec, _ = build_features(self.texts[docs], self.test_texts[docs])
        X_all, X_test = reps[rep]
        info = dict(model=label(cand), n_fit=X_all.shape[0], dim_full_train=len(vec.vocabulary_))
        t0 = time.perf_counter()
        if kind == "MLP":
            seeds = sorted(s for (c, s, v) in self.runs if c == cand and v == "default")
            best_epochs = [self.runs[cand, s, "default"]["best_epoch"] for s in seeds]
            epochs = int(round(np.median(best_epochs)))
            model = new_mlp(hp, SEED)
            for ep in range(1, epochs + 1):
                model.partial_fit(X_all, self.y, classes=CLASSES)
                self.curves.append(
                    dict(run="final_full_train", width=hp, seed=SEED, docs=docs, epoch=ep,
                         train_loss=model.loss_curve_[-1])
                )
            info.update(best_epoch_by_seed=dict(zip(seeds, best_epochs)), epochs=epochs)
        else:
            model = make_model(kind, hp).fit(X_all, self.y)
        info["fit_s"] = time.perf_counter() - t0
        info["n_params"] = n_params(model)
        pred = model.predict(X_test).astype(int)
        pd.Series(pred).to_csv("predictions.csv", index=False, header=False)
        back = pd.read_csv("predictions.csv", header=None)[0]
        assert len(back) == len(self.test_texts["raw"]) == X_test.shape[0]
        assert back.between(0, 9).all() and (back.to_numpy() == pred).all()
        info.update(test_rows=len(back), pred_class_counts=np.bincount(back, minlength=10).tolist())
        self.info["final"] = info
        log.info("final fit: %s", info)

    def save(self):
        pd.DataFrame(list(self.runs.values())).to_csv(OUT / "results.csv", index=False)
        pd.DataFrame(self.curves).to_csv(OUT / "mlp_curves.csv", index=False)
        pd.DataFrame(self.comparisons).to_csv(OUT / "comparisons.csv", index=False)
        config = dict(
            SEED=SEED, REPEAT_SEEDS=REPEAT_SEEDS, SMALL_GAP=SMALL_GAP, VAL_SIZE=VAL_SIZE,
            VEC_HP=VEC_HP, TFIDF="TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)",
            LR_MAX_ITER=LR_MAX_ITER, C_GRID=C_GRID, NB_ALPHAS=NB_ALPHAS, WIDTHS=WIDTHS,
            MLP_HP=MLP_HP, EPOCHS=EPOCHS, TIGHT=TIGHT,
        )
        versions = dict(
            python=sys.version.split()[0], platform=platform.platform(), sklearn=sklearn.__version__,
            numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
            matplotlib=matplotlib.__version__,
        )
        payload = dict(config=config, versions=versions, decisions=self.decisions, **self.info)
        (OUT / "summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=to_jsonable), encoding="utf-8"
        )


def plot_curves(csv_path=OUT / "mlp_curves.csv", png_path=OUT / "loss_curve.png"):
    df = pd.read_csv(csv_path)
    df = df[(df["run"] == "split") & (df["docs"] == "raw")]
    colors = dict(zip(WIDTHS, ("tab:blue", "tab:orange", "tab:green")))
    fig = Figure(figsize=(11, 2.9))
    ax_loss, ax_f1, ax_rep = fig.subplots(1, 3)

    def mark_best(ax, g, color):
        best = g.loc[g["val_macro_f1"].idxmax()]
        ax.scatter(best["epoch"], best["val_macro_f1"], marker="*", s=110, color=color,
                   edgecolor="k", linewidth=0.5, zorder=3)

    for width, g in df[df["seed"] == SEED].groupby("width"):
        c = colors[width]
        ax_loss.plot(g["epoch"], g["train_loss"], lw=1.2, color=c, label=f"h={width}")
        ax_loss.plot(g["epoch"], g["val_logloss"], "--", lw=1.2, color=c)
        ax_f1.plot(g["epoch"], g["val_macro_f1"], lw=1.0, marker=".", ms=3, color=c, label=f"h={width}")
        mark_best(ax_f1, g, c)
    rep = df[(df["seed"] != SEED) & (df["epoch"] >= 3)]
    for (width, seed), g in rep.groupby(["width", "seed"]):
        c = colors[width]
        first = seed == REPEAT_SEEDS[0]
        ax_rep.plot(g["epoch"], g["val_macro_f1"], "-" if first else "--", lw=1.0, color=c,
                    label=f"h={width}" if first else None)
        mark_best(ax_rep, g, c)
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("cross-entropy (log scale)")
    ax_loss.set_title(f"seed={SEED}: train loss_curve_ (solid) / val log-loss (dashed)", fontsize=9)
    ax_loss.legend(fontsize=7, loc="center right")
    ax_f1.set_ylabel("val macro-F1")
    ax_f1.set_title(f"seed={SEED}: val macro-F1 (* = best epoch)", fontsize=9)
    ax_f1.legend(fontsize=7)
    ax_rep.set_title(
        f"seed {REPEAT_SEEDS[0]} solid / {REPEAT_SEEDS[1]} dashed, epoch>=3 (* = best)", fontsize=9
    )
    ax_rep.set_ylabel("val macro-F1")
    ax_rep.legend(fontsize=7, ncol=3, loc="lower center")
    for ax in (ax_loss, ax_f1, ax_rep):
        ax.set_xlabel("epoch", fontsize=8)
        ax.yaxis.label.set_size(8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("MLP on TF-IDF, raw per-epoch values (no smoothing)", fontsize=9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)


def main():
    OUT.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(OUT / "run.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    t0 = time.perf_counter()
    exp = Experiment()
    exp.describe_data()
    exp.check_checkpointing()

    for kind in ("LR", "MNB"):
        exp.compare(f"Q1 representation, {kind} hp=1 fixed", [(kind, 1.0, r, "raw") for r in REPS])
    exp.q1_robustness()
    exp.save()

    nb = exp.compare("Q2 MNB alpha", [("MNB", a, "tfidf", "raw") for a in NB_ALPHAS])
    lr = exp.compare("Q2 LR C", [("LR", c, "tfidf", "raw") for c in C_GRID])
    svc = exp.compare("Q2 SVC C", [("SVC", c, "tfidf", "raw") for c in C_GRID])
    exp.optimization_check(lr)
    exp.optimization_check(svc)
    exp.save()

    mlp = exp.compare(
        "Q3 MLP width", [("MLP", w, "tfidf", "raw") for w in WIDTHS], use_complexity=True
    )
    exp.mlp_summary()
    exp.save()

    cands = [nb, lr, svc, mlp]
    final = exp.compare("Final model selection", cands, use_complexity=True)
    exp.paired(final, [c for c in cands if c != final])
    exp.derived(lr, final)
    exp.per_class(cands, final)
    exp.save()

    kind, hp, rep, _ = final
    exp.compare("Ablation: strip e-mail header", [final, (kind, hp, rep, "nohdr")])
    log.info("ablation is post-hoc only; final model stays %s", label(final))

    exp.final_fit(final)
    exp.info["total_runtime_s"] = time.perf_counter() - t0
    exp.save()
    plot_curves()
    log.info("done in %.1fs", exp.info["total_runtime_s"])


if __name__ == "__main__":
    main()
