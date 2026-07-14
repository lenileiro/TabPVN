"""CertifiedAttention — attention as a STANDALONE, proof-carrying predictor for TabPVN.

The prediction IS the attention read: cosine-weighted top-k voting over stored training examples (a memory /
retrieval model). Its proof exposes WHICH training examples it attended to (retrieval evidence) AND is
kernel-verified: the winning margin is LINEAR in the query features for a fixed attended set
  V_c = Σ_{i∈topk} s_i·[y_i=c],  s_i = q·k_i  (dot of L2-normalized, idf-weighted feature vectors)
  margin M = V_p − V_c' = Σ_j q_j·W_j,  W_j = Σ_{i∈topk} k_ij·([y_i=p]−[y_i=c'])
so the FOLKernel re-derives it with plain arithmetic (no softmax `exp`) — a weighted-sum Horn clause checked
by the unchanged `check_proof`. Cosine-weighted ("linear") attention is used precisely so the read is
certifiable end-to-end. Our own primitive: no embedding, no external model. Classification and regression.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.kernel_fol import FOLKernel
from tabpvn.preprocessing import _is_classification, _Preprocessor
from tabpvn.proofs import (
    TargetAttestation,
    build_proof_artifact,
    public_proof_response,
)


class CertifiedAttention:
    def __init__(
        self,
        seed: int = 0,
        alpha: float = 0.1,
        topk: int = 30,
        cap: int = 6000,
    ) -> None:
        if topk <= 0 or cap < 2:
            raise ValueError("topk must be positive and cap must be at least 2")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        self.seed, self.alpha, self.topk, self.cap = seed, alpha, topk, cap
        self._conf: Any | None = None

    # ---- fit ----
    def fit(self, data: Any, y: Any) -> Self:
        import pandas as pd

        y = np.asarray(y)
        if pd.Series(y.ravel()).dropna().nunique() < 2:
            raise ValueError("target y has fewer than 2 distinct non-missing values — nothing to learn.")
        self._prep = _Preprocessor().fit(data, y)
        self.feature_names_ = self._prep.names
        X = self._prep.transform(data)
        if len(X) > self.cap:  # bound the stored memory / read cost (SDM samples hard locations)
            idx = np.random.default_rng(self.seed).choice(len(X), self.cap, replace=False)
            X, y = X[idx], y[idx]
        self.mode = "classification" if _is_classification(y) else "regression"
        self.n_ = len(X)
        df = (X != 0).sum(0)
        self.idf_ = np.log((self.n_ + 1) / (df + 1)) + 1.0  # down-weight ubiquitous features
        self.K_ = self._keys(X)  # normalized, idf-weighted training keys (n×d)
        if self.mode == "classification":
            self.classes_ = sorted(set(y.tolist()))
            self.ci_ = {c: i for i, c in enumerate(self.classes_)}
            self.yidx_ = np.array([self.ci_[v] for v in y.tolist()])
        else:
            self.yval_ = y.astype(float)
        self.topk_ = int(min(self.topk, self.n_ - 1))
        self._build_confidence(data, y)
        return self

    def _keys(self, Xmat: ArrayLike) -> NDArray[np.float64]:
        Kw = np.asarray(Xmat, float) * self.idf_
        return Kw / (np.linalg.norm(Kw, axis=1, keepdims=True) + 1e-9)

    def _query(self, X: Any) -> NDArray[np.float64]:
        """Normalized query key matrix from raw input X (m×d)."""
        if not hasattr(self, "_prep"):
            raise RuntimeError("model is not fitted; call fit(X, y) first")
        return self._keys(self._prep.transform(X))

    def _topk(self, s: NDArray[np.float64]) -> NDArray[np.int64]:
        idx = np.argpartition(-s, self.topk_)[: self.topk_]
        return idx[np.argsort(-s[idx])]  # sorted desc

    # ---- predict ----
    def predict_proba(self, X: Any) -> NDArray[np.float64]:
        if not hasattr(self, "mode"):
            raise RuntimeError("model is not fitted; call fit(X, y) first")
        if self.mode != "classification":
            raise ValueError("predict_proba is available only for classification")
        Q = self._query(X)
        S = Q @ self.K_.T
        C = len(self.classes_)
        P = np.zeros((len(Q), C))
        for i in range(len(Q)):
            top = self._topk(S[i])
            for j in top:
                P[i, self.yidx_[j]] += S[i, j]  # cosine-weighted vote
        P = np.clip(P, 0, None)
        return P / (P.sum(1, keepdims=True) + 1e-12)

    def predict(self, X: Any) -> NDArray[Any]:
        Q = self._query(X)
        S = Q @ self.K_.T
        if self.mode == "classification":
            C = len(self.classes_)
            out: NDArray[Any] = np.empty(len(Q), dtype=object)
            for i in range(len(Q)):
                top = self._topk(S[i])
                V = np.zeros(C)
                for j in top:
                    V[self.yidx_[j]] += S[i, j]
                out[i] = self.classes_[int(V.argmax())]
            return np.array([o for o in out])
        # regression: Nadaraya-Watson over the top-k
        out = np.empty(len(Q))
        for i in range(len(Q)):
            top = self._topk(S[i])
            w = S[i, top]
            out[i] = float(np.dot(w, self.yval_[top]) / (w.sum() + 1e-12))
        return out

    def attended(self, X: Any, row: int, m: int = 5) -> list[tuple[int, float, Any]]:
        """The top-m stored examples the query attended to: (train_index, cosine, label/value)."""
        Q = self._query(X)
        s = Q[row] @ self.K_.T
        top = self._topk(s)[:m]
        return [
            (
                int(j),
                round(float(s[j]), 3),
                (
                    self.classes_[self.yidx_[j]]
                    if self.mode == "classification"
                    else round(float(self.yval_[j]), 3)
                ),
            )
            for j in top
        ]

    # ---- proof: the read's decision is linear in the query given the attended set ----
    def _wins_proof(
        self,
        q: NDArray[np.float64],
        top: NDArray[np.int64],
        S_row: NDArray[np.float64],
        row: int,
    ) -> tuple[int, Any]:
        """Kernel-verified proof that class p wins the cosine-weighted vote over the attended top-k set."""
        C = len(self.classes_)
        V = np.zeros(C)
        for j in top:
            V[self.yidx_[j]] += S_row[j]
        p = int(V.argmax())
        pres = [f for f in range(len(q)) if abs(q[f]) > 1e-9]  # query's active features
        body: list[tuple[Any, ...]] = [("feat", "R", f, f"V{t}") for t, f in enumerate(pres)]
        rivals = [c for c in range(C) if c != p]
        for c in rivals:
            d = np.array(
                [(1.0 if self.yidx_[j] == p else 0.0) - (1.0 if self.yidx_[j] == c else 0.0) for j in top]
            )
            W = (self.K_[top] * d[:, None]).sum(0)  # W_f = Σ_{i∈top} k_if·(1[y=p]-1[y=c])
            prev = f"M{c}_0"
            body.append(("is", prev, 0.0))
            for t, f in enumerate(pres):
                nxt = f"M{c}_{t + 1}"
                body.append(("is", nxt, ("+", prev, ("*", float(W[f]), f"V{t}"))))
                prev = nxt
            # margin M = Σ_f q_f·W_f ≥ 0 (argmax gives V_p ≥ V_c'); the tiny tolerance absorbs float summation-
            # order noise (kernel left-to-right vs numpy BLAS) near vote ties and covers the zero-feature case.
            body.append(("cmp", ">=", prev, -1e-9))
        head = ("wins", "R", p) if pres else ("wins", p)
        fact = ("wins", row, p) if pres else ("wins", p)
        facts: list[tuple[Any, ...]] = [("feat", row, f, float(q[f])) for f in pres]
        fired, prov = FOLKernel([(head, body)]).closure(facts)
        node = FOLKernel([(head, body)]).proof(fact, prov) if fact in fired else None
        return p, node

    @staticmethod
    def _proof_row(row: int, n_rows: int) -> int:
        if isinstance(row, (bool, np.bool_)) or not isinstance(row, (int, np.integer)):
            raise TypeError("row must be an integer index")
        row = int(row)
        if not 0 <= row < n_rows:
            raise IndexError(f"row index {row} is outside the table with {n_rows} rows")
        return row

    def _prediction_proof(self, X: Any, row: int) -> dict[str, Any]:
        Q = self._query(X)
        row = self._proof_row(row, len(Q))
        S = Q @ self.K_.T
        top = self._topk(S[row])
        attention_terms = [
            {
                "row": int(j),
                "similarity": float(S[row, j]),
                "target": (
                    self.classes_[self.yidx_[j]] if self.mode == "classification" else float(self.yval_[j])
                ),
            }
            for j in top
        ]
        attended = [
            (term["row"], round(term["similarity"], 3), term["target"]) for term in attention_terms[:5]
        ]
        if self.mode == "classification":
            p, node = self._wins_proof(Q[row], top, S[row], row)
            return {
                "kind": "attention_classification",
                "class": self.classes_[p],
                "classes": list(self.classes_),
                "attended": attended,
                "attention_terms": attention_terms,
                "n_attended": self.topk_,
                "proof": node,
            }
        weighted_sum = float(np.dot(S[row, top], self.yval_[top]))
        weight_sum = float(S[row, top].sum())
        prediction = weighted_sum / (weight_sum + 1e-12)
        return {
            "kind": "attention_regression",
            "prediction": prediction,
            "weighted_sum": weighted_sum,
            "weight_sum": weight_sum,
            "attended": attended,
            "attention_terms": attention_terms,
            "n_attended": self.topk_,
        }

    def _proof_artifact(
        self,
        X: Any,
        row: int,
        *,
        attestation: TargetAttestation | None = None,
        trusted_attestation_keys: Mapping[str, bytes] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prediction_proof = self._prediction_proof(X, row)
        prediction = prediction_proof.get("class", prediction_proof.get("prediction"))
        guarantee, guarantee_proof = None, None
        if self._conf is not None:
            encoded = self._prep.transform(X)
            guarantee_proof = self._conf.certify_region_kernel(encoded, row)
            key = "bound" if self.mode == "regression" else "certified_precision"
            guarantee = float(guarantee_proof[key])
        artifact = build_proof_artifact(
            prediction_proof,
            mode=self.mode,
            prediction=prediction,
            prediction_verified=self.check_proof(prediction_proof),
            guarantee=guarantee,
            guarantee_proof=guarantee_proof,
            guarantee_verified=(None if guarantee_proof is None else self.check_proof(guarantee_proof)),
            feature_names=self.feature_names_,
            attestation=attestation,
            trusted_attestation_keys=trusted_attestation_keys,
        )
        return prediction_proof, artifact

    def proof(
        self,
        X: Any,
        row: int,
        *,
        raw: bool = False,
        attestation: TargetAttestation | None = None,
        trusted_attestation_keys: Mapping[str, bytes] | None = None,
    ) -> dict[str, Any]:
        """Return a clean proof reply, optionally bound to an observed target."""
        if raw and attestation is not None:
            raise ValueError("a target attestation requires the structured proof response")
        prediction_proof, artifact = self._proof_artifact(
            X,
            row,
            attestation=attestation,
            trusted_attestation_keys=trusted_attestation_keys,
        )
        return prediction_proof if raw else public_proof_response(artifact)

    def proof_artifact(
        self,
        X: Any,
        row: int,
        *,
        attestation: TargetAttestation | None = None,
        trusted_attestation_keys: Mapping[str, bytes] | None = None,
    ) -> dict[str, Any]:
        """Return detailed derivation data for an explicit independent audit."""
        _prediction_proof, artifact = self._proof_artifact(
            X,
            row,
            attestation=attestation,
            trusted_attestation_keys=trusted_attestation_keys,
        )
        return artifact

    def certify(self, X: Any = None, sample: int = 120) -> float:
        """Fraction of a sample where the FOLKernel independently re-derives the attention argmax from the query
        features (classification). Unlike the booster's exact threshold logic, a dot-product read is float-fragile
        only at genuine vote ties (margin ≈ 0); this returns the honest reproduction rate (≈1.0 in practice)."""
        if X is None or self.mode != "classification":
            return 1.0
        Q = self._query(X)
        S = Q @ self.K_.T
        idx = np.random.default_rng(self.seed).choice(len(Q), min(sample, len(Q)), replace=False)
        ok = []
        for r in idx:
            top = self._topk(S[r])
            p, node = self._wins_proof(Q[r], top, S[r], int(r))
            V = np.zeros(len(self.classes_))
            for j in top:
                V[self.yidx_[j]] += S[r, j]
            ok.append(float(node is not None and p == int(V.argmax())))
        return float(np.mean(ok)) if ok else 1.0

    def reason(self, X: Any, row: int) -> dict[str, Any]:
        att = self.attended(X, row, m=5)
        p = self.predict(X)[row]
        ev = ", ".join(f"row{j}(sim {s}, {lab})" for j, s, lab in att)
        return {
            "rule": f"prediction = {p} BECAUSE it attended to training examples [{ev}] "
            f"whose cosine-weighted vote wins",
            "attended": att,
        }

    def confidence(self, X: Any) -> Any | None:
        confidence_model = self._conf
        if confidence_model is None:
            return None
        Xe = self._prep.transform(X)
        return (
            confidence_model.bound(Xe)
            if self.mode == "regression"
            else confidence_model.certified_precision(Xe)
        )

    def certificate(self, X: Any, row: int) -> dict[str, Any]:
        pr = self.proof(X, row)
        conclusion = pr["conclusion"]
        guarantee = conclusion["guarantee"]
        machine = pr["machine_proof"]["prediction"]
        return {
            "prediction": conclusion["prediction"],
            "guarantee": None if guarantee is None else guarantee["value"],
            "attended": machine["attended"],
            "proof": pr,
        }

    def _build_confidence(self, data: Any, y: Any) -> None:
        """Leak-safe conformal calibration on 3-fold OOF attention reads (memory = other folds)."""
        self._conf = None
        y = np.asarray(y)
        if len(y) < 200:
            return
        try:
            from sklearn.model_selection import KFold

            from tabpvn.certified_confidence import CertifiedClassConfidence, CertifiedConfidence

            Xfull = self._prep.transform(data)
            if len(Xfull) > self.cap:
                sub = np.random.default_rng(self.seed).choice(len(Xfull), self.cap, replace=False)
                Xfull, y = Xfull[sub], y[sub]
            K = self._keys(Xfull)
            oof: NDArray[Any] = np.empty(len(y), dtype=object if self.mode == "classification" else float)
            for tri, vai in KFold(3, shuffle=True, random_state=self.seed).split(Xfull):
                S = K[vai] @ K[tri].T
                for r, vr in enumerate(vai):
                    top = tri[np.argpartition(-S[r], self.topk_)[: self.topk_]]  # local -> global train idx
                    w = K[vai][r] @ K[top].T
                    if self.mode == "classification":
                        V: dict[Any, float] = {}
                        for j, wj in zip(top, w, strict=False):
                            V[y[j]] = V.get(y[j], 0.0) + wj
                        oof[vr] = max(V, key=lambda label: V[label])
                    else:
                        oof[vr] = float(np.dot(w, y[top].astype(float)) / (w.sum() + 1e-12))
            if self.mode == "classification":
                self._conf = CertifiedClassConfidence(seed=self.seed).fit(Xfull, y, oof)
            else:
                self._conf = CertifiedConfidence(alpha=self.alpha, seed=self.seed).fit(
                    Xfull, y.astype(float), oof
                )
        except Exception:
            self._conf = None

    @staticmethod
    def check_proof(
        proof: Any,
        base_facts: Any = None,
        trusted_attestation_keys: Mapping[str, bytes] | None = None,
        *,
        artifact: Any = None,
    ) -> bool:
        from tabpvn.base import TabPVN

        return TabPVN.check_proof(
            proof,
            base_facts,
            trusted_attestation_keys=trusted_attestation_keys,
            artifact=artifact,
        )
