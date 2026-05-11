"""
ImbGBM — unified sklearn-compatible wrapper around the imbgbm CLI.

Every algorithm variant is a constructor parameter:

    m = ImbGBM(loss="focal", sampler="pu_goss", stack_ranker=True)
    m.fit(X_train, y_train, cat_features=[0,1,2])
    probs = m.predict_proba(X_test)          # calibrated p(y=1|x)
    rank  = m.predict_rank(X_test)           # raw rank score (audience building)
    bid   = m.predict_bid(X_test)            # calibrated p_bid (RTB pricing)

Quick presets
─────────────
    ImbGBM.focal_leaf()       # best AUC on clean labels
    ImbGBM.focal_adapt()      # best calibration on clean labels
    ImbGBM.pu_goss()          # best ECE under PU contamination
    ImbGBM.rankcal()          # two-stage: best AUC + best ECE under PU
"""

import os, subprocess, tempfile, shutil
import numpy as np
from sklearn.model_selection import KFold

IMBGBM_BIN = os.environ.get(
    "IMBGBM_BIN", "/home/user/imbgbm/target/release/imbgbm"
)


class ImbGBM:
    # ── Constructor ──────────────────────────────────────────────────────────

    def __init__(
        self,
        # ── Core ──────────────────────────────────────────────────────────
        n_rounds: int   = 500,
        learning_rate: float = 0.05,
        max_depth: int  = 7,
        subsample: float = 0.8,
        col_subsample: float = 1.0,  # feature fraction per tree (1.0 = all)
        seed: int       = 42,
        early_stopping_rounds: int = 0,
        # ── Loss ──────────────────────────────────────────────────────────
        loss: str       = "focal",   # bce | focal | pu | density_ratio
        gamma: float    = 2.0,       # focal: focusing parameter
        alpha: float    = 0.25,      # focal: class-balance weight
        pu_prior: float = 0.05,      # pu / density_ratio: class prior
        # ── Sampler ───────────────────────────────────────────────────────
        sampler: str    = "uniform", # uniform | goss | adaptive | pu_goss
        # ── Splitter ──────────────────────────────────────────────────────
        splitter: str   = "standard",# standard | variance | pu
        # ── Calibration ───────────────────────────────────────────────────
        calibrate: bool = True,      # OOF leaf-probability calibration
        platt: bool     = False,     # Platt scaling on OOF boosted scores
        estimate_pu_rate: bool = False,  # Elkan-Noto PU correction
        # ── Seed expansion ────────────────────────────────────────────────
        seed_expansion: float = 0.0, # >0 enables biased-positive expansion
        # ── Two-stage RankCal stacking ────────────────────────────────────
        stack_ranker: bool = False,  # add OOF rank features before M_cal
        n_stack_folds: int = 5,      # folds for OOF rank score generation
        # Ranker config (only used when stack_ranker=True).
        # Defaults to Focal+leaf; override to customise the first stage.
        ranker_n_rounds: int   = 400,
        ranker_lr: float       = 0.05,
        ranker_max_depth: int  = 7,
        ranker_subsample: float = 0.8,
        # ── Target encoding for categorical features ──────────────────────
        cat_smoothing: float = 10.0,
        # ── Internal ──────────────────────────────────────────────────────
        work_dir: str = None,       # None → auto temp-dir (cleaned on del)
    ):
        self.n_rounds             = n_rounds
        self.learning_rate        = learning_rate
        self.max_depth            = max_depth
        self.subsample            = subsample
        self.col_subsample        = col_subsample
        self.seed                 = seed
        self.early_stopping_rounds = early_stopping_rounds
        self.loss                 = loss
        self.gamma                = gamma
        self.alpha                = alpha
        self.pu_prior             = pu_prior
        self.sampler              = sampler
        self.splitter             = splitter
        self.calibrate            = calibrate
        self.platt                = platt
        self.estimate_pu_rate     = estimate_pu_rate
        self.seed_expansion       = seed_expansion
        self.stack_ranker         = stack_ranker
        self.n_stack_folds        = n_stack_folds
        self.ranker_n_rounds      = ranker_n_rounds
        self.ranker_lr            = ranker_lr
        self.ranker_max_depth     = ranker_max_depth
        self.ranker_subsample     = ranker_subsample
        self.cat_smoothing        = cat_smoothing

        self._own_workdir = work_dir is None
        self._work_dir    = work_dir or tempfile.mkdtemp(prefix="imbgbm_")

        # set after fit
        self._model_path     = None   # M_cal (or sole model if no stacking)
        self._ranker_path    = None   # M_rank (full model, stacking only)
        self._oof_scores     = None   # training OOF rank scores (stacking)
        self._cat_features   = None
        self._cat_cards      = None
        self._n_train        = None

    def __del__(self):
        if self._own_workdir and os.path.isdir(self._work_dir):
            shutil.rmtree(self._work_dir, ignore_errors=True)

    # ── Presets ──────────────────────────────────────────────────────────────

    @classmethod
    def focal_leaf(cls, **kw):
        """Best AUC on clean labels — Focal loss + OOF leaf calibration."""
        return cls(loss="focal", sampler="uniform", calibrate=True, **kw)

    @classmethod
    def focal_adapt(cls, **kw):
        """Best calibration on clean labels — Focal + Adaptive sampling."""
        kw.setdefault("n_rounds", 800)
        kw.setdefault("learning_rate", 0.04)
        kw.setdefault("subsample", 0.5)
        return cls(loss="focal", sampler="adaptive", calibrate=True, **kw)

    @classmethod
    def pu_goss(cls, **kw):
        """Best ECE under PU contamination — Focal + PU-GOSS sampler."""
        return cls(loss="focal", sampler="pu_goss", calibrate=False,
                   subsample=0.30, **kw)

    @classmethod
    def rankcal(cls, **kw):
        """
        Two-stage RankCal-PUGBDT: best AUC + best ECE under PU contamination.

        M_rank = Focal+leaf  → rank_score  (audience building)
        M_cal  = Focal+GOSS  → p_bid       (RTB bidding)
        """
        kw.setdefault("ranker_n_rounds", 400)
        kw.setdefault("ranker_lr", 0.05)
        kw.setdefault("ranker_max_depth", 7)
        kw.setdefault("ranker_subsample", 0.8)
        kw.setdefault("subsample", 0.30)
        return cls(loss="focal", sampler="pu_goss", calibrate=False,
                   stack_ranker=True, **kw)

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X, y, cat_features=None):
        """
        Train the model.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
            Feature matrix.  May contain raw integer category codes if
            cat_features is provided; they will be OOF target-encoded.
        y : array-like (n_samples,)
            Binary labels (0/1).  For PU learning pass observed labels s
            (labeled positives = 1, unlabeled = 0).
        cat_features : list[int] or None
            Indices of categorical columns.  Each column should contain
            integer codes in [0, cardinality).  If None, X is treated as
            already numeric.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self._n_train    = len(y)
        self._cat_features = cat_features

        # ── Target-encode categoricals ──────────────────────────────────
        if cat_features is not None:
            cards = [int(X[:, j].max()) + 1 for j in cat_features]
            self._cat_cards = cards
            # We need a held-out test placeholder for encoding — fit saves
            # the encoding so transform() can apply it to test data later.
            X_enc, self._te_means, self._te_smoothed = \
                self._fit_target_encode(X, y, cat_features, cards)
        else:
            X_enc = X

        if self.stack_ranker:
            self._fit_stacked(X_enc, y)
        else:
            self._fit_single(X_enc, y, out_path=self._main_model_path())

        return self

    # ── predict ──────────────────────────────────────────────────────────────

    def predict_proba(self, X):
        """
        Calibrated probability p(y=1|x).

        With stack_ranker=True this is p_bid (M_cal output); otherwise the
        calibrated / Platt / PU-corrected output of the single model.
        """
        X_enc = self._encode_test(X)
        if self.stack_ranker:
            return self._predict_bid(X_enc)
        return self._predict_single(X_enc, self._model_path)

    def predict_rank(self, X):
        """
        Raw ranking score (higher = more likely positive).

        With stack_ranker=True this is M_rank's calibrated output — use
        this for audience building and top-N retrieval.
        Without stacking, same as predict_proba.
        """
        X_enc = self._encode_test(X)
        if self.stack_ranker:
            return self._run_predict_csv(X_enc, self._ranker_path, "--calibrated")
        return self._predict_single(X_enc, self._model_path)

    def predict_bid(self, X):
        """
        Calibrated p_bid for RTB bidding.

        With stack_ranker=True this is M_cal's output on augmented features.
        Without stacking, same as predict_proba.
        """
        return self.predict_proba(X)

    # ── Internal: single-stage training ──────────────────────────────────────

    def _main_model_path(self):
        return os.path.join(self._work_dir, "model.json")

    def _fit_single(self, X_enc, y, out_path):
        train_csv = os.path.join(self._work_dir, "_train_single.csv")
        np.savetxt(train_csv, np.c_[X_enc, y], delimiter=",", fmt="%.6f")
        self._cli_train(train_csv, out_path)
        self._model_path = out_path

    def _predict_single(self, X_enc, model_path):
        mode = self._predict_mode()
        return self._run_predict_csv(X_enc, model_path, mode)

    def _predict_mode(self):
        if self.estimate_pu_rate: return "--pu"
        if self.platt:            return "--platt"
        if self.calibrate:        return "--calibrated"
        return None

    # ── Internal: two-stage stacked training ─────────────────────────────────

    def _fit_stacked(self, X_enc, y):
        # Stage 1: 5-fold OOF with M_rank (Focal+leaf config) ─────────────
        oof_scores = self._oof_rank_scores(X_enc, y)
        self._oof_scores = oof_scores

        # Stage 1b: full M_rank on all training data (for test-time inference)
        ranker_path = os.path.join(self._work_dir, "m_rank_full.json")
        train_csv   = os.path.join(self._work_dir, "_train_rank_full.csv")
        np.savetxt(train_csv, np.c_[X_enc, y], delimiter=",", fmt="%.6f")
        self._cli_train_ranker(train_csv, ranker_path)
        self._ranker_path = ranker_path

        # Stage 2: augment features and train M_cal ────────────────────────
        RF          = self._rank_features(oof_scores, ref=oof_scores)
        X_aug       = np.c_[X_enc, RF]
        cal_path    = os.path.join(self._work_dir, "m_cal.json")
        train_aug   = os.path.join(self._work_dir, "_train_aug.csv")
        np.savetxt(train_aug, np.c_[X_aug, y], delimiter=",", fmt="%.6f")
        self._cli_train(train_aug, cal_path)
        self._model_path = cal_path

    def _predict_bid(self, X_enc):
        # Get rank scores from full M_rank
        rank_scores = self._run_predict_csv(X_enc, self._ranker_path, "--calibrated")
        RF          = self._rank_features(rank_scores, ref=self._oof_scores)
        X_aug       = np.c_[X_enc, RF]
        mode        = self._predict_mode()
        return self._run_predict_csv(X_aug, self._model_path, mode)

    def _oof_rank_scores(self, X_enc, y):
        scores = np.zeros(len(y), dtype=np.float32)
        kf = KFold(n_splits=self.n_stack_folds, shuffle=True,
                   random_state=self.seed)
        for k, (tr, va) in enumerate(kf.split(X_enc)):
            fold_tr  = os.path.join(self._work_dir, f"fold{k}_tr.csv")
            fold_va  = os.path.join(self._work_dir, f"fold{k}_va.csv")
            fold_mdl = os.path.join(self._work_dir, f"fold{k}_rank.json")
            np.savetxt(fold_tr, np.c_[X_enc[tr], y[tr]], delimiter=",", fmt="%.6f")
            np.savetxt(fold_va, X_enc[va],                delimiter=",", fmt="%.6f")
            self._cli_train_ranker(fold_tr, fold_mdl)
            scores[va] = self._run_predict_csv(X_enc[va], fold_mdl, "--calibrated")
        return scores

    @staticmethod
    def _rank_features(scores, ref):
        p   = scores.clip(1e-7, 1 - 1e-7).astype(np.float32)
        lg  = np.log(p / (1.0 - p)).astype(np.float32)
        pct = (np.searchsorted(np.sort(ref), scores, side="left")
               / len(ref)).astype(np.float32)
        return np.column_stack([p, lg, pct])

    # ── Internal: CLI helpers ─────────────────────────────────────────────────

    def _cli_train(self, train_csv, out_path):
        args = [
            "train",
            "--input",         train_csv,
            "--output",        out_path,
            "--loss",          self.loss,
            "--n-rounds",      str(self.n_rounds),
            "--learning-rate", str(self.learning_rate),
            "--max-depth",     str(self.max_depth),
            "--subsample",     str(self.subsample),
            "--col-subsample", str(self.col_subsample),
            "--sampler",       self.sampler,
            "--splitter",      self.splitter,
            "--gamma",         str(self.gamma),
            "--alpha",         str(self.alpha),
            "--pu-prior",      str(self.pu_prior),
            "--seed",          str(self.seed),
            "--early-stopping-rounds", str(self.early_stopping_rounds),
            "--seed-expansion", str(self.seed_expansion),
        ]
        if self.calibrate:        args.append("--calibrate")
        if self.platt:            args.append("--platt")
        if self.estimate_pu_rate: args.append("--estimate-pu-rate")
        self._run(args)

    def _cli_train_ranker(self, train_csv, out_path):
        args = [
            "train",
            "--input",         train_csv,
            "--output",        out_path,
            "--loss",          "focal",
            "--n-rounds",      str(self.ranker_n_rounds),
            "--learning-rate", str(self.ranker_lr),
            "--max-depth",     str(self.ranker_max_depth),
            "--subsample",     str(self.ranker_subsample),
            "--sampler",       "uniform",
            "--splitter",      "standard",
            "--gamma",         str(self.gamma),
            "--alpha",         str(self.alpha),
            "--seed",          str(self.seed),
            "--early-stopping-rounds", "0",
            "--calibrate",
        ]
        self._run(args)

    def _run_predict_csv(self, X_enc, model_path, mode_flag):
        tmp = os.path.join(self._work_dir, "_pred_input.csv")
        np.savetxt(tmp, X_enc, delimiter=",", fmt="%.6f")
        args = ["predict", "--input", tmp, "--model", model_path]
        if mode_flag:
            args.append(mode_flag)
        r = self._run(args)
        return np.array([float(x) for x in r.stdout.strip().splitlines()],
                        dtype=np.float32)

    @staticmethod
    def _run(args):
        r = subprocess.run([IMBGBM_BIN] + args, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"imbgbm failed:\n{r.stderr[-3000:]}")
        return r

    # ── Internal: target encoding ─────────────────────────────────────────────

    def _fit_target_encode(self, X, y, cat_features, cards, smoothing=None):
        sm = smoothing or self.cat_smoothing
        gm = float(y.mean())
        n  = len(y)
        X_enc = X.astype(np.float32).copy()
        te_means    = {}
        te_smoothed = {}
        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        for j, card in zip(cat_features, cards):
            cs = np.zeros(card); cc = np.zeros(card)
            np.add.at(cs, X[:, j].astype(int), y)
            np.add.at(cc, X[:, j].astype(int), 1)
            smoothed = (cs + gm * sm) / (cc + sm)
            te_smoothed[j] = smoothed
            te_means[j]    = gm
            oof = np.full(n, gm, dtype=np.float32)
            for tr_idx, va_idx in kf.split(X):
                cs2 = np.zeros(card); cc2 = np.zeros(card)
                np.add.at(cs2, X[tr_idx, j].astype(int), y[tr_idx])
                np.add.at(cc2, X[tr_idx, j].astype(int), 1)
                sm2 = (cs2 + gm * sm) / (cc2 + sm)
                oof[va_idx] = sm2[X[va_idx, j].astype(int)]
            X_enc[:, j] = oof
        return X_enc, te_means, te_smoothed

    def _encode_test(self, X):
        X = np.asarray(X, dtype=np.float32).copy()
        if self._cat_features is not None:
            for j in self._cat_features:
                smoothed = self._te_smoothed[j]
                gm       = self._te_means[j]
                codes    = X[:, j].astype(int).clip(0, len(smoothed) - 1)
                X[:, j]  = smoothed[codes]
        return X

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self):
        parts = [
            f"loss={self.loss!r}",
            f"sampler={self.sampler!r}",
            f"splitter={self.splitter!r}",
            f"calibrate={self.calibrate}",
        ]
        if self.platt:             parts.append("platt=True")
        if self.estimate_pu_rate:  parts.append("estimate_pu_rate=True")
        if self.seed_expansion > 0: parts.append(f"seed_expansion={self.seed_expansion}")
        if self.stack_ranker:      parts.append("stack_ranker=True")
        return f"ImbGBM({', '.join(parts)})"
