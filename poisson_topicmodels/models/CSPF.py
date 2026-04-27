from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import scipy.sparse as sparse
from numpyro import param, plate, sample
from numpyro.distributions import constraints
from scipy import stats as sp_stats

from .numpyro_model import NumpyroModel


class CSPF(NumpyroModel):
    """
    Covariate Seeded Poisson Factorization with grouped design-adaptive shrinkage.

    This implementation preserves the CSPF interface while replacing the internal
    covariate-effect specification with the model in ``CSPF_model_new.tex``.
    """

    def __init__(
        self,
        counts: sparse.csr_matrix,
        vocab: np.ndarray,
        keywords: Dict[Any, List[str]],
        residual_topics: int,
        batch_size: int,
        X_design_matrix: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()

        if not sparse.issparse(counts):
            raise TypeError(f"counts must be a scipy sparse matrix, got {type(counts).__name__}")

        D, V = counts.shape
        if D == 0 or V == 0:
            raise ValueError(f"counts matrix is empty: shape ({D}, {V})")

        if vocab.shape[0] != V:
            raise ValueError(f"vocab size {vocab.shape[0]} != counts columns {V}")

        if not isinstance(keywords, dict):
            raise TypeError(f"keywords must be dict, got {type(keywords).__name__}")

        if residual_topics < 0:
            raise ValueError(f"residual_topics must be >= 0, got {residual_topics}")

        if batch_size <= 0 or batch_size > D:
            raise ValueError(f"batch_size must satisfy 0 < batch_size <= {D}, got {batch_size}")

        covariate_names: List[str]
        x_np: np.ndarray

        if X_design_matrix is not None:
            if isinstance(X_design_matrix, pd.DataFrame):
                covariate_names = [str(col) for col in X_design_matrix.columns]
                x_np = np.asarray(X_design_matrix.values)
            else:
                x_np = np.asarray(X_design_matrix)

            if x_np.ndim != 2:
                raise ValueError(f"covariates must be 2D, got shape {x_np.shape}")

            if not isinstance(X_design_matrix, pd.DataFrame):
                covariate_names = [f"cov_{i}" for i in range(x_np.shape[1])]

            if x_np.shape[0] != D:
                raise ValueError(f"covariates has {x_np.shape[0]} rows, expected {D}")
            if x_np.shape[1] == 0:
                raise ValueError("covariates matrix is empty (0 columns)")
        else:
            x_np = np.ones((D, 1), dtype=np.float32)
            covariate_names = ["intercept_cov"]

        # vocab_set = set(vocab)
        # for topic_id, words in keywords.items():
        #     for word in words:
        #         if word not in vocab_set:
        #             raise ValueError(f"Keyword '{word}' (topic {topic_id}) not in vocabulary")

        self.counts = counts
        self.D = D
        self.V = V
        self.vocab = vocab
        self.K = residual_topics + len(keywords)
        self.keywords = keywords
        self.residual_topics = residual_topics
        self.batch_size = batch_size

        vocab_lookup = {word: index for index, word in enumerate(vocab)}
        kw_indices_topics = [
            (idx, vocab_lookup[keyword])
            for idx, topic_id in enumerate(keywords.keys())
            for keyword in keywords[topic_id]
            if keyword in vocab_lookup
        ]
        self.Tilde_V = len(kw_indices_topics)
        self.kw_indices = tuple(zip(*kw_indices_topics)) if kw_indices_topics else ((), ())

        self.X_design_matrix = jnp.array(x_np)
        self.C = self.X_design_matrix.shape[1]
        self.covariates = covariate_names

        self.group_index = self._build_group_index(self.covariates)
        self.G = int(self.group_index.max()) + 1 if self.C > 0 else 0
        self.group_scaling_diag = self._compute_group_scaling_diag(x_np, self.group_index, self.G)

        self.b_theta = 0.3

        self.softplus_inv_one = float(np.log(np.expm1(1.0)))
        self.s_lambda0 = 1.0

        self.a_tau = 0.5
        self.a_rho_tau = 0.5
        self.b_rho_tau = 1.0
        self.a_delta = 0.5
        self.a_rho_delta = 0.5
        self.b_rho_delta = 1.0

    @staticmethod
    def _build_group_index(covariate_names: List[str]) -> np.ndarray:
        """
        Infer covariate groups from names using explicit separators.

        Supported separators: ``::``, ``=``, ``[name]`` notation.
        If none is present, each covariate is treated as its own group.
        """
        group_keys: List[str] = []
        for name in covariate_names:
            if "::" in name:
                key = name.split("::", 1)[0]
            elif "=" in name:
                key = name.split("=", 1)[0]
            elif "[" in name and name.endswith("]"):
                key = name.split("[", 1)[0]
            else:
                key = name
            group_keys.append(key)

        key_to_id: Dict[str, int] = {}
        ids: List[int] = []
        for key in group_keys:
            if key not in key_to_id:
                key_to_id[key] = len(key_to_id)
            ids.append(key_to_id[key])

        return np.asarray(ids, dtype=np.int32)

    @staticmethod
    def _compute_group_scaling_diag(
        x_np: np.ndarray, group_index: np.ndarray, G: int
    ) -> jnp.ndarray:
        """
        Compute diagonal entries of ``(X_g^T X_g)^{-1}`` per covariate column.

        For one-hot dummy columns this equals ``1 / n_j`` as in the model spec.
        """
        C = x_np.shape[1]
        scaling = np.zeros(C, dtype=np.float32)
        ridge = 1e-8

        for g in range(G):
            cols = np.where(group_index == g)[0]
            xg = x_np[:, cols]
            xtx = xg.T @ xg
            xtx_inv = np.linalg.inv(xtx + ridge * np.eye(xtx.shape[0], dtype=np.float32))
            scaling[cols] = np.diag(xtx_inv)

        return jnp.asarray(scaling)

    def _model(self, Y_batch: jnp.ndarray, d_batch: jnp.ndarray) -> None:
        with plate("k", size=self.K, dim=-2):
            with plate("k_v", size=self.V, dim=-1):
                beta = sample("beta", dist.Gamma(0.3, 0.3))

        with plate("tilde_v", size=self.Tilde_V):
            beta_tilde = sample("beta_tilde", dist.Gamma(1.0, 0.3))

        beta = beta.at[self.kw_indices].add(beta_tilde)

        with plate("k_intercept", size=self.K):
            lambda_0 = sample(
                "lambda_intercept", dist.Normal(self.softplus_inv_one, self.s_lambda0)
            )
            rho_tau = sample(
                "rho_tau", dist.Gamma(self.a_rho_tau, self.b_rho_tau)
            )  # tau (equation 8)
            tau2 = sample("tau2", dist.Gamma(self.a_tau, rho_tau))  # rho | tau (equation 8)

        with plate("g", size=self.G, dim=-2):
            with plate("g_k", size=self.K, dim=-1):
                rho_delta = sample(
                    "rho_delta", dist.Gamma(self.a_rho_delta, self.b_rho_delta)
                )  # equation 9
                delta2 = sample(
                    "delta2", dist.Gamma(self.a_delta, rho_delta)
                )  # delta | rho (equation 9)

        group_index = jnp.asarray(self.group_index)
        delta2_per_cov = delta2[group_index, :]

        lambda_scale = jnp.sqrt(tau2[None, :] * delta2_per_cov * self.group_scaling_diag[:, None])

        with plate("c", size=self.C, dim=-2):
            with plate("c_k", size=self.K, dim=-1):
                lambda_ = sample("lambda", dist.Normal(0.0, lambda_scale))

        eta_theta = lambda_0[None, :] + jnp.matmul(self.X_design_matrix, lambda_)  # equation 2
        mu_theta = jax.nn.softplus(eta_theta)[d_batch]  # equation 2
        theta_rate = self.b_theta / mu_theta  # equation 1

        with plate("d", size=self.D, subsample_size=self.batch_size, dim=-2):
            with plate("d_k", size=self.K, dim=-1):
                theta = sample("theta", dist.Gamma(self.b_theta, theta_rate))

            P = jnp.matmul(theta, beta)

            with plate("d_v", size=self.V, dim=-1):
                sample("Y_batch", dist.Poisson(P), obs=Y_batch)

    def _guide(self, Y_batch, d_batch):
        a_beta = param(
            "beta_shape", init_value=jnp.ones([self.K, self.V]), constraint=constraints.positive
        )
        b_beta = param(
            "beta_rate",
            init_value=jnp.ones([self.K, self.V]) * self.D / 1000 * 2,
            constraint=constraints.positive,
        )

        a_theta = param(
            "theta_shape", init_value=jnp.ones([self.D, self.K]), constraint=constraints.positive
        )
        b_theta = param(
            "theta_rate",
            init_value=jnp.ones([self.D, self.K]) * self.D / 1000,
            constraint=constraints.positive,
        )

        a_beta_tilde = param(
            "beta_tilde_shape",
            init_value=jnp.ones([self.Tilde_V]) * 2,
            constraint=constraints.positive,
        )
        b_beta_tilde = param(
            "beta_tilde_rate", init_value=jnp.ones([self.Tilde_V]), constraint=constraints.positive
        )

        location_lambda0 = param("lambda_intercept_location", init_value=jnp.zeros([self.K]))
        scale_lambda0 = param(
            "lambda_intercept_scale",
            init_value=jnp.ones([self.K]),
            constraint=constraints.positive,
        )

        location_lambda = param("lambda_location", init_value=jnp.zeros([self.C, self.K]))
        scale_lambda = param(
            "lambda_scale", init_value=jnp.ones([self.C, self.K]), constraint=constraints.positive
        )

        a_rho_tau = param(
            "rho_tau_shape", init_value=jnp.ones([self.K]), constraint=constraints.positive
        )
        b_rho_tau = param(
            "rho_tau_rate", init_value=jnp.ones([self.K]), constraint=constraints.positive
        )
        a_tau2 = param("tau2_shape", init_value=jnp.ones([self.K]), constraint=constraints.positive)
        b_tau2 = param("tau2_rate", init_value=jnp.ones([self.K]), constraint=constraints.positive)

        a_rho_delta = param(
            "rho_delta_shape",
            init_value=jnp.ones([self.G, self.K]),
            constraint=constraints.positive,
        )
        b_rho_delta = param(
            "rho_delta_rate", init_value=jnp.ones([self.G, self.K]), constraint=constraints.positive
        )
        a_delta2 = param(
            "delta2_shape", init_value=jnp.ones([self.G, self.K]), constraint=constraints.positive
        )
        b_delta2 = param(
            "delta2_rate", init_value=jnp.ones([self.G, self.K]), constraint=constraints.positive
        )

        with plate("k", size=self.K, dim=-2):
            with plate("k_v", size=self.V, dim=-1):
                sample("beta", dist.Gamma(a_beta, b_beta))

        with plate("tilde_v", size=self.Tilde_V):
            sample("beta_tilde", dist.Gamma(a_beta_tilde, b_beta_tilde))

        with plate("k_intercept", size=self.K):
            sample("lambda_intercept", dist.Normal(location_lambda0, scale_lambda0))
            sample("rho_tau", dist.Gamma(a_rho_tau, b_rho_tau))
            sample("tau2", dist.Gamma(a_tau2, b_tau2))

        with plate("g", size=self.G, dim=-2):
            with plate("g_k", size=self.K, dim=-1):
                sample("rho_delta", dist.Gamma(a_rho_delta, b_rho_delta))
                sample("delta2", dist.Gamma(a_delta2, b_delta2))

        with plate("c", size=self.C, dim=-2):
            with plate("c_k", size=self.K, dim=-1):
                sample("lambda", dist.Normal(location_lambda, scale_lambda))

        with plate("d", size=self.D, subsample_size=self.batch_size, dim=-2):
            with plate("d_k", size=self.K, dim=-1):
                sample("theta", dist.Gamma(a_theta[d_batch], b_theta[d_batch]))

    def return_topics(self):
        def recode_cats(argmaxes, keywords):
            num_keywords = len(keywords.keys())
            max_index = num_keywords - 1
            keyword_keys = np.array(list(keywords.keys())).astype(str)

            argmaxes_clipped = np.clip(argmaxes, 0, max_index)

            topics = np.where(
                argmaxes <= max_index,
                keyword_keys[argmaxes_clipped],
                f"No_keyword_topic_{argmaxes - max_index}",
            )

            return topics

        E_theta = self.estimated_params["theta_shape"] / self.estimated_params["theta_rate"]

        categories = np.argmax(E_theta, axis=1)
        topics = recode_cats(np.array(categories), self.keywords)

        return topics, E_theta

    def return_beta(self):
        E_beta = self.estimated_params["beta_shape"] / self.estimated_params["beta_rate"]
        E_beta_tilde = (
            self.estimated_params["beta_tilde_shape"] / self.estimated_params["beta_tilde_rate"]
        )
        E_beta = E_beta.at[self.kw_indices].add(E_beta_tilde)

        return pd.DataFrame(jnp.transpose(E_beta), index=self.vocab, columns=self._topic_names())

    def return_covariate_effects(self) -> pd.DataFrame:
        """Return point estimates of covariate effects (lambda).

        Returns
        -------
        pd.DataFrame
            DataFrame with covariates as rows and topics as columns.
        """
        index = self.covariates
        return pd.DataFrame(
            self.estimated_params["lambda_location"], index=index, columns=self._topic_names()
        )

    def return_covariate_effects_ci(self, ci: float = 0.95) -> pd.DataFrame:
        """Return covariate effects with credible intervals.

        Uses the Normal variational posterior for lambda:
        ``mean = lambda_location``, ``CI = mean +/- z * lambda_scale``.

        Parameters
        ----------
        ci : float, optional
            Credible-interval level (default 0.95).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``['covariate', 'topic', 'mean',
            'lower', 'upper']``.

        Raises
        ------
        ValueError
            If model has not been trained yet.
        """
        if not self.estimated_params:
            raise ValueError("Model must be trained before calling return_covariate_effects_ci()")

        loc = np.asarray(self.estimated_params["lambda_location"])  # (C, K)
        scale = np.asarray(self.estimated_params["lambda_scale"])  # (C, K)
        z = sp_stats.norm.ppf(1.0 - (1.0 - ci) / 2.0)

        topic_names = self._topic_names()
        rows = []
        for c_idx, cov_name in enumerate(self.covariates):
            for k_idx, topic_name in enumerate(topic_names):
                rows.append(
                    {
                        "covariate": cov_name,
                        "topic": topic_name,
                        "mean": float(loc[c_idx, k_idx]),
                        "lower": float(loc[c_idx, k_idx] - z * scale[c_idx, k_idx]),
                        "upper": float(loc[c_idx, k_idx] + z * scale[c_idx, k_idx]),
                    }
                )
        return pd.DataFrame(rows)

    def _summary_extra(self) -> str:
        """CSPF-specific summary information."""
        lines = [
            f"  Keywords:                 {len(self.keywords)} seeded topics",
            f"  Residual topics:          {self.residual_topics}",
            f"  Covariates (C):           {self.C}",
            f"  Covariate names:          {', '.join(self.covariates)}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Forest-plot visualisation
    # ------------------------------------------------------------------

    def _topic_names(self) -> List[str]:
        """Return ordered list of all topic names (keyword + residual)."""
        return list(self.keywords.keys()) + [
            f"residual_topic_{i + 1}" for i in range(self.residual_topics)
        ]

    def _group_names(self) -> List[str]:
        """Return ordered list of covariate-group names."""
        seen: Dict[str, None] = {}
        for name in self.covariates:
            key = name.split("::", 1)[0] if "::" in name else name
            if key not in seen:
                seen[key] = None
        return list(seen.keys())

    @staticmethod
    def _gamma_ci(
        shape: np.ndarray, rate: np.ndarray, ci: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Point estimate and CI for a Gamma variational posterior."""
        mean = shape / rate
        alpha_lo = (1.0 - ci) / 2.0
        alpha_hi = 1.0 - alpha_lo
        lo = sp_stats.gamma.ppf(alpha_lo, a=shape, scale=1.0 / rate)
        hi = sp_stats.gamma.ppf(alpha_hi, a=shape, scale=1.0 / rate)
        return mean, lo, hi

    # ---- public API ---------------------------------------------------

    def plot_cov_effects(
        self,
        ci: float = 0.95,
        include_shrinkage: bool = False,
        topics: Optional[List[str]] = None,
        group_colors: Optional[Dict[str, str]] = None,
        figsize_per_topic: Tuple[float, float] = (5.0, 0.28),
        save_path: Optional[str] = None,
    ) -> Dict[str, Tuple[plt.Figure, np.ndarray]]:
        r"""Plot covariate effects as forest plots.

        Parameters
        ----------
        ci : float, optional
            Credible-interval level (default ``0.95`` for 95 % CI).
        include_shrinkage : bool, optional
            If ``True``, additionally produce forest plots for
            :math:`\lambda_0` (intercept), :math:`\tau^2_k` (global
            shrinkage), and :math:`\delta^2_{gk}` (group shrinkage).
        topics : list of str, optional
            Subset of topic names to plot.  If ``None`` (default), all
            topics are plotted.
        group_colors : dict, optional
            Mapping ``{group_name: colour}`` used to colour the
            covariate labels on the y-axis.  Groups are inferred from
            the ``::`` separator in covariate names.  If ``None`` a
            default qualitative palette is used.
        figsize_per_topic : tuple of float, optional
            ``(width, height_per_covariate)`` used to auto-size the
            lambda panels.  Default ``(5.0, 0.28)``.
        save_path : str, optional
            Directory (or file path) where figures are saved.  When a
            directory is given, individual PNGs are written; when a file
            path is given, only the lambda figure is saved there.
            If ``None``, figures are not saved.

        Returns
        -------
        dict
            ``{"lambda": (fig, axes), ...}`` and, when
            *include_shrinkage* is ``True``, additional entries
            ``"lambda_intercept"``, ``"tau2"``, ``"delta2"``.
        """
        import os

        if not self.estimated_params:
            raise RuntimeError("No estimated parameters found. Train the model first.")

        all_topic_names = self._topic_names()
        if topics is not None:
            sel = [i for i, t in enumerate(all_topic_names) if t in topics]
            if not sel:
                raise ValueError(f"None of {topics} found in model topics {all_topic_names}")
            plot_topics = [all_topic_names[i] for i in sel]
            topic_idx = sel
        else:
            plot_topics = all_topic_names
            topic_idx = list(range(len(all_topic_names)))

        # -- colours per covariate group ----------------------------------
        grp_names = self._group_names()
        if group_colors is None:
            _qualitative = [
                "#4E79A7",
                "#F28E2B",
                "#E15759",
                "#76B7B2",
                "#59A14F",
                "#EDC948",
                "#B07AA1",
                "#FF9DA7",
                "#9C755F",
                "#BAB0AC",
            ]
            group_colors = {g: _qualitative[i % len(_qualitative)] for i, g in enumerate(grp_names)}

        def _cov_color(name: str) -> str:
            key = name.split("::", 1)[0] if "::" in name else name
            return group_colors.get(key, "#333333")

        results: Dict[str, Tuple[plt.Figure, np.ndarray]] = {}

        # ================================================================
        # Lambda forest plot
        # ================================================================
        loc = np.asarray(self.estimated_params["lambda_location"])  # (C, K)
        scale = np.asarray(self.estimated_params["lambda_scale"])  # (C, K)
        z = sp_stats.norm.ppf(1.0 - (1.0 - ci) / 2.0)

        n_topics = len(plot_topics)
        n_cov = loc.shape[0]

        with plt.rc_context(self._setup_academic_style()):
            fig_w = figsize_per_topic[0]
            fig_h = max(3.0, n_cov * figsize_per_topic[1])

            ncols = min(n_topics, 4)
            nrows = int(np.ceil(n_topics / ncols))
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(fig_w * ncols, fig_h * nrows),
                sharey=True,
                squeeze=False,
            )
            axes_flat = axes.flatten()

            # Pre-compute global x-range across all panels for shared scale
            all_lo = loc[:, topic_idx] - z * scale[:, topic_idx]
            all_hi = loc[:, topic_idx] + z * scale[:, topic_idx]
            global_xmin = float(np.min(all_lo))
            global_xmax = float(np.max(all_hi))
            x_pad = (global_xmax - global_xmin) * 0.08
            global_xmin -= x_pad
            global_xmax += x_pad

            for panel_i, (ki, tname) in enumerate(zip(topic_idx, plot_topics)):
                ax = axes_flat[panel_i]
                means = loc[:, ki]
                lo = means - z * scale[:, ki]
                hi = means + z * scale[:, ki]

                y_pos = np.arange(n_cov)[::-1]
                colors = [_cov_color(c) for c in self.covariates]

                # CI lines
                for j in range(n_cov):
                    ax.plot(
                        [lo[j], hi[j]],
                        [y_pos[j], y_pos[j]],
                        color=colors[j],
                        linewidth=1.2,
                        solid_capstyle="round",
                    )
                # point estimates
                ax.scatter(
                    means,
                    y_pos,
                    s=18,
                    zorder=5,
                    color=[colors[j] for j in range(n_cov)],
                    edgecolors="white",
                    linewidths=0.3,
                )

                # Zero reference line — thick solid, semi-transparent
                ax.axvline(0, color="#333333", linewidth=1.4, linestyle="-", alpha=0.45, zorder=1)
                ax.set_xlim(global_xmin, global_xmax)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(
                    list(self.covariates),
                    fontsize=7,
                    color="#222222",
                )
                # Colour y-tick labels by group
                for tick_label, cov_name in zip(ax.get_yticklabels(), self.covariates):
                    tick_label.set_color(_cov_color(cov_name))

                ax.set_title(tname, fontweight="bold", pad=6)
                ax.set_xlabel(r"$\lambda$")
                ax.margins(y=0.02)

            # hide unused panels
            for j in range(n_topics, len(axes_flat)):
                axes_flat[j].set_visible(False)

            # Build legend from group colours
            from matplotlib.lines import Line2D

            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=group_colors[g],
                    linestyle="None",
                    markersize=5,
                    label=g,
                )
                for g in grp_names
                if g in group_colors
            ]
            fig.legend(
                handles=legend_handles,
                title="Covariate group",
                loc="lower center",
                ncol=min(len(legend_handles), 6),
                frameon=False,
                bbox_to_anchor=(0.5, -0.01),
            )

            fig.suptitle(
                f"Covariate Effects on Topic Intensity ({int(ci * 100)}% CI)",
                fontsize=12,
                fontweight="bold",
                y=1.02,
            )
            fig.tight_layout()
            results["lambda"] = (fig, axes)

            if save_path is not None:
                _save = (
                    os.path.join(save_path, "forest_lambda.png")
                    if os.path.isdir(save_path)
                    else save_path
                )
                fig.savefig(_save, dpi=200, bbox_inches="tight")

        # ================================================================
        # Optional shrinkage panels
        # ================================================================
        if include_shrinkage:
            with plt.rc_context(self._setup_academic_style()):
                # --- lambda_intercept ---
                loc0 = np.asarray(self.estimated_params["lambda_intercept_location"])
                scale0 = np.asarray(self.estimated_params["lambda_intercept_scale"])
                means0 = loc0[topic_idx]
                lo0 = means0 - z * scale0[topic_idx]
                hi0 = means0 + z * scale0[topic_idx]

                fig_int, ax_int = plt.subplots(figsize=(4.5, max(2.5, 0.35 * n_topics)))
                y_pos = np.arange(n_topics)[::-1]
                for j in range(n_topics):
                    ax_int.plot(
                        [lo0[j], hi0[j]],
                        [y_pos[j], y_pos[j]],
                        color="#4E79A7",
                        linewidth=1.3,
                        solid_capstyle="round",
                    )
                ax_int.scatter(
                    means0,
                    y_pos,
                    s=22,
                    zorder=5,
                    color="#4E79A7",
                    edgecolors="white",
                    linewidths=0.4,
                )
                ax_int.axvline(
                    0, color="#333333", linewidth=1.4, linestyle="-", alpha=0.45, zorder=0
                )
                ax_int.set_yticks(y_pos)
                ax_int.set_yticklabels(plot_topics, fontsize=8)
                ax_int.set_xlabel(r"$\lambda_0$")
                ax_int.set_title(
                    f"Intercept $\\lambda_0$ ({int(ci * 100)}% CI)",
                    fontweight="bold",
                    pad=6,
                )
                ax_int.margins(y=0.04)
                fig_int.tight_layout()
                results["lambda_intercept"] = (fig_int, np.array([ax_int]))

                if save_path is not None and os.path.isdir(save_path):
                    fig_int.savefig(
                        os.path.join(save_path, "forest_lambda_intercept.png"),
                        dpi=200,
                        bbox_inches="tight",
                    )

                # --- tau2 (global shrinkage per topic) ---
                tau2_s = np.asarray(self.estimated_params["tau2_shape"])
                tau2_r = np.asarray(self.estimated_params["tau2_rate"])
                tau_mean, tau_lo, tau_hi = self._gamma_ci(tau2_s[topic_idx], tau2_r[topic_idx], ci)

                fig_tau, ax_tau = plt.subplots(figsize=(4.5, max(2.5, 0.35 * n_topics)))
                for j in range(n_topics):
                    ax_tau.plot(
                        [tau_lo[j], tau_hi[j]],
                        [y_pos[j], y_pos[j]],
                        color="#E15759",
                        linewidth=1.3,
                        solid_capstyle="round",
                    )
                ax_tau.scatter(
                    tau_mean,
                    y_pos,
                    s=22,
                    zorder=5,
                    color="#E15759",
                    edgecolors="white",
                    linewidths=0.4,
                )
                ax_tau.axvline(
                    0, color="#333333", linewidth=1.4, linestyle="-", alpha=0.45, zorder=0
                )
                ax_tau.set_yticks(y_pos)
                ax_tau.set_yticklabels(plot_topics, fontsize=8)
                ax_tau.set_xlabel(r"$\tau^2$")
                ax_tau.set_title(
                    f"Global Shrinkage $\\tau^2_k$ ({int(ci * 100)}% CI)",
                    fontweight="bold",
                    pad=6,
                )
                ax_tau.margins(y=0.04)
                fig_tau.tight_layout()
                results["tau2"] = (fig_tau, np.array([ax_tau]))

                if save_path is not None and os.path.isdir(save_path):
                    fig_tau.savefig(
                        os.path.join(save_path, "forest_tau2.png"),
                        dpi=200,
                        bbox_inches="tight",
                    )

                # --- delta2 (group shrinkage, per group × topic) ---
                d2_s = np.asarray(self.estimated_params["delta2_shape"])  # (G, K)
                d2_r = np.asarray(self.estimated_params["delta2_rate"])

                n_groups = d2_s.shape[0]
                grp_labels = self._group_names()

                ncols_d = min(n_topics, 4)
                nrows_d = int(np.ceil(n_topics / ncols_d))
                fig_d, axes_d = plt.subplots(
                    nrows_d,
                    ncols_d,
                    figsize=(4.5 * ncols_d, max(2.5, 0.35 * n_groups) * nrows_d),
                    sharey=True,
                    squeeze=False,
                )
                axes_d_flat = axes_d.flatten()

                # Pre-compute global x-range for delta2 panels
                all_d_means = []
                all_d_los = []
                all_d_his = []
                for ki in topic_idx:
                    dm, dl, dh = self._gamma_ci(d2_s[:, ki], d2_r[:, ki], ci)
                    all_d_means.append(dm)
                    all_d_los.append(dl)
                    all_d_his.append(dh)
                d_global_xmin = float(np.min(np.concatenate(all_d_los)))
                d_global_xmax = float(np.max(np.concatenate(all_d_his)))
                d_x_pad = (d_global_xmax - d_global_xmin) * 0.08
                d_global_xmin = max(0.0, d_global_xmin - d_x_pad)
                d_global_xmax += d_x_pad

                for panel_i, (ki, tname) in enumerate(zip(topic_idx, plot_topics)):
                    ax = axes_d_flat[panel_i]
                    d_mean, d_lo, d_hi = self._gamma_ci(d2_s[:, ki], d2_r[:, ki], ci)
                    yp = np.arange(n_groups)[::-1]
                    for j in range(n_groups):
                        ax.plot(
                            [d_lo[j], d_hi[j]],
                            [yp[j], yp[j]],
                            color="#59A14F",
                            linewidth=1.3,
                            solid_capstyle="round",
                        )
                    ax.scatter(
                        d_mean,
                        yp,
                        s=22,
                        zorder=5,
                        color="#59A14F",
                        edgecolors="white",
                        linewidths=0.4,
                    )
                    ax.axvline(
                        0, color="#333333", linewidth=1.4, linestyle="-", alpha=0.45, zorder=0
                    )
                    ax.set_xlim(d_global_xmin, d_global_xmax)
                    ax.set_yticks(yp)
                    ax.set_yticklabels(grp_labels, fontsize=8)
                    ax.set_xlabel(r"$\delta^2$")
                    ax.set_title(tname, fontweight="bold", pad=6)
                    ax.margins(y=0.04)

                for j in range(n_topics, len(axes_d_flat)):
                    axes_d_flat[j].set_visible(False)

                fig_d.suptitle(
                    f"Group Shrinkage $\\delta^2_{{gk}}$ ({int(ci * 100)}% CI)",
                    fontsize=12,
                    fontweight="bold",
                    y=1.02,
                )
                fig_d.tight_layout()
                results["delta2"] = (fig_d, axes_d)

                if save_path is not None and os.path.isdir(save_path):
                    fig_d.savefig(
                        os.path.join(save_path, "forest_delta2.png"),
                        dpi=200,
                        bbox_inches="tight",
                    )

        return results
