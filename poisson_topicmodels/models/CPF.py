from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import scipy.sparse as sparse
from numpyro import plate, sample
from numpyro.distributions import constraints
from scipy import stats as sp_stats

# Abstract class - defining the minimum requirements for the probabilistic model
from .numpyro_model import NumpyroModel


# Create numpyro model
class CPF(NumpyroModel):
    """
    Covariate Poisson Factorization (CPF) topic model.

    Topic model that incorporates document-level covariates to capture how topics
    vary with external variables (e.g., author attributes, temporal features).

    Parameters
    ----------
    counts : scipy.sparse.csr_matrix
        Document-term matrix of shape (D, V) with word counts.
    vocab : np.ndarray
        Vocabulary array of shape (V,) containing word terms.
    covariates : np.ndarray or pd.DataFrame
        Document-level covariates of shape (D, C) where C is number of features.
    num_topics : int
        Number of topics K. Must be > 0.
    batch_size : int
        Mini-batch size for stochastic variational inference.
        Must satisfy 0 < batch_size <= D.
    initparams : dict, optional
        User-specified initial values for variational parameters in the guide.
    constantparams : dict, optional
        User-specified constant values for latent variables (not updated by SVI).
    hyperparams : dict, optional
        User-specified hyperparameters overriding default prior settings.
    link_function : {"softplus", "exp"}, optional
        Positive link function used to map the linear predictor for
        document-topic intensities to their Gamma prior mean.
        Default is "softplus".

    Attributes
    ----------
    D : int
        Number of documents.
    V : int
        Vocabulary size.
    K : int
        Number of topics.
    C : int
        Number of covariate features.
    counts : scipy.sparse.csr_matrix
        Document-term matrix.
    vocab : np.ndarray
        Vocabulary array.
    X_design_matrix : jnp.ndarray
        Design matrix of covariates.
    G : int
        Number of covariate groups.
    group_scaling_diag : jnp.ndarray
        Per-covariate scaling derived from (X_g^T X_g)^{-1}.

    Examples
    --------
    >>> from scipy.sparse import random
    >>> import numpy as np
    >>> from topicmodels import CPF
    >>> counts = random(100, 500, density=0.01, format='csr')
    >>> vocab = np.array([f'word_{i}' for i in range(500)])
    >>> covariates = np.random.randn(100, 3)  # 3 covariate features
    >>> model = CPF(counts, vocab, covariates, num_topics=10, batch_size=32)
    >>> params = model.train_step(num_steps=100, lr=0.01, random_seed=42)
    """

    def __init__(
        self,
        counts: sparse.csr_matrix,
        vocab: np.ndarray,
        num_topics: int,
        batch_size: int,
        X_design_matrix: Optional[np.ndarray] = None,
        initparams: Optional[Dict[str, Any]] = None,
        constantparams: Optional[Dict[str, Any]] = None,
        hyperparams: Optional[Dict[str, float]] = None,
        link_function: str = "softplus",
    ) -> None:
        """
        Initialize the CPF model with input validation.

        Parameters
        ----------
        counts : scipy.sparse.csr_matrix
            Document-term matrix.
        vocab : np.ndarray
            Vocabulary array.
        num_topics : int
            Number of topics.
        batch_size : int
            Mini-batch size.
        X_design_matrix : np.ndarray or pd.DataFrame, optional
            Document-level covariates.
        initparams : dict, optional
            Initial values for variational parameters.
        constantparams : dict, optional
            Fixed values for latent variables.
        hyperparams : dict, optional
            Hyperparameters overriding default priors.

        Raises
        ------
        TypeError
            If counts is not sparse or covariates have wrong type.
        ValueError
            If dimensions are invalid or inconsistent.
        """
        super().__init__(
            initparams=initparams,
            constantparams=constantparams,
            hyperparams=hyperparams,
        )

        # Input validation
        if not sparse.issparse(counts):
            raise TypeError(f"counts must be a scipy sparse matrix, got {type(counts).__name__}")

        D, V = counts.shape
        if D == 0 or V == 0:
            raise ValueError(f"counts matrix is empty: shape ({D}, {V})")

        if vocab.shape[0] != V:
            raise ValueError(f"vocab size {vocab.shape[0]} != counts columns {V}")

        if num_topics <= 0:
            raise ValueError(f"num_topics must be > 0, got {num_topics}")

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

        if link_function not in {"softplus", "exp"}:
            raise ValueError(
                f"link_function must be one of {{'softplus', 'exp'}}, got {link_function!r}"
            )

        self.link_function = link_function

        self.counts = counts
        self.D = D
        self.V = V
        self.vocab = vocab
        self.K = num_topics
        self.batch_size = batch_size

        self.X_design_matrix = jnp.array(x_np)
        self.C = self.X_design_matrix.shape[1]
        self.covariates = covariate_names

        self.group_index = self._build_group_index(self.covariates)
        self.G = int(self.group_index.max()) + 1 if self.C > 0 else 0
        self.group_scaling_diag = self._compute_group_scaling_diag(x_np, self.group_index, self.G)

    def _link_function(self, x: jnp.ndarray) -> jnp.ndarray:
        """Map unconstrained linear predictors to positive values."""
        if self.link_function == "softplus":
            return jax.nn.softplus(x)
        if self.link_function == "exp":
            return jnp.exp(x)
        raise ValueError(f"Unsupported link_function: {self.link_function}")

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

    # -- Model --
    def _model(self, Y_batch: jnp.ndarray, d_batch: jnp.ndarray) -> None:
        """
        Define the probabilistic generative model using NumPyro.

        Model structure:
        - Beta (K x V): topic-word distributions
        - Lambda_0 (K,): topic-specific intercepts for log-intensity.
        - Lambda (C x K): covariate effects on topics with grouped,
          design-adaptive shrinkage.
        - Theta (D x K): document-topic intensities with a Gamma prior
          whose mean depends on covariates.
        - Y_batch (batch_size x V): observed word counts

        Parameters
        ----------
        Y_batch : jnp.ndarray
            Batch of observed word counts (batch_size, V).
        d_batch : jnp.ndarray
            Document indices in batch (batch_size,).
        """

        # Topic distributions
        with plate("k", size=self.K, dim=-2):
            with plate("k_v", size=self.V, dim=-1):
                beta = self._sample(
                    "beta",
                    dist.Gamma(
                        self._hyperparam("a_beta", 0.3, positive=True),
                        self._hyperparam("b_beta", 0.3, positive=True),
                    ),
                    dimensions=(self.K, self.V),
                    positive=True,
                )

        # Intercept + global shrinkage
        with plate("k_intercept", size=self.K):
            lambda_0 = self._sample(
                "lambda_intercept",
                dist.Normal(
                    self._hyperparam("mu_lambda0", float(np.log(np.expm1(1.0)))),
                    self._hyperparam("s_lambda0", 1.0, positive=True),
                ),
                dimensions=(self.K,),
            )

            rho_tau = self._sample(
                "rho_tau",
                dist.Gamma(
                    self._hyperparam("a_rho_tau", 0.5, positive=True),
                    self._hyperparam("b_rho_tau", 1.0, positive=True),
                ),
                dimensions=(self.K,),
                positive=True,
            )

            tau2 = self._sample(
                "tau2",
                dist.Gamma(self._hyperparam("a_tau", 0.5, positive=True), rho_tau),
                dimensions=(self.K,),
                positive=True,
            )

        # Group-specific shrinkage
        with plate("g", size=self.G, dim=-2):
            with plate("g_k", size=self.K, dim=-1):
                rho_delta = self._sample(
                    "rho_delta",
                    dist.Gamma(
                        self._hyperparam("a_rho_delta", 0.5, positive=True),
                        self._hyperparam("b_rho_delta", 1.0, positive=True),
                    ),
                    dimensions=(self.G, self.K),
                    positive=True,
                )

                delta2 = self._sample(
                    "delta2",
                    dist.Gamma(self._hyperparam("a_delta", 0.5, positive=True), rho_delta),
                    dimensions=(self.G, self.K),
                    positive=True,
                )

        group_index = jnp.asarray(self.group_index)
        delta2_per_cov = delta2[group_index, :]

        lambda_scale = jnp.sqrt(tau2[None, :] * delta2_per_cov * self.group_scaling_diag[:, None])

        # Covariate effects
        with plate("c", size=self.C, dim=-2):
            with plate("c_k", size=self.K, dim=-1):
                lambda_ = self._sample(
                    "lambda",
                    dist.Normal(0.0, lambda_scale),
                    dimensions=(self.C, self.K),
                )

        eta_theta = lambda_0[None, :] + jnp.matmul(self.X_design_matrix, lambda_)
        mu_theta = self._link_function(eta_theta)[d_batch]
        b_theta = self._hyperparam("b_theta", 0.3, positive=True)
        theta_rate = b_theta / mu_theta

        # Document distribution
        with plate("d", size=self.D, subsample_size=self.batch_size, dim=-2):
            with plate("d_k", size=self.K, dim=-1):
                theta = self._sample(
                    "theta",
                    dist.Gamma(b_theta, theta_rate),
                    dimensions=(self.batch_size, self.K),
                    positive=True,
                )

            # Poisson rate
            P = jnp.matmul(theta, beta)

            with plate("d_v", size=self.V, dim=-1):
                sample("Y_batch", dist.Poisson(P), obs=Y_batch)

    # -- Guide, i.e. variational family --
    def _guide(self, Y_batch: jnp.ndarray, d_batch: jnp.ndarray) -> None:
        """
        Define the variational guide (approximate posterior).

        Uses Gamma variational families for topic-word factors, document-topic
        intensities, and shrinkage parameters, and Normal variational families
        for topic-specific intercepts and covariate effects.

        Parameters
        ----------
        Y_batch : jnp.ndarray
            Batch of observed word counts.
        d_batch : jnp.ndarray
            Document indices in batch.
        """

        if not self._is_constant("beta"):
            a_beta = self._param(
                "beta_shape", init_value=jnp.ones([self.K, self.V]), constraint=constraints.positive
            )
            b_beta = self._param(
                "beta_rate",
                init_value=jnp.ones([self.K, self.V]) * self.D / 1000 * 2,
                constraint=constraints.positive,
            )

            with plate("k", size=self.K, dim=-2):
                with plate("k_v", size=self.V, dim=-1):
                    sample("beta", dist.Gamma(a_beta, b_beta))

        if not self._is_constant("lambda_intercept"):
            location_lambda0 = self._param(
                "lambda_intercept_location",
                init_value=jnp.zeros([self.K]),
            )
            scale_lambda0 = self._param(
                "lambda_intercept_scale",
                init_value=jnp.ones([self.K]),
                constraint=constraints.positive,
            )

            with plate("k_intercept", size=self.K):
                sample("lambda_intercept", dist.Normal(location_lambda0, scale_lambda0))

        if not self._is_constant("rho_tau"):
            a_rho_tau = self._param(
                "rho_tau_shape",
                init_value=jnp.ones([self.K]),
                constraint=constraints.positive,
            )
            b_rho_tau = self._param(
                "rho_tau_rate",
                init_value=jnp.ones([self.K]),
                constraint=constraints.positive,
            )

            with plate("k_intercept", size=self.K):
                sample("rho_tau", dist.Gamma(a_rho_tau, b_rho_tau))

        if not self._is_constant("tau2"):
            a_tau2 = self._param(
                "tau2_shape",
                init_value=jnp.ones([self.K]),
                constraint=constraints.positive,
            )
            b_tau2 = self._param(
                "tau2_rate",
                init_value=jnp.ones([self.K]),
                constraint=constraints.positive,
            )

            with plate("k_intercept", size=self.K):
                sample("tau2", dist.Gamma(a_tau2, b_tau2))

        if not self._is_constant("rho_delta"):
            a_rho_delta = self._param(
                "rho_delta_shape",
                init_value=jnp.ones([self.G, self.K]),
                constraint=constraints.positive,
            )
            b_rho_delta = self._param(
                "rho_delta_rate",
                init_value=jnp.ones([self.G, self.K]),
                constraint=constraints.positive,
            )

            with plate("g", size=self.G, dim=-2):
                with plate("g_k", size=self.K, dim=-1):
                    sample("rho_delta", dist.Gamma(a_rho_delta, b_rho_delta))

        if not self._is_constant("delta2"):
            a_delta2 = self._param(
                "delta2_shape",
                init_value=jnp.ones([self.G, self.K]),
                constraint=constraints.positive,
            )
            b_delta2 = self._param(
                "delta2_rate",
                init_value=jnp.ones([self.G, self.K]),
                constraint=constraints.positive,
            )

            with plate("g", size=self.G, dim=-2):
                with plate("g_k", size=self.K, dim=-1):
                    sample("delta2", dist.Gamma(a_delta2, b_delta2))

        if not self._is_constant("lambda"):
            location_lambda = self._param(
                "lambda_location",
                init_value=jnp.zeros([self.C, self.K]),
            )
            scale_lambda = self._param(
                "lambda_scale",
                init_value=jnp.ones([self.C, self.K]),
                constraint=constraints.positive,
            )

            with plate("c", size=self.C, dim=-2):
                with plate("c_k", size=self.K, dim=-1):
                    sample("lambda", dist.Normal(location_lambda, scale_lambda))

        if not self._is_constant("theta"):
            a_theta = self._param(
                "theta_shape",
                init_value=jnp.ones([self.D, self.K]),
                constraint=constraints.positive,
            )
            b_theta = self._param(
                "theta_rate",
                init_value=jnp.ones([self.D, self.K]) * self.D / 1000,
                constraint=constraints.positive,
            )

            with plate("d", size=self.D, subsample_size=self.batch_size, dim=-2):
                with plate("d_k", size=self.K, dim=-1):
                    sample("theta", dist.Gamma(a_theta[d_batch], b_theta[d_batch]))

    def _topic_names(self) -> List[str]:
        """Return ordered list of topic names."""
        return [f"topic_{i + 1}" for i in range(self.K)]

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

    def return_covariate_effects(self) -> pd.DataFrame:
        """Return point estimates of covariate effects (lambda)."""
        if not self.estimated_params:
            raise ValueError("Model must be trained before calling return_covariate_effects()")

        index = self.covariates
        if self._is_constant("lambda"):
            values = np.asarray(self._constantparams["lambda"])
        else:
            values = np.asarray(self.estimated_params["lambda_location"])

        return pd.DataFrame(values, index=index, columns=self._topic_names())

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

        if self._is_constant("lambda"):
            loc = np.asarray(self._constantparams["lambda"])
            scale = np.zeros_like(loc)
        else:
            loc = np.asarray(self.estimated_params["lambda_location"])
            scale = np.asarray(self.estimated_params["lambda_scale"])

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
        if self._is_constant("lambda"):
            loc = np.asarray(self._constantparams["lambda"])
            scale = np.zeros_like(loc)
        else:
            loc = np.asarray(self.estimated_params["lambda_location"])
            scale = np.asarray(self.estimated_params["lambda_scale"])

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
                if self._is_constant("lambda_intercept"):
                    loc0 = np.asarray(self._constantparams["lambda_intercept"])
                    scale0 = np.zeros_like(loc0)
                else:
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
                if self._is_constant("tau2"):
                    tau_mean = np.asarray(self._constantparams["tau2"])[topic_idx]
                    tau_lo = tau_mean
                    tau_hi = tau_mean
                else:
                    tau2_s = np.asarray(self.estimated_params["tau2_shape"])
                    tau2_r = np.asarray(self.estimated_params["tau2_rate"])
                    tau_mean, tau_lo, tau_hi = self._gamma_ci(
                        tau2_s[topic_idx], tau2_r[topic_idx], ci
                    )

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
                if self._is_constant("delta2"):
                    d2_const = np.asarray(self._constantparams["delta2"])
                    n_groups = d2_const.shape[0]
                    grp_labels = self._group_names()
                else:
                    d2_s = np.asarray(self.estimated_params["delta2_shape"])
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
                all_d_los = []
                all_d_his = []
                for ki in topic_idx:
                    if self._is_constant("delta2"):
                        d_mean = d2_const[:, ki]
                        d_lo = d_mean
                        d_hi = d_mean
                    else:
                        d_mean, d_lo, d_hi = self._gamma_ci(d2_s[:, ki], d2_r[:, ki], ci)
                    all_d_los.append(d_lo)
                    all_d_his.append(d_hi)
                d_global_xmin = float(np.min(np.concatenate(all_d_los)))
                d_global_xmax = float(np.max(np.concatenate(all_d_his)))
                d_x_pad = (d_global_xmax - d_global_xmin) * 0.08
                d_global_xmin = max(0.0, d_global_xmin - d_x_pad)
                d_global_xmax += d_x_pad

                for panel_i, (ki, tname) in enumerate(zip(topic_idx, plot_topics)):
                    ax = axes_d_flat[panel_i]
                    if self._is_constant("delta2"):
                        d_mean = d2_const[:, ki]
                        d_lo = d_mean
                        d_hi = d_mean
                    else:
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

    def _summary_extra(self) -> str:
        """CPF-specific summary information."""
        lines = [
            f"  Covariates (C):           {self.C}",
            f"  Covariate groups (G):     {self.G}",
            f"  Covariate names:          {', '.join(self.covariates)}",
        ]
        return "\n".join(lines)
