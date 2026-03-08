from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import scipy.sparse as sparse
from numpyro import param, plate, sample
from numpyro.distributions import constraints

from .numpyro_model import NumpyroModel


class CSPF2(NumpyroModel):
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
                covariate_names = [f"cov_{i}" for i in range(x_np.shape[1])]

            if x_np.ndim != 2:
                raise ValueError(f"covariates must be 2D, got shape {x_np.shape}")
            if x_np.shape[0] != D:
                raise ValueError(f"covariates has {x_np.shape[0]} rows, expected {D}")
            if x_np.shape[1] == 0:
                raise ValueError("covariates matrix is empty (0 columns)")
        else:
            x_np = np.ones((D, 1), dtype=np.float32)
            covariate_names = ["intercept_cov"]

        vocab_set = set(vocab)
        for topic_id, words in keywords.items():
            for word in words:
                if word not in vocab_set:
                    raise ValueError(f"Keyword '{word}' (topic {topic_id}) not in vocabulary")

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

        rs_names = [f"residual_topic_{i+1}" for i in range(self.residual_topics)]

        return pd.DataFrame(
            jnp.transpose(E_beta), index=self.vocab, columns=list(self.keywords.keys()) + rs_names
        )

    def return_covariate_effects(self):
        topic_names = list(self.keywords.keys())
        rs_names = [f"residual_topic_{i+1}" for i in range(self.residual_topics)]
        cols = topic_names + rs_names
        index = self.covariates
        return pd.DataFrame(self.estimated_params["lambda_location"], index=index, columns=cols)
