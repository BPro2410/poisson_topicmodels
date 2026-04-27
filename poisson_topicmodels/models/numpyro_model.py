from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from jax import jit, random
from numpyro.infer import SVI, TraceMeanField_ELBO
from optax import adam
from tqdm import tqdm
from wordcloud import WordCloud

from .Metrics import Metrics


class NumpyroModel(ABC):
    """
    Abstract base class for all used probabilistic models.
    Each model has to implement at least their own Model and Guide.

    Attributes
    ----------
    Metrics : Metrics
        Instance metrics tracker (per instance, not shared).
    estimated_params : dict
        Estimated parameters after training.
    D : int
        Number of documents.
    V : int
        Vocabulary size.
    batch_size : int
        Mini-batch size for stochastic variational inference.
    counts : scipy.sparse.csr_matrix
        Document-term matrix.
    vocab : np.ndarray
        Vocabulary array.
    K : int
        Number of topics.
    """

    def __init__(self) -> None:
        """Initialize base model with per-instance metrics."""
        self.Metrics = Metrics(loss=[])
        self.estimated_params: Dict[str, Any] = {}
        # These will be set by child classes, declared here for type checking
        self.D: int
        self.V: int
        self.batch_size: int
        self.counts: sparse.csr_matrix
        self.vocab: np.ndarray
        self.K: int
        self._dense_counts_cache: Optional[jax.Array] = None

    @abstractmethod
    def _model(self, Y_batch: Any, d_batch: Any) -> None:
        """Define the probabilistic model."""
        pass

    @abstractmethod
    def _guide(self, Y_batch: Any, d_batch: Any) -> None:
        """Define the variational guide."""
        pass

    def _prepare_dense_cache(
        self, cache_dense_counts: Optional[bool], dense_cache_max_gb: float
    ) -> None:
        """Optionally cache counts as a dense JAX array for faster mini-batching."""
        if jax.default_backend().lower() == "metal":
            # Metal backend currently errors on this device_put path.
            self._dense_counts_cache = None
            return

        if cache_dense_counts is False:
            self._dense_counts_cache = None
            return

        if cache_dense_counts is None:
            dense_size_bytes = self.D * self.V * np.dtype(np.float32).itemsize
            cache_dense_counts = dense_size_bytes <= dense_cache_max_gb * (1024**3)

        if cache_dense_counts:
            dense_counts = np.asarray(self.counts.toarray(), dtype=np.float32, order="C")
            self._dense_counts_cache = jax.device_put(jnp.asarray(dense_counts))
        else:
            self._dense_counts_cache = None

    def _get_batch(self, rng: jax.Array, Y: sparse.csr_matrix) -> Tuple[jnp.ndarray, ...]:
        """
        Helper function to obtain a batch of data, convert from scipy.sparse to jax.numpy.array.

        Parameters
        ----------
        rng : jax.random.PRNGKey
            Random number generator key.
        Y : scipy.sparse.csr_matrix
            The word counts array.

        Returns
        -------
        tuple
            Y_batch : numpy.ndarray
                Word counts for the batch.
            D_batch : numpy.ndarray
                Indices of documents in the batch.
        """
        D_batch = random.randint(rng, shape=(self.batch_size,), minval=0, maxval=self.D)
        if self._dense_counts_cache is not None:
            Y_batch = self._dense_counts_cache[D_batch]
        else:
            Y_batch = jnp.asarray(Y[np.asarray(D_batch)].toarray(), dtype=jnp.float32)

        # Ensure the shape of Y_batch is (batch_size, V)
        assert Y_batch.shape == (
            self.batch_size,
            self.V,
        ), f"Shape mismatch: {Y_batch.shape} != ({self.batch_size}, {self.V})"

        return Y_batch, D_batch

    def train_step(
        self,
        num_steps: int,
        lr: float,
        random_seed: Optional[int] = None,
        jit_compile: bool = True,
        cache_dense_counts: Optional[bool] = None,
        dense_cache_max_gb: float = 0.75,
    ) -> Dict[str, Any]:
        """
        Train the model using Stochastic Variational Inference (SVI).

        Parameters
        ----------
        num_steps : int
            Number of training iterations. Must be > 0.
        lr : float
            Learning rate for the optimizer. Must be > 0.
        random_seed : int, optional
            Seed for JAX random number generator. If provided, ensures
            reproducible results. Default is None (random initialization).
        jit_compile : bool, optional
            Whether to JIT compile SVI updates. Keep enabled for long runs;
            disable to avoid compile overhead in very short runs.
        cache_dense_counts : bool | None, optional
            If True, cache sparse counts as dense array for faster batching.
            If None, auto-enable when estimated dense matrix size fits in
            ``dense_cache_max_gb``.
        dense_cache_max_gb : float, optional
            Maximum dense cache size in GB used by auto mode.

        Returns
        -------
        dict
            Estimated parameters after training.

        Raises
        ------
        ValueError
            If num_steps <= 0 or lr <= 0.
        """
        if num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {num_steps}")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if dense_cache_max_gb <= 0:
            raise ValueError(f"dense_cache_max_gb must be > 0, got {dense_cache_max_gb}")

        svi_batch = SVI(
            model=self._model, guide=self._guide, optim=adam(lr), loss=TraceMeanField_ELBO()
        )
        svi_batch_update = jit(svi_batch.update) if jit_compile else svi_batch.update

        self._prepare_dense_cache(
            cache_dense_counts=cache_dense_counts, dense_cache_max_gb=dense_cache_max_gb
        )

        # Initialize RNG
        if random_seed is not None:
            init_rng = jax.random.PRNGKey(random_seed)
        else:
            init_rng = jax.random.PRNGKey(0)

        Y_batch, D_batch = self._get_batch(init_rng, self.counts)

        svi_state = svi_batch.init(jax.random.PRNGKey(1), Y_batch=Y_batch, d_batch=D_batch)

        rngs = random.split(jax.random.PRNGKey(2), num_steps)
        pbar = tqdm(range(num_steps))

        for step in pbar:
            Y_batch, D_batch = self._get_batch(rngs[step], self.counts)
            svi_state, loss = svi_batch_update(svi_state, Y_batch=Y_batch, d_batch=D_batch)
            loss = loss / self.D
            self.Metrics.loss.append(float(loss))
            if step % 10 == 0:
                pbar.set_description(
                    "Init loss: "
                    + "{:10.4f}".format(self.Metrics.loss[0])
                    + "; Avg loss (last 10 iter): "
                    + "{:10.4f}".format(jnp.array(self.Metrics.loss[-10:]).mean())
                )

        self.estimated_params = svi_batch.get_params(svi_state)
        self._dense_counts_cache = None

        return self.estimated_params

    def return_topics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the topics for each document.

        Returns
        -------
        categories : np.ndarray
            Array of topic indices for each document (shape: D,).
        E_theta : np.ndarray
            Estimated topic proportions for each document (shape: D, K).

        Raises
        ------
        ValueError
            If model has not been trained yet (no estimated parameters).
        """
        if not self.estimated_params:
            raise ValueError("Model must be trained before calling return_topics()")

        E_theta = self.estimated_params["theta_shape"] / self.estimated_params["theta_rate"]
        return np.argmax(E_theta, axis=1), E_theta

    def return_beta(self) -> pd.DataFrame:
        """
        Return the beta matrix (word-topic associations) for the model.

        Returns
        -------
        pd.DataFrame
            DataFrame with words as index and topics as columns,
            containing word-topic probability estimates.

        Raises
        ------
        ValueError
            If model has not been trained yet (no estimated parameters).
        """
        if not self.estimated_params:
            raise ValueError("Model must be trained before calling return_beta()")

        E_beta = self.estimated_params["beta_shape"] / self.estimated_params["beta_rate"]
        return pd.DataFrame(jnp.transpose(E_beta), index=self.vocab)

    def return_top_words_per_topic(self, n=10):
        beta = self.return_beta()
        return {topic: beta[topic].nlargest(n).index.tolist() for topic in beta}

    def plot_model_loss(
        self, window: int = 10, save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the training loss over time with full and smoothed curves.

        Parameters
        ----------
        window : int, optional
            Window size for moving average smoothing. Default is 10.
        save_path : str, optional
            Path to save the figure.

        Returns
        -------
        tuple of (plt.Figure, plt.Axes)
        """
        if not self.Metrics.loss:
            raise ValueError("No training loss data available. Train the model first.")

        losses = self.Metrics.loss

        with plt.rc_context(self._setup_academic_style()):
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))

            # Full loss curve
            axes[0].plot(losses)
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("ELBO Loss")
            axes[0].set_title("Training Loss Over Time")

            # Smoothed loss (moving average)
            smoothed = pd.Series(losses).rolling(window=window, center=True).mean()
            axes[1].plot(smoothed, linewidth=2)
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("ELBO Loss (smoothed)")
            axes[1].set_title(f"Training Loss (Moving Average, window={window})")

            fig.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")

        print(f"Initial loss: {losses[0]:.4f}")
        print(f"Final loss: {losses[-1]:.4f}")
        print(f"Loss reduction: {(1 - losses[-1] / losses[0]) * 100:.1f}%")

        return fig, axes

    def plot_topic_wordclouds(
        self,
        n_words: int = 50,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot wordclouds for each topic based on beta values.

        Parameters
        ----------
        n_words : int, optional
            Maximum number of words per wordcloud (default 50).
        figsize : tuple, optional
            Figure size ``(width, height)`` (default ``(16, 12)``).
        save_path : str, optional
            Path to save the figure.

        Returns
        -------
        tuple of (plt.Figure, np.ndarray of Axes)
        """
        if not self.estimated_params:
            raise ValueError("Model must be trained before calling plot_topic_wordclouds()")

        beta_df = self.return_beta()
        K = beta_df.shape[1]

        n_cols = int(np.ceil(np.sqrt(K)))
        n_rows = int(np.ceil(K / n_cols))

        with plt.rc_context(self._setup_academic_style()):
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = np.atleast_2d(axes)
            axes_flat = axes.flatten()

            for topic_idx in range(K):
                topic_col = beta_df.iloc[:, topic_idx]
                top_words = topic_col.nlargest(n_words)
                word_freq = dict(top_words)

                if word_freq:
                    wc = WordCloud(
                        width=400,
                        height=300,
                        background_color="white",
                        relative_scaling=0.5,
                        min_font_size=10,
                    ).generate_from_frequencies(word_freq)
                    axes_flat[topic_idx].imshow(wc, interpolation="bilinear")

                col_name = beta_df.columns[topic_idx]
                title = str(col_name) if not isinstance(col_name, int) else f"Topic {col_name}"
                axes_flat[topic_idx].set_title(title, fontsize=11, fontweight="bold")
                axes_flat[topic_idx].axis("off")

            for idx in range(K, len(axes_flat)):
                axes_flat[idx].axis("off")

            fig.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, axes

    # ------------------------------------------------------------------
    # Academic-style plotting
    # ------------------------------------------------------------------

    @staticmethod
    def _setup_academic_style() -> Dict[str, Any]:
        """Return matplotlib rcParams overrides for a clean academic look.

        All plot methods in the library use this via
        ``plt.rc_context(self._setup_academic_style())``.

        Returns
        -------
        dict
            Matplotlib rcParams dictionary.
        """
        return {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.0,
            "axes.grid": True,
            "axes.grid.axis": "x",
            "grid.alpha": 0.15,
            "grid.linewidth": 0.4,
            "grid.color": "#999999",
        }

    # ------------------------------------------------------------------
    # Model summary
    # ------------------------------------------------------------------

    def summary(self, n_top_words: int = 5) -> str:
        """Return a formatted text summary of the fitted model.

        Includes model class name, dimensions, loss trajectory, and
        top words per topic.  Subclasses can extend the output by
        overriding :meth:`_summary_extra`.

        Parameters
        ----------
        n_top_words : int, optional
            Number of top words to show per topic (default 5).

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines: List[str] = []
        sep = "=" * 60
        lines.append(sep)
        lines.append(f"  Model:                    {self.__class__.__name__}")
        lines.append(f"  Topics (K):               {self.K}")
        lines.append(f"  Vocabulary (V):           {self.V}")
        lines.append(f"  Documents (D):            {self.D}")
        lines.append(f"  Batch size:               {self.batch_size}")

        # Subclass-specific info
        extra = self._summary_extra()
        if extra:
            lines.append(extra)

        lines.append(sep)

        # Loss information
        if self.Metrics.loss:
            lines.append(f"  Initial ELBO loss:        {self.Metrics.loss[0]:.4f}")
            lines.append(f"  Final ELBO loss:          {self.Metrics.loss[-1]:.4f}")
            reduction = (1 - self.Metrics.loss[-1] / self.Metrics.loss[0]) * 100
            lines.append(f"  Loss reduction:           {reduction:.1f}%")
            lines.append(f"  Training steps:           {len(self.Metrics.loss)}")
        else:
            lines.append("  Model has not been trained yet.")

        lines.append(sep)

        # Top words per topic
        if self.estimated_params:
            try:
                top_words = self.return_top_words_per_topic(n=n_top_words)
                lines.append("  Top words per topic:")
                for topic, words in top_words.items():
                    label = str(topic)
                    lines.append(f"    {label:>25s}: {', '.join(words)}")
            except Exception:
                lines.append("  (top words not available for this model)")

        lines.append(sep)
        result = "\n".join(lines)
        print(result)
        return result

    def _summary_extra(self) -> str:
        """Hook for subclass-specific summary lines.

        Override in subclasses to append model-specific information
        to :meth:`summary`.  Return an empty string to add nothing.

        Returns
        -------
        str
        """
        return ""

    # ------------------------------------------------------------------
    # Topic-quality metrics
    # ------------------------------------------------------------------

    def compute_topic_coherence(
        self,
        texts: Optional[List[List[str]]] = None,
        metric: str = "c_npmi",
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Compute topic coherence scores (NPMI or UMass).

        Parameters
        ----------
        texts : list of list of str, optional
            Tokenised reference corpus.  If ``None``, word co-occurrence
            is estimated from ``self.counts`` and ``self.vocab``.
        metric : ``{'c_npmi', 'u_mass'}``, optional
            Coherence measure (default ``'c_npmi'``).
        top_n : int, optional
            Number of top words per topic used for the calculation
            (default 10).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``['topic', 'coherence']``.
        """
        beta_df = self.return_beta()
        K = beta_df.shape[1]
        topic_names = [str(c) for c in beta_df.columns]

        # Build top-n word lists per topic
        top_words_per_topic: List[List[str]] = []
        for k in range(K):
            col = beta_df.iloc[:, k]
            top_words_per_topic.append(col.nlargest(top_n).index.tolist())

        # Build co-occurrence from counts matrix
        bow = np.asarray(self.counts.toarray(), dtype=np.float32)
        binary = (bow > 0).astype(np.float32)
        D_ref = binary.shape[0]
        vocab_list = list(self.vocab)
        word2idx = {w: i for i, w in enumerate(vocab_list)}

        if texts is not None:
            # Build binary doc-word matrix from tokenised texts
            V = len(vocab_list)
            rows: List[np.ndarray] = []
            for doc_tokens in texts:
                vec = np.zeros(V, dtype=np.float32)
                for tok in doc_tokens:
                    idx = word2idx.get(tok)
                    if idx is not None:
                        vec[idx] = 1.0
                rows.append(vec)
            binary = np.stack(rows)
            D_ref = binary.shape[0]

        eps = 1e-12
        scores: List[float] = []
        for words in top_words_per_topic:
            indices = [word2idx[w] for w in words if w in word2idx]
            n = len(indices)
            if n < 2:
                scores.append(float("nan"))
                continue

            pairs_total = 0.0
            pair_count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    wi, wj = indices[i], indices[j]
                    d_wi = float(binary[:, wi].sum())
                    d_wj = float(binary[:, wj].sum())
                    d_wi_wj = float((binary[:, wi] * binary[:, wj]).sum())

                    if metric == "u_mass":
                        # UMass: log( (D(w_i, w_j) + eps) / D(w_j) )
                        pairs_total += np.log((d_wi_wj + eps) / (d_wj + eps))
                    else:
                        # NPMI: log2(P(wi,wj) / (P(wi)*P(wj))) / -log2(P(wi,wj))
                        p_wi = d_wi / D_ref
                        p_wj = d_wj / D_ref
                        p_wi_wj = d_wi_wj / D_ref
                        if p_wi_wj < eps:
                            pmi = 0.0
                        else:
                            pmi = np.log2((p_wi_wj + eps) / (p_wi * p_wj + eps))
                        denom = -np.log2(p_wi_wj + eps)
                        npmi = pmi / denom if denom > eps else 0.0
                        pairs_total += npmi
                    pair_count += 1

            scores.append(pairs_total / max(pair_count, 1))

        df = pd.DataFrame({"topic": topic_names, "coherence": scores})
        self.Metrics.coherence_scores = df
        return df

    def compute_topic_diversity(self, top_n: int = 25) -> float:
        """Compute topic diversity (Dieng et al., 2020).

        Measures the fraction of unique words across all topics' top-n
        lists.  Values near 1.0 indicate diverse topics; near 0
        indicates redundancy.

        Parameters
        ----------
        top_n : int, optional
            Number of top words per topic (default 25).

        Returns
        -------
        float
            Topic diversity score in ``[0, 1]``.
        """
        beta_df = self.return_beta()
        K = beta_df.shape[1]
        all_words: List[str] = []
        for k in range(K):
            col = beta_df.iloc[:, k]
            all_words.extend(col.nlargest(top_n).index.tolist())

        diversity = len(set(all_words)) / max(len(all_words), 1)
        self.Metrics.diversity = diversity
        return diversity

    # ------------------------------------------------------------------
    # Additional post-fitting plots
    # ------------------------------------------------------------------

    def plot_topic_prevalence(self, save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Horizontal bar chart of mean topic prevalence across documents.

        Parameters
        ----------
        save_path : str, optional
            Path to save the figure.

        Returns
        -------
        tuple of (plt.Figure, plt.Axes)
        """
        if not self.estimated_params:
            raise ValueError("Model must be trained first.")

        _, E_theta = self.return_topics()
        mean_prev = np.asarray(E_theta).mean(axis=0)

        beta_df = self.return_beta()
        topic_labels = [str(c) for c in beta_df.columns]

        order = np.argsort(mean_prev)

        with plt.rc_context(self._setup_academic_style()):
            fig, ax = plt.subplots(figsize=(6, max(3, 0.35 * len(topic_labels))))
            ax.barh(
                np.arange(len(order)),
                mean_prev[order],
                color="#4E79A7",
                edgecolor="white",
                linewidth=0.3,
            )
            ax.set_yticks(np.arange(len(order)))
            ax.set_yticklabels([topic_labels[i] for i in order])
            ax.set_xlabel("Mean topic weight")
            ax.set_title("Topic prevalence")
            fig.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax

    def plot_topic_correlation(
        self, save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Heatmap of pairwise cosine similarity between topic-word vectors.

        Parameters
        ----------
        save_path : str, optional
            Path to save the figure.

        Returns
        -------
        tuple of (plt.Figure, plt.Axes)
        """
        if not self.estimated_params:
            raise ValueError("Model must be trained first.")

        beta_df = self.return_beta()
        beta_mat = beta_df.values.T  # (K, V)
        norms = np.linalg.norm(beta_mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = beta_mat / norms
        sim = normed @ normed.T

        topic_labels = [str(c) for c in beta_df.columns]

        with plt.rc_context(self._setup_academic_style()):
            fig, ax = plt.subplots(
                figsize=(max(5, 0.6 * len(topic_labels)), max(4, 0.5 * len(topic_labels)))
            )
            im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(topic_labels)))
            ax.set_xticklabels(topic_labels, rotation=45, ha="right")
            ax.set_yticks(range(len(topic_labels)))
            ax.set_yticklabels(topic_labels)
            ax.set_title("Topic similarity (cosine)")
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax

    def plot_document_topic_heatmap(
        self,
        n_docs: int = 50,
        sort_by_topic: bool = False,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Heatmap of document-topic proportions for a subset of documents.

        Parameters
        ----------
        n_docs : int, optional
            Number of documents to display (default 50).
        sort_by_topic : bool, optional
            If True, sort documents by their dominant topic (default False).
        save_path : str, optional
            Path to save the figure.

        Returns
        -------
        tuple of (plt.Figure, plt.Axes)
        """
        if not self.estimated_params:
            raise ValueError("Model must be trained first.")

        cats, E_theta = self.return_topics()
        E_theta = np.asarray(E_theta)

        n_docs = min(n_docs, E_theta.shape[0])

        if sort_by_topic:
            order = np.argsort(cats)[:n_docs]
        else:
            order = np.arange(n_docs)

        subset = E_theta[order]
        # Row-normalise for visualisation
        row_sums = subset.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        subset_norm = subset / row_sums

        beta_df = self.return_beta()
        topic_labels = [str(c) for c in beta_df.columns]

        with plt.rc_context(self._setup_academic_style()):
            fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(topic_labels)), max(6, 0.15 * n_docs)))
            im = ax.imshow(subset_norm, aspect="auto", cmap="YlOrRd", interpolation="nearest")
            ax.set_xlabel("Topic")
            ax.set_ylabel("Document")
            ax.set_xticks(range(len(topic_labels)))
            ax.set_xticklabels(topic_labels, rotation=45, ha="right")
            ax.set_title(f"Document-topic proportions (n={n_docs})")
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax
