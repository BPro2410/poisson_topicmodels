from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

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

    @abstractmethod
    def _model(self, Y_batch: Any, d_batch: Any) -> None:
        """Define the probabilistic model."""
        pass

    @abstractmethod
    def _guide(self, Y_batch: Any, d_batch: Any) -> None:
        """Define the variational guide."""
        pass

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
        D_batch = random.choice(rng, jnp.arange(self.D), shape=(self.batch_size,))
        # Y_batch = jax.device_put(jnp.array(Y[D_batch].toarray()), jax.devices("cpu")[0])
        Y_batch = jnp.array(Y[D_batch].toarray())

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

        svi_batch = SVI(
            model=self._model, guide=self._guide, optim=adam(lr), loss=TraceMeanField_ELBO()
        )
        svi_batch_update = jit(svi_batch.update)

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
            Window size for moving average smoothing. Default is 100.
        """
        if not self.Metrics.loss:
            raise ValueError("No training loss data available. Train the model first.")

        losses = self.Metrics.loss

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # Full loss curve
        axes[0].plot(losses)
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("ELBO Loss")
        axes[0].set_title("Training Loss Over Time")
        axes[0].grid(True, alpha=0.3)

        # Smoothed loss (moving average)
        smoothed = pd.Series(losses).rolling(window=window, center=True).mean()
        axes[1].plot(smoothed, linewidth=2)
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("ELBO Loss (smoothed)")
        axes[1].set_title(f"Training Loss (Moving Average, window={window})")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        # plt.show()

        print(f"Initial loss: {losses[0]:.4f}")
        print(f"Final loss: {losses[-1]:.4f}")
        print(f"Loss reduction: {(1 - losses[-1] / losses[0]) * 100:.1f}%")

        if save_path:
            plt.savefig(save_path)

        return fig, axes

    def plot_topic_wordclouds(
        self,
        n_words: int = 50,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot wordclouds for each topic based on beta values (word-topic associations).

        Parameters
        ----------
        n_words : int, optional
            Maximum number of words to display in each wordcloud. Default is 50.
        figsize : tuple, optional
            Figure size as (width, height). Default is (16, 12).
        save_path : str, optional
            Path to save the figure. If None, figure is not saved. Default is None.

        Returns
        -------
        fig : plt.Figure
            The matplotlib figure object.
        axes : np.ndarray
            Array of axes objects for the subplots.

        Raises
        ------
        ValueError
            If model has not been trained yet (no estimated parameters).
        """
        if not self.estimated_params:
            raise ValueError("Model must be trained before calling plot_topic_wordclouds()")

        beta_df = self.return_beta()
        K = beta_df.shape[1]

        # Calculate grid dimensions
        n_cols = int(np.ceil(np.sqrt(K)))
        n_rows = int(np.ceil(K / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()  # Flatten to 1D array for easier iteration

        for topic_idx in range(K):
            # Get top words and their frequencies for this topic
            topic_col = beta_df.iloc[:, topic_idx]
            top_words = topic_col.nlargest(n_words)

            # Create frequency dictionary for wordcloud
            word_freq = dict(top_words)

            # Generate wordcloud
            if word_freq:
                wc = WordCloud(
                    width=400,
                    height=300,
                    background_color="white",
                    relative_scaling=0.5,
                    min_font_size=10,
                ).generate_from_frequencies(word_freq)

                axes[topic_idx].imshow(wc, interpolation="bilinear")

            axes[topic_idx].set_title(f"Topic {topic_idx}", fontsize=14, fontweight="bold")
            axes[topic_idx].axis("off")

        # Hide unused subplots
        for idx in range(K, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, axes
