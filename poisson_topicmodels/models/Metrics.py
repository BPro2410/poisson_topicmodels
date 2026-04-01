from dataclasses import dataclass, field
from typing import Any, List, Optional

import pandas as pd


@dataclass
class Metrics:
    """
    Data class for storing training and evaluation metrics.

    Tracks model performance during training by recording loss values
    at each iteration, and stores topic-quality metrics computed
    post-fitting.

    Attributes
    ----------
    loss : List[float]
        List of loss values for each training iteration.
    coherence_scores : pd.DataFrame or None
        Per-topic coherence scores computed by
        :meth:`NumpyroModel.compute_topic_coherence`.
    diversity : float or None
        Topic diversity score computed by
        :meth:`NumpyroModel.compute_topic_diversity`.

    Examples
    --------
    >>> metrics = Metrics(loss=[])
    >>> metrics.loss.append(0.5)
    >>> len(metrics.loss)
    1
    """

    loss: List[Any] = field(default_factory=list)
    coherence_scores: Optional[pd.DataFrame] = field(default=None, repr=False)
    diversity: Optional[float] = None

    def reset(self) -> None:
        """Reset all metrics to empty state."""
        self.loss = []
        self.coherence_scores = None
        self.diversity = None

    def last_loss(self) -> Any:
        """Get the most recent loss value."""
        return self.loss[-1] if self.loss else None
