from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Metrics:
    """
    Data class for storing training metrics.

    Tracks model performance during training by recording loss values
    at each iteration.

    Attributes
    ----------
    loss : List[float]
        List of loss values for each training iteration.

    Examples
    --------
    >>> metrics = Metrics(loss=[])
    >>> metrics.loss.append(0.5)
    >>> len(metrics.loss)
    1
    """

    loss: List[Any] = field(default_factory=list)

    def reset(self) -> None:
        """Reset all metrics to empty state."""
        self.loss = []

    def last_loss(self) -> Any:
        """Get the most recent loss value."""
        return self.loss[-1] if self.loss else None
