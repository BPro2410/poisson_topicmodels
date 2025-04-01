from dataclasses import dataclass

@dataclass
class Metrics:
    """
    Data class for storing metrics.
    
    Attributes
    ----------
    loss : list
        List of loss values during training.
    """
    loss: list

