from dataclasses import dataclass


@dataclass
class Box:
    id: int
    document: int
    name: str
    is_alphabetic: bool
    is_numeric: bool
    mean_length: int
    start_x_norm: float
    start_y_norm: float
    end_x_norm: float
    end_y_norm: float