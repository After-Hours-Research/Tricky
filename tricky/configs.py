from dataclasses import dataclass


@dataclass
class DataConfig:
    """Data configuration"""
    url: str
    text_column: str
    