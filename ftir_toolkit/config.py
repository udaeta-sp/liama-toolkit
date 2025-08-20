from dataclasses import dataclass

@dataclass(frozen=True)
class BandsConfig:
    # diagnostic bands: (center_cm1, fullwidth_cm1)
    triple = [(1602.0, 100.0), (1313.0, 100.0), (778.0, 100.0)]
    colors = ["green", "cyan", "red"]

@dataclass(frozen=True)
class DisplayConfig:
    decimals_alturas: int = 4  # fixed decimals per column in heights table
