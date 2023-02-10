import puzzler
from dataclasses import dataclass

AffineTransform = puzzler.align.AffineTransform

@dataclass
class Raft:

    pieces: dict[str,AffineTransform]
