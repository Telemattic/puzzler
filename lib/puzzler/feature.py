import puzzler.geometry
from dataclasses import dataclass

@dataclass
class Feature:

    fit_indexes: tuple

@dataclass
class Tab(Feature):

    ellipse: puzzler.geometry.Ellipse
    indent: bool
    tangent_indexes: tuple

@dataclass
class Edge(Feature):

    line: puzzler.geometry.Line
