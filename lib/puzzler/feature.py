import puzzler.geometry
from dataclasses import dataclass

@dataclass
class Feature:
    pass

@dataclass
class Tab(Feature):

    fit_indexes: tuple[int,int]
    ellipse: puzzler.geometry.Ellipse
    indent: bool
    tangent_indexes: tuple[int,int] = (None, None)
    
@dataclass
class Edge(Feature):

    fit_indexes: tuple[int,int]
    line: puzzler.geometry.Line
