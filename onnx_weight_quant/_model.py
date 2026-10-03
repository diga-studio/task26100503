"""In-memory representation of the ONNX subset handled by this library."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# TensorProto.DataType values used here.
TENSOR_FLOAT = 1
TENSOR_INT8 = 3
TENSOR_INT32 = 6
TENSOR_INT64 = 7

# AttributeProto.AttributeType values.
ATTR_FLOAT = 1
ATTR_INT = 2
ATTR_STRING = 3
ATTR_FLOATS = 6
ATTR_INTS = 7


@dataclass
class Attribute:
    name: str
    kind: int
    value: object  # float | int | bytes | List[float] | List[int]


@dataclass
class Tensor:
    name: str
    dtype: int
    dims: List[int]
    raw_data: bytes = b""


@dataclass
class Node:
    op_type: str
    inputs: List[str]
    outputs: List[str]
    name: str = ""
    attributes: Dict[str, Attribute] = field(default_factory=dict)


@dataclass
class ValueInfo:
    name: str
    elem_type: int = TENSOR_FLOAT
    # Each dim is an int (static) or a str (dynamic dim_param).
    shape: Optional[List[object]] = None


@dataclass
class Graph:
    name: str = ""
    nodes: List[Node] = field(default_factory=list)
    initializers: List[Tensor] = field(default_factory=list)
    inputs: List[ValueInfo] = field(default_factory=list)
    outputs: List[ValueInfo] = field(default_factory=list)
    value_infos: List[ValueInfo] = field(default_factory=list)


@dataclass
class Model:
    ir_version: int = 8
    producer_name: str = ""
    graph: Graph = field(default_factory=Graph)
    # (domain, version) pairs; "" is the default ai.onnx domain.
    opset_imports: List[Tuple[str, int]] = field(default_factory=lambda: [("", 17)])
