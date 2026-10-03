"""Minimal ONNX protobuf wire-format reader/writer (no `onnx` dependency).

Supports the subset of onnx.proto3 needed for single-graph models with
MatMul / Gemm / Identity / Transpose / Add / Relu / DequantizeLinear nodes
and embedded (raw_data / float_data) tensor payloads.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

# TensorProto.DataType values used here.
FLOAT = 1
INT8 = 3
INT32 = 6
INT64 = 7

# AttributeProto.AttributeType values.
ATTR_FLOAT = 1
ATTR_INT = 2
ATTR_STRING = 3
ATTR_TENSOR = 4
ATTR_FLOATS = 6
ATTR_INTS = 7
ATTR_STRINGS = 8


# ---------------------------------------------------------------------------
# Low-level wire format helpers
# ---------------------------------------------------------------------------

def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _write_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _parse_fields(buf: bytes) -> list[tuple[int, int, object]]:
    """Parse a message into (field_number, wire_type, value) triples.

    Values are ints for varints, bytes for length-delimited fields,
    and ints for 32/64-bit fixed fields (raw little-endian int).
    """
    fields: list[tuple[int, int, object]] = []
    pos = 0
    n = len(buf)
    while pos < n:
        key, pos = _read_varint(buf, pos)
        number = key >> 3
        wire = key & 7
        if wire == 0:
            value, pos = _read_varint(buf, pos)
        elif wire == 1:
            value = struct.unpack_from("<Q", buf, pos)[0]
            pos += 8
        elif wire == 2:
            length, pos = _read_varint(buf, pos)
            value = buf[pos : pos + length]
            pos += length
        elif wire == 5:
            value = struct.unpack_from("<I", buf, pos)[0]
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        fields.append((number, wire, value))
    return fields


def _tag(number: int, wire: int) -> bytes:
    return _write_varint((number << 3) | wire)


def _enc_varint_field(number: int, value: int) -> bytes:
    return _tag(number, 0) + _write_varint(value)


def _enc_svarint_field(number: int, value: int) -> bytes:
    # int64 negative values are sign-extended to 64 bits.
    return _enc_varint_field(number, value & 0xFFFFFFFFFFFFFFFF)


def _enc_fixed32_field(number: int, value: float) -> bytes:
    return _tag(number, 5) + struct.pack("<f", value)


def _enc_bytes_field(number: int, value: bytes) -> bytes:
    return _tag(number, 2) + _write_varint(len(value)) + value


def _enc_str_field(number: int, value: str) -> bytes:
    return _enc_bytes_field(number, value.encode("utf-8"))


def _enc_packed_floats(number: int, values: list[float]) -> bytes:
    payload = struct.pack(f"<{len(values)}f", *values)
    return _enc_bytes_field(number, payload)


def _enc_packed_ints(number: int, values: list[int]) -> bytes:
    payload = b"".join(_write_varint(v & 0xFFFFFFFFFFFFFFFF) for v in values)
    return _enc_bytes_field(number, payload)


def _decode_packed_varints(buf: bytes) -> list[int]:
    out = []
    pos = 0
    while pos < len(buf):
        v, pos = _read_varint(buf, pos)
        # Interpret as signed 64-bit two's complement.
        if v >= 1 << 63:
            v -= 1 << 64
        out.append(v)
    return out


# ---------------------------------------------------------------------------
# Message dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Tensor:
    name: str = ""
    dims: list[int] = field(default_factory=list)
    data_type: int = 0
    raw_data: bytes | None = None
    float_data: list[float] | None = None


@dataclass
class Attribute:
    name: str = ""
    type: int = 0
    f: float | None = None
    i: int | None = None
    s: bytes | None = None
    t: Tensor | None = None
    floats: list[float] | None = None
    ints: list[int] | None = None
    strings: list[bytes] | None = None


@dataclass
class Node:
    op_type: str = ""
    name: str = ""
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    attributes: list[Attribute] = field(default_factory=list)
    domain: str = ""

    def get_attribute(self, name: str, default=None):
        for attr in self.attributes:
            if attr.name == name:
                if attr.type == ATTR_FLOAT:
                    return attr.f
                if attr.type == ATTR_INT:
                    return attr.i
                if attr.type == ATTR_STRING:
                    return attr.s
                if attr.type == ATTR_INTS:
                    return attr.ints
                if attr.type == ATTR_FLOATS:
                    return attr.floats
                return attr
        return default


@dataclass
class ValueInfo:
    name: str = ""
    elem_type: int = 0
    shape: list[int | str | None] = field(default_factory=list)


@dataclass
class Graph:
    name: str = ""
    nodes: list[Node] = field(default_factory=list)
    initializers: list[Tensor] = field(default_factory=list)
    inputs: list[ValueInfo] = field(default_factory=list)
    outputs: list[ValueInfo] = field(default_factory=list)
    value_info: list[ValueInfo] = field(default_factory=list)


@dataclass
class Model:
    ir_version: int = 0
    producer_name: str = ""
    graph: Graph = field(default_factory=Graph)
    opset_imports: list[tuple[str, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def _decode_tensor(buf: bytes) -> Tensor:
    tensor = Tensor()
    for number, wire, value in _parse_fields(buf):
        if number == 1:  # dims
            if wire == 2:
                tensor.dims.extend(_decode_packed_varints(value))
            else:
                v = value if value < (1 << 63) else value - (1 << 64)
                tensor.dims.append(v)
        elif number == 2:
            tensor.data_type = value
        elif number == 4:  # float_data
            if wire == 2:
                count = len(value) // 4
                tensor.float_data = list(struct.unpack(f"<{count}f", value))
            else:
                if tensor.float_data is None:
                    tensor.float_data = []
                tensor.float_data.append(struct.unpack("<f", struct.pack("<I", value))[0])
        elif number == 8:
            tensor.name = value.decode("utf-8")
        elif number == 9:
            tensor.raw_data = value
    return tensor


def _decode_attribute(buf: bytes) -> Attribute:
    attr = Attribute()
    for number, wire, value in _parse_fields(buf):
        if number == 1:
            attr.name = value.decode("utf-8")
        elif number == 2:
            attr.f = struct.unpack("<f", struct.pack("<I", value))[0]
        elif number == 3:
            v = value if value < (1 << 63) else value - (1 << 64)
            attr.i = v
        elif number == 4:
            attr.s = value
        elif number == 5:
            attr.t = _decode_tensor(value)
        elif number == 7:  # floats
            if wire == 2:
                count = len(value) // 4
                attr.floats = list(struct.unpack(f"<{count}f", value))
            else:
                if attr.floats is None:
                    attr.floats = []
                attr.floats.append(struct.unpack("<f", struct.pack("<I", value))[0])
        elif number == 8:  # ints
            if wire == 2:
                attr.ints = _decode_packed_varints(value)
            else:
                if attr.ints is None:
                    attr.ints = []
                v = value if value < (1 << 63) else value - (1 << 64)
                attr.ints.append(v)
        elif number == 9:
            if attr.strings is None:
                attr.strings = []
            attr.strings.append(value)
        elif number == 20:
            attr.type = value
    return attr


def _decode_node(buf: bytes) -> Node:
    node = Node()
    for number, wire, value in _parse_fields(buf):
        if number == 1:
            node.inputs.append(value.decode("utf-8"))
        elif number == 2:
            node.outputs.append(value.decode("utf-8"))
        elif number == 3:
            node.name = value.decode("utf-8")
        elif number == 4:
            node.op_type = value.decode("utf-8")
        elif number == 5:
            node.attributes.append(_decode_attribute(value))
        elif number == 7:
            node.domain = value.decode("utf-8")
    return node


def _decode_value_info(buf: bytes) -> ValueInfo:
    vi = ValueInfo()
    for number, wire, value in _parse_fields(buf):
        if number == 1:
            vi.name = value.decode("utf-8")
        elif number == 2:  # TypeProto
            for tnum, _twire, tval in _parse_fields(value):
                if tnum == 1:  # tensor_type
                    for snum, _swire, sval in _parse_fields(tval):
                        if snum == 1:
                            vi.elem_type = sval
                        elif snum == 2:  # shape
                            for dnum, _dwire, dval in _parse_fields(sval):
                                if dnum == 1:  # dim
                                    dim_value = None
                                    for fnum, _fwire, fval in _parse_fields(dval):
                                        if fnum == 1:
                                            dim_value = fval if fval < (1 << 63) else fval - (1 << 64)
                                        elif fnum == 2:
                                            dim_value = fval.decode("utf-8")
                                    vi.shape.append(dim_value)
    return vi


def _decode_graph(buf: bytes) -> Graph:
    graph = Graph()
    for number, wire, value in _parse_fields(buf):
        if number == 1:
            graph.nodes.append(_decode_node(value))
        elif number == 2:
            graph.name = value.decode("utf-8")
        elif number == 5:
            graph.initializers.append(_decode_tensor(value))
        elif number == 11:
            graph.inputs.append(_decode_value_info(value))
        elif number == 12:
            graph.outputs.append(_decode_value_info(value))
        elif number == 13:
            graph.value_info.append(_decode_value_info(value))
    return graph


def load_model(path: str) -> Model:
    with open(path, "rb") as fh:
        buf = fh.read()
    return loads_model(buf)


def loads_model(buf: bytes) -> Model:
    model = Model()
    for number, wire, value in _parse_fields(buf):
        if number == 1:
            model.ir_version = value
        elif number == 2:
            model.producer_name = value.decode("utf-8")
        elif number == 7:
            model.graph = _decode_graph(value)
        elif number == 8:  # opset_import
            domain = ""
            version = 0
            for onum, _owire, oval in _parse_fields(value):
                if onum == 1:
                    domain = oval.decode("utf-8")
                elif onum == 2:
                    version = oval
            model.opset_imports.append((domain, version))
    return model


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _encode_tensor(tensor: Tensor) -> bytes:
    out = bytearray()
    if tensor.dims:
        out += _enc_packed_ints(1, tensor.dims)
    if tensor.data_type:
        out += _enc_varint_field(2, tensor.data_type)
    if tensor.float_data:
        out += _enc_packed_floats(4, tensor.float_data)
    if tensor.name:
        out += _enc_str_field(8, tensor.name)
    if tensor.raw_data is not None:
        out += _enc_bytes_field(9, tensor.raw_data)
    return bytes(out)


def _encode_attribute(attr: Attribute) -> bytes:
    out = bytearray()
    out += _enc_str_field(1, attr.name)
    if attr.type == ATTR_FLOAT and attr.f is not None:
        out += _enc_fixed32_field(2, attr.f)
    elif attr.type == ATTR_INT and attr.i is not None:
        out += _enc_svarint_field(3, attr.i)
    elif attr.type == ATTR_STRING and attr.s is not None:
        out += _enc_bytes_field(4, attr.s)
    elif attr.type == ATTR_TENSOR and attr.t is not None:
        out += _enc_bytes_field(5, _encode_tensor(attr.t))
    elif attr.type == ATTR_FLOATS and attr.floats is not None:
        out += _enc_packed_floats(7, attr.floats)
    elif attr.type == ATTR_INTS and attr.ints is not None:
        out += _enc_packed_ints(8, attr.ints)
    elif attr.type == ATTR_STRINGS and attr.strings is not None:
        for s in attr.strings:
            out += _enc_bytes_field(9, s)
    if attr.type:
        out += _enc_varint_field(20, attr.type)
    return bytes(out)


def _encode_node(node: Node) -> bytes:
    out = bytearray()
    for name in node.inputs:
        out += _enc_str_field(1, name)
    for name in node.outputs:
        out += _enc_str_field(2, name)
    if node.name:
        out += _enc_str_field(3, node.name)
    out += _enc_str_field(4, node.op_type)
    for attr in node.attributes:
        out += _enc_bytes_field(5, _encode_attribute(attr))
    if node.domain:
        out += _enc_str_field(7, node.domain)
    return bytes(out)


def _encode_value_info(vi: ValueInfo) -> bytes:
    shape = bytearray()
    for dim in vi.shape:
        dim_msg = bytearray()
        if isinstance(dim, str):
            dim_msg += _enc_str_field(2, dim)
        elif dim is not None:
            dim_msg += _enc_svarint_field(1, dim)
        shape += _enc_bytes_field(1, bytes(dim_msg))
    tensor_type = _enc_varint_field(1, vi.elem_type) + _enc_bytes_field(2, bytes(shape))
    type_proto = _enc_bytes_field(1, tensor_type)
    out = bytearray()
    out += _enc_str_field(1, vi.name)
    out += _enc_bytes_field(2, type_proto)
    return bytes(out)


def _encode_graph(graph: Graph) -> bytes:
    out = bytearray()
    for node in graph.nodes:
        out += _enc_bytes_field(1, _encode_node(node))
    if graph.name:
        out += _enc_str_field(2, graph.name)
    for tensor in graph.initializers:
        out += _enc_bytes_field(5, _encode_tensor(tensor))
    for vi in graph.inputs:
        out += _enc_bytes_field(11, _encode_value_info(vi))
    for vi in graph.outputs:
        out += _enc_bytes_field(12, _encode_value_info(vi))
    for vi in graph.value_info:
        out += _enc_bytes_field(13, _encode_value_info(vi))
    return bytes(out)


def dumps_model(model: Model) -> bytes:
    out = bytearray()
    if model.ir_version:
        out += _enc_varint_field(1, model.ir_version)
    if model.producer_name:
        out += _enc_str_field(2, model.producer_name)
    out += _enc_bytes_field(7, _encode_graph(model.graph))
    for domain, version in model.opset_imports:
        opset = bytearray()
        if domain:
            opset += _enc_str_field(1, domain)
        opset += _enc_varint_field(2, version)
        out += _enc_bytes_field(8, bytes(opset))
    return bytes(out)


def save_model(model: Model, path: str) -> None:
    with open(path, "wb") as fh:
        fh.write(dumps_model(model))
