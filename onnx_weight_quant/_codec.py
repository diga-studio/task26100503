"""Decode/encode ONNX ModelProto bytes for the supported message subset."""

from __future__ import annotations

import struct
from typing import List

import numpy as np

from . import _wire as w
from ._model import (
    ATTR_FLOAT,
    ATTR_FLOATS,
    ATTR_INT,
    ATTR_INTS,
    ATTR_STRING,
    TENSOR_FLOAT,
    TENSOR_INT8,
    TENSOR_INT32,
    TENSOR_INT64,
    Attribute,
    Graph,
    Model,
    Node,
    Tensor,
    ValueInfo,
)

_DTYPE_TO_NP = {
    TENSOR_FLOAT: np.float32,
    TENSOR_INT8: np.int8,
    TENSOR_INT32: np.int32,
    TENSOR_INT64: np.int64,
}
_NP_TO_DTYPE = {np.dtype(v): k for k, v in _DTYPE_TO_NP.items()}


def tensor_to_array(tensor: Tensor) -> np.ndarray:
    """Return the tensor payload as a numpy array (from raw_data)."""
    np_dtype = _DTYPE_TO_NP[tensor.dtype]
    arr = np.frombuffer(tensor.raw_data, dtype=np.dtype(np_dtype).newbyteorder("<"))
    return arr.reshape(tensor.dims).copy()


def array_to_tensor(name: str, arr: np.ndarray) -> Tensor:
    """Build a Tensor storing the array as little-endian raw_data."""
    arr = np.ascontiguousarray(arr)
    np_dtype = np.dtype(arr.dtype)
    if np_dtype not in _NP_TO_DTYPE:
        raise ValueError(f"unsupported array dtype {np_dtype}")
    if np_dtype.byteorder == ">":
        arr = arr.byteswap()
    return Tensor(
        name=name,
        dtype=_NP_TO_DTYPE[np_dtype],
        dims=[int(d) for d in arr.shape],
        raw_data=arr.tobytes(order="C"),
    )


def _signed(value: int, bits: int) -> int:
    return value - (1 << bits) if value >= (1 << (bits - 1)) else value


def _decode_tensor(fields: List[w.Field]) -> Tensor:
    name = ""
    dtype = TENSOR_FLOAT
    dims: List[int] = []
    raw = b""
    typed: List[int] = []
    typed_kind = 0
    floats: List[float] = []
    for num, wire, val in fields:
        if num == 1:  # dims
            if wire == w.WIRE_LENGTH_DELIMITED:
                dims = w.unpack_varints(val)
            elif wire == w.WIRE_VARINT:
                dims = [int(val)]
        elif num == 2:
            dtype = int(val)
        elif num == 4:  # float_data (packed fixed32)
            data = val if wire == w.WIRE_LENGTH_DELIMITED else bytes(val)
            floats = list(struct.unpack("<%df" % (len(data) // 4), data))
            typed_kind = TENSOR_FLOAT
        elif num == 5:  # int32_data (packed varint)
            typed = [_signed(v, 64) for v in w.unpack_varints(val)]
            typed_kind = TENSOR_INT32
        elif num == 7:  # int64_data (packed varint)
            typed = [_signed(v, 64) for v in w.unpack_varints(val)]
            typed_kind = TENSOR_INT64
        elif num == 8:
            name = val.decode("utf-8")
        elif num == 9:
            raw = bytes(val)
    if not raw and typed_kind:
        if typed_kind == TENSOR_FLOAT:
            raw = np.asarray(floats, dtype=np.float32).tobytes()
        elif typed_kind == TENSOR_INT32:
            raw = np.asarray(typed, dtype=np.int32).tobytes()
        else:
            raw = np.asarray(typed, dtype=np.int64).tobytes()
    return Tensor(name=name, dtype=dtype, dims=dims, raw_data=raw)


def _encode_tensor(tensor: Tensor) -> bytes:
    out = bytearray()
    if tensor.dims:
        out += w.encode_field(1, w.WIRE_LENGTH_DELIMITED, w.pack_varints(tensor.dims))
    out += w.encode_field(2, w.WIRE_VARINT, tensor.dtype)
    if tensor.name:
        out += w.encode_field(8, w.WIRE_LENGTH_DELIMITED, tensor.name.encode("utf-8"))
    if tensor.raw_data:
        out += w.encode_field(9, w.WIRE_LENGTH_DELIMITED, tensor.raw_data)
    return bytes(out)


def _decode_attribute(fields: List[w.Field]) -> Attribute:
    name = ""
    kind = 0
    value: object = 0
    for num, wire, val in fields:
        if num == 1:
            name = val.decode("utf-8")
        elif num == 2:
            value = struct.unpack("<f", bytes(val))[0]
            kind = ATTR_FLOAT
        elif num == 3:
            value = _signed(int(val), 64)
            kind = ATTR_INT
        elif num == 4:
            value = bytes(val)
            kind = ATTR_STRING
        elif num == 7:
            data = val if wire == w.WIRE_LENGTH_DELIMITED else bytes(val)
            value = list(struct.unpack("<%df" % (len(data) // 4), data))
            kind = ATTR_FLOATS
        elif num == 8:
            value = [_signed(v, 64) for v in w.unpack_varints(val)]
            kind = ATTR_INTS
        elif num == 20:
            kind = int(val)
    return Attribute(name=name, kind=kind, value=value)


def _encode_attribute(attr: Attribute) -> bytes:
    out = bytearray()
    out += w.encode_field(1, w.WIRE_LENGTH_DELIMITED, attr.name.encode("utf-8"))
    if attr.kind == ATTR_FLOAT:
        out += w.encode_field(2, w.WIRE_FIXED32, struct.pack("<f", float(attr.value)))
    elif attr.kind == ATTR_INT:
        out += w.encode_field(3, w.WIRE_VARINT, int(attr.value))
    elif attr.kind == ATTR_STRING:
        out += w.encode_field(4, w.WIRE_LENGTH_DELIMITED, bytes(attr.value))
    elif attr.kind == ATTR_FLOATS:
        out += w.encode_field(
            7, w.WIRE_LENGTH_DELIMITED,
            struct.pack("<%df" % len(attr.value), *[float(v) for v in attr.value]),
        )
    elif attr.kind == ATTR_INTS:
        out += w.encode_field(
            8, w.WIRE_LENGTH_DELIMITED,
            w.pack_varints([int(v) for v in attr.value]),
        )
    else:
        raise ValueError(f"unsupported attribute kind {attr.kind}")
    out += w.encode_field(20, w.WIRE_VARINT, attr.kind)
    return bytes(out)


def _decode_node(fields: List[w.Field]) -> Node:
    node = Node(op_type="", inputs=[], outputs=[])
    for num, wire, val in fields:
        if num == 1:
            node.inputs.append(val.decode("utf-8"))
        elif num == 2:
            node.outputs.append(val.decode("utf-8"))
        elif num == 3:
            node.name = val.decode("utf-8")
        elif num == 4:
            node.op_type = val.decode("utf-8")
        elif num == 5:
            attr = _decode_attribute(w.decode_message(val))
            node.attributes[attr.name] = attr
    return node


def _encode_node(node: Node) -> bytes:
    out = bytearray()
    for name in node.inputs:
        out += w.encode_field(1, w.WIRE_LENGTH_DELIMITED, name.encode("utf-8"))
    for name in node.outputs:
        out += w.encode_field(2, w.WIRE_LENGTH_DELIMITED, name.encode("utf-8"))
    if node.name:
        out += w.encode_field(3, w.WIRE_LENGTH_DELIMITED, node.name.encode("utf-8"))
    out += w.encode_field(4, w.WIRE_LENGTH_DELIMITED, node.op_type.encode("utf-8"))
    for attr in node.attributes.values():
        out += w.encode_field(5, w.WIRE_LENGTH_DELIMITED, _encode_attribute(attr))
    return bytes(out)


def _decode_value_info(fields: List[w.Field]) -> ValueInfo:
    info = ValueInfo(name="")
    for num, wire, val in fields:
        if num == 1:
            info.name = val.decode("utf-8")
        elif num == 2:  # TypeProto
            for tnum, _, tval in w.decode_message(val):
                if tnum != 1:  # tensor_type
                    continue
                for fnum, _, fval in w.decode_message(tval):
                    if fnum == 1:
                        info.elem_type = int(fval)
                    elif fnum == 2:  # shape
                        shape: List[object] = []
                        for snum, _, sval in w.decode_message(fval):
                            if snum != 1:  # dim
                                continue
                            dim: object = 0
                            for dnum, _, dval in w.decode_message(sval):
                                if dnum == 1:
                                    dim = _signed(int(dval), 64)
                                elif dnum == 2:
                                    dim = dval.decode("utf-8")
                            shape.append(dim)
                        info.shape = shape
    return info


def _encode_value_info(info: ValueInfo) -> bytes:
    tensor_type = bytearray()
    tensor_type += w.encode_field(1, w.WIRE_VARINT, info.elem_type)
    if info.shape is not None:
        shape = bytearray()
        for dim in info.shape:
            dim_msg = bytearray()
            if isinstance(dim, str):
                dim_msg += w.encode_field(2, w.WIRE_LENGTH_DELIMITED, dim.encode("utf-8"))
            else:
                dim_msg += w.encode_field(1, w.WIRE_VARINT, int(dim))
            shape += w.encode_field(1, w.WIRE_LENGTH_DELIMITED, bytes(dim_msg))
        tensor_type += w.encode_field(2, w.WIRE_LENGTH_DELIMITED, bytes(shape))
    type_proto = w.encode_field(1, w.WIRE_LENGTH_DELIMITED, bytes(tensor_type))
    out = bytearray()
    out += w.encode_field(1, w.WIRE_LENGTH_DELIMITED, info.name.encode("utf-8"))
    out += w.encode_field(2, w.WIRE_LENGTH_DELIMITED, type_proto)
    return bytes(out)


def decode_model(buf: bytes) -> Model:
    model = Model(opset_imports=[])
    graph = Graph()
    for num, wire, val in w.decode_message(buf):
        if num == 1:
            model.ir_version = int(val)
        elif num == 2:
            model.producer_name = val.decode("utf-8")
        elif num == 7:
            for gnum, _, gval in w.decode_message(val):
                if gnum == 1:
                    graph.nodes.append(_decode_node(w.decode_message(gval)))
                elif gnum == 2:
                    graph.name = gval.decode("utf-8")
                elif gnum == 5:
                    graph.initializers.append(_decode_tensor(w.decode_message(gval)))
                elif gnum == 11:
                    graph.inputs.append(_decode_value_info(w.decode_message(gval)))
                elif gnum == 12:
                    graph.outputs.append(_decode_value_info(w.decode_message(gval)))
                elif gnum == 13:
                    graph.value_infos.append(_decode_value_info(w.decode_message(gval)))
        elif num == 8:
            domain = ""
            version = 0
            for onum, _, oval in w.decode_message(val):
                if onum == 1:
                    domain = oval.decode("utf-8")
                elif onum == 2:
                    version = int(oval)
            model.opset_imports.append((domain, version))
    model.graph = graph
    return model


def encode_model(model: Model) -> bytes:
    graph = bytearray()
    for node in model.graph.nodes:
        graph += w.encode_field(1, w.WIRE_LENGTH_DELIMITED, _encode_node(node))
    if model.graph.name:
        graph += w.encode_field(2, w.WIRE_LENGTH_DELIMITED, model.graph.name.encode("utf-8"))
    for tensor in model.graph.initializers:
        graph += w.encode_field(5, w.WIRE_LENGTH_DELIMITED, _encode_tensor(tensor))
    for info in model.graph.inputs:
        graph += w.encode_field(11, w.WIRE_LENGTH_DELIMITED, _encode_value_info(info))
    for info in model.graph.outputs:
        graph += w.encode_field(12, w.WIRE_LENGTH_DELIMITED, _encode_value_info(info))
    for info in model.graph.value_infos:
        graph += w.encode_field(13, w.WIRE_LENGTH_DELIMITED, _encode_value_info(info))

    out = bytearray()
    out += w.encode_field(1, w.WIRE_VARINT, model.ir_version)
    producer = model.producer_name or "onnx_weight_quant"
    out += w.encode_field(2, w.WIRE_LENGTH_DELIMITED, producer.encode("utf-8"))
    out += w.encode_field(7, w.WIRE_LENGTH_DELIMITED, bytes(graph))
    for domain, version in model.opset_imports:
        opset = bytearray()
        if domain:
            opset += w.encode_field(1, w.WIRE_LENGTH_DELIMITED, domain.encode("utf-8"))
        opset += w.encode_field(2, w.WIRE_VARINT, version)
        out += w.encode_field(8, w.WIRE_LENGTH_DELIMITED, bytes(opset))
    return bytes(out)

