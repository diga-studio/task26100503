"""Minimal protobuf wire-format primitives (varint / fixed / length-delimited)."""

from __future__ import annotations

from typing import List, Tuple

WIRE_VARINT = 0
WIRE_FIXED64 = 1
WIRE_LENGTH_DELIMITED = 2
WIRE_FIXED32 = 5

# A decoded field: (field_number, wire_type, value).
# value is int for WIRE_VARINT, bytes for WIRE_LENGTH_DELIMITED / fixed types.
Field = Tuple[int, int, object]


def encode_varint(value: int) -> bytes:
    value &= 0xFFFFFFFFFFFFFFFF
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def decode_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift >= 70:
            raise ValueError("varint too long")


def encode_tag(field_number: int, wire_type: int) -> bytes:
    return encode_varint((field_number << 3) | wire_type)


def decode_message(buf: bytes) -> List[Field]:
    fields: List[Field] = []
    pos = 0
    end = len(buf)
    while pos < end:
        tag, pos = decode_varint(buf, pos)
        field_number = tag >> 3
        wire_type = tag & 7
        if wire_type == WIRE_VARINT:
            value, pos = decode_varint(buf, pos)
        elif wire_type == WIRE_FIXED64:
            value = buf[pos:pos + 8]
            pos += 8
        elif wire_type == WIRE_LENGTH_DELIMITED:
            length, pos = decode_varint(buf, pos)
            value = buf[pos:pos + length]
            pos += length
        elif wire_type == WIRE_FIXED32:
            value = buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wire_type}")
        fields.append((field_number, wire_type, value))
    return fields


def encode_field(field_number: int, wire_type: int, payload: object) -> bytes:
    tag = encode_tag(field_number, wire_type)
    if wire_type == WIRE_VARINT:
        return tag + encode_varint(int(payload))
    if wire_type == WIRE_LENGTH_DELIMITED:
        data = bytes(payload)
        return tag + encode_varint(len(data)) + data
    if wire_type in (WIRE_FIXED32, WIRE_FIXED64):
        return tag + bytes(payload)
    raise ValueError(f"unsupported wire type {wire_type}")


def pack_varints(values: List[int]) -> bytes:
    return b"".join(encode_varint(v) for v in values)


def unpack_varints(buf: bytes) -> List[int]:
    values: List[int] = []
    pos = 0
    while pos < len(buf):
        value, pos = decode_varint(buf, pos)
        values.append(value)
    return values
