"""Graph analysis and rewrite: quantize MatMul/Gemm right operands."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from onnxquant import checker, proto
from onnxquant.proto import Model, Tensor
from onnxquant.quantize import quantize_matrix


@dataclass
class QuantizeReport:
    quantized_use_points: list[str] = field(default_factory=list)
    removed_initializers: list[str] = field(default_factory=list)
    removed_nodes: list[str] = field(default_factory=list)


def _tensor_to_array(tensor: Tensor) -> np.ndarray:
    if tensor.raw_data is not None:
        dtype = {proto.FLOAT: np.float32, proto.INT8: np.int8}[tensor.data_type]
        return np.frombuffer(tensor.raw_data, dtype=dtype).reshape(tensor.dims)
    if tensor.float_data is not None:
        return np.array(tensor.float_data, dtype=np.float32).reshape(tensor.dims)
    raise ValueError(f"initializer {tensor.name!r} carries no data")


def _array_to_tensor(name: str, array: np.ndarray) -> Tensor:
    dtype = {np.dtype(np.float32): proto.FLOAT, np.dtype(np.int8): proto.INT8}[array.dtype]
    return Tensor(
        name=name,
        dims=list(array.shape),
        data_type=dtype,
        raw_data=array.tobytes(),
    )


def _trace_alias_chain(
    tensor_name: str, producers: dict[str, proto.Node]
) -> tuple[str, list[proto.Node]]:
    """Follow an Identity/Transpose chain back to its source tensor name."""
    chain: list[proto.Node] = []
    current = tensor_name
    while current in producers and producers[current].op_type in ("Identity", "Transpose"):
        node = producers[current]
        chain.append(node)
        current = node.inputs[0]
    return current, chain


def _apply_chain(array: np.ndarray, chain: list[proto.Node]) -> np.ndarray:
    """Apply the alias chain (source -> use point order) to a weight array."""
    for node in reversed(chain):
        if node.op_type == "Transpose":
            perm = node.get_attribute("perm")
            if perm is None:
                perm = list(reversed(range(array.ndim)))
            array = np.transpose(array, axes=list(perm))
        # Identity: no-op
    return array


def _unique_name(prefix: str, taken: set[str]) -> str:
    candidate = prefix
    index = 0
    while candidate in taken:
        index += 1
        candidate = f"{prefix}_{index}"
    taken.add(candidate)
    return candidate


def quantize_model(model: Model) -> tuple[Model, QuantizeReport]:
    checker.check_model(model)
    graph = model.graph
    report = QuantizeReport()

    initializers = {t.name: t for t in graph.initializers}
    producers: dict[str, proto.Node] = {}
    for node in graph.nodes:
        for out in node.outputs:
            producers[out] = node

    # Initializers also listed as graph inputs are overridable defaults:
    # none of their use points may be quantized.
    overridable = {vi.name for vi in graph.inputs if vi.name in initializers}

    taken = {t.name for t in graph.initializers}
    for vi in graph.inputs + graph.outputs + graph.value_info:
        taken.add(vi.name)
    for node in graph.nodes:
        taken.update(node.inputs, node.outputs)

    # Cache: (use-point tensor name, channel_axis) -> dequant output name.
    dequant_cache: dict[tuple[str, int], str] = {}
    # Dequant nodes to insert immediately before each consumer node,
    # preserving topological order.
    pending: dict[int, list[proto.Node]] = {}

    for node in graph.nodes:
        if node.op_type not in ("MatMul", "Gemm") or len(node.inputs) < 2:
            continue
        use_name = node.inputs[1]
        source_name, chain = _trace_alias_chain(use_name, producers)
        if source_name in overridable or source_name not in initializers:
            continue
        source = initializers[source_name]
        if source.data_type != proto.FLOAT or len(source.dims) != 2:
            continue
        weight = _tensor_to_array(source)
        if weight.size == 0 or not np.all(np.isfinite(weight)):
            continue
        effective = _apply_chain(weight, chain)
        if effective.ndim != 2:
            continue
        if node.op_type == "Gemm":
            trans_b = node.get_attribute("transB", 0) or 0
            channel_axis = 0 if trans_b else 1
        else:
            channel_axis = 1

        cache_key = (use_name, channel_axis)
        if cache_key in dequant_cache:
            node.inputs[1] = dequant_cache[cache_key]
            report.quantized_use_points.append(node.name or node.op_type)
            continue

        qvalues, scales = quantize_matrix(effective, channel_axis)
        base = use_name.replace(".", "_").replace("/", "_")
        q_name = _unique_name(f"{base}__q", taken)
        s_name = _unique_name(f"{base}__scale", taken)
        z_name = _unique_name(f"{base}__zp", taken)
        dq_name = _unique_name(f"{base}__dequant", taken)

        graph.initializers.append(_array_to_tensor(q_name, qvalues))
        graph.initializers.append(_array_to_tensor(s_name, scales.astype(np.float32)))
        graph.initializers.append(
            _array_to_tensor(z_name, np.zeros(scales.shape, dtype=np.int8))
        )
        pending.setdefault(id(node), []).append(
            proto.Node(
                op_type="DequantizeLinear",
                name=f"{dq_name}__node",
                inputs=[q_name, s_name, z_name],
                outputs=[dq_name],
                attributes=[
                    proto.Attribute(name="axis", type=proto.ATTR_INT, i=channel_axis)
                ],
            )
        )
        dequant_cache[cache_key] = dq_name
        node.inputs[1] = dq_name
        report.quantized_use_points.append(node.name or node.op_type)

    ordered_nodes: list[proto.Node] = []
    for node in graph.nodes:
        ordered_nodes.extend(pending.get(id(node), []))
        ordered_nodes.append(node)
    graph.nodes = ordered_nodes

    # Dead-code elimination: drop alias nodes and initializers that no longer
    # feed any node input or graph output (graph outputs count as uses).
    changed = True
    while changed:
        changed = False
        used = {vi.name for vi in graph.outputs}
        for node in graph.nodes:
            used.update(name for name in node.inputs if name)
        kept_nodes = []
        for node in graph.nodes:
            if all(out in used for out in node.outputs):
                kept_nodes.append(node)
            else:
                changed = True
                report.removed_nodes.append(node.name or node.op_type)
        graph.nodes = kept_nodes

    used = {vi.name for vi in graph.outputs}
    for node in graph.nodes:
        used.update(name for name in node.inputs if name)
    kept_initializers = []
    for tensor in graph.initializers:
        if tensor.name in used:
            kept_initializers.append(tensor)
        else:
            report.removed_initializers.append(tensor.name)
    graph.initializers = kept_initializers

    checker.check_model(model, quantized=True)
    return model, report


def quantize_file(input_path: str, output_path: str) -> QuantizeReport:
    model = proto.load_model(input_path)
    model, report = quantize_model(model)
    proto.save_model(model, output_path)
    return report
