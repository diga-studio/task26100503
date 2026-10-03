"""Graph analysis and rewrite: quantize MatMul/Gemm right operands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ._codec import array_to_tensor, tensor_to_array
from ._model import ATTR_INT, ATTR_INTS, TENSOR_FLOAT, Attribute, Graph, Model, Node
from ._quantize import quantize_per_column

_ALIAS_OPS = ("Identity", "Transpose")
_TARGET_OPS = ("MatMul", "Gemm")


@dataclass
class QuantizedUse:
    """One rewritten use point of a weight."""

    consumer_node: str  # name or "node[i]" fallback of the MatMul/Gemm
    op_type: str
    source_initializer: str
    transposed: bool  # effective matrix is the transpose of the source
    q_initializer: str
    scale_initializer: str


@dataclass
class TransformReport:
    quantized_uses: List[QuantizedUse] = field(default_factory=list)
    removed_initializers: List[str] = field(default_factory=list)
    removed_nodes: List[str] = field(default_factory=list)


def transform_model(model: Model) -> TransformReport:
    """Rewrite the graph in place; return a report of what changed."""
    graph = model.graph
    report = TransformReport()
    initializers = {t.name: t for t in graph.initializers}
    graph_input_names = {info.name for info in graph.inputs}
    producers: Dict[str, int] = {}
    for idx, node in enumerate(graph.nodes):
        for out in node.outputs:
            producers[out] = idx

    used_names: Set[str] = set()
    for node in graph.nodes:
        used_names.update(node.inputs)
        used_names.update(node.outputs)
    for info in graph.inputs + graph.outputs + graph.value_infos:
        used_names.add(info.name)
    used_names.update(initializers)

    counter = [0]

    def fresh(base: str) -> str:
        while True:
            counter[0] += 1
            name = f"onnx_weight_quant__{base}_{counter[0]}"
            if name not in used_names:
                used_names.add(name)
                return name

    chain_node_candidates: Set[int] = set()
    quantized_sources: Set[str] = set()

    for idx, node in enumerate(graph.nodes):
        if node.op_type not in _TARGET_OPS or len(node.inputs) < 2:
            continue
        trace = _trace_chain(graph, producers, node.inputs[1])
        if trace is None:
            continue
        source_name, transposed, chain_indices = trace
        tensor = initializers.get(source_name)
        if tensor is None or source_name in graph_input_names:
            continue
        if tensor.dtype != TENSOR_FLOAT or len(tensor.dims) != 2 or 0 in tensor.dims:
            continue
        weights = tensor_to_array(tensor)
        if not np.isfinite(weights).all():
            continue

        effective = weights.T if transposed else weights
        gemm_trans_b = False
        if node.op_type == "Gemm":
            attr = node.attributes.get("transB")
            gemm_trans_b = bool(attr.value) if attr is not None else False
            if gemm_trans_b:
                effective = effective.T

        q, scales = quantize_per_column(effective)
        q_name = fresh(f"{source_name}__q")
        s_name = fresh(f"{source_name}__s")
        graph.initializers.append(array_to_tensor(q_name, q))
        graph.initializers.append(array_to_tensor(s_name, scales))

        dq_out = fresh(f"{source_name}__deq")
        dq_node = Node(
            op_type="DequantizeLinear",
            inputs=[q_name, s_name],
            outputs=[dq_out],
            name=fresh("dequant"),
            attributes={"axis": Attribute("axis", ATTR_INT, 1)},
        )
        graph.nodes.append(dq_node)

        feed = dq_out
        if node.op_type == "Gemm" and gemm_trans_b:
            trans_out = fresh(f"{source_name}__deq_t")
            graph.nodes.append(
                Node(
                    op_type="Transpose",
                    inputs=[dq_out],
                    outputs=[trans_out],
                    name=fresh("dequant_transpose"),
                    attributes={"perm": Attribute("perm", ATTR_INTS, [1, 0])},
                )
            )
            feed = trans_out
        node.inputs[1] = feed

        chain_node_candidates.update(chain_indices)
        quantized_sources.add(source_name)
        report.quantized_uses.append(
            QuantizedUse(
                consumer_node=node.name or f"node[{idx}]",
                op_type=node.op_type,
                source_initializer=source_name,
                transposed=transposed != gemm_trans_b,
                q_initializer=q_name,
                scale_initializer=s_name,
            )
        )

    _collect_garbage(graph, report, quantized_sources, chain_node_candidates)
    return report


def _trace_chain(
    graph: Graph, producers: Dict[str, int], tensor_name: str
) -> Optional[Tuple[str, bool, List[int]]]:
    """Follow an Identity/Transpose chain back to its source tensor.

    Returns (source_name, transposed, chain_node_indices) or None if the
    chain does not end at a plain tensor (initializer or graph input).
    """
    transposed = False
    chain_indices: List[int] = []
    current = tensor_name
    seen: Set[str] = set()
    while current in producers and current not in seen:
        seen.add(current)
        idx = producers[current]
        node = graph.nodes[idx]
        if node.op_type not in _ALIAS_OPS or len(node.inputs) != 1:
            break
        chain_indices.append(idx)
        if node.op_type == "Transpose" and _transpose_flips(node):
            transposed = not transposed
        current = node.inputs[0]
    if current in producers:
        return None
    return current, transposed, chain_indices


def _transpose_flips(node: Node) -> bool:
    attr = node.attributes.get("perm")
    if attr is None:
        return True  # default perm reverses dims; for 2-D that is a transpose
    return list(attr.value) == [1, 0]


def _collect_garbage(
    graph: Graph,
    report: TransformReport,
    quantized_sources: Set[str],
    chain_node_candidates: Set[int],
) -> None:
    graph_output_names = {info.name for info in graph.outputs}

    def consumed_names() -> Set[str]:
        names = set(graph_output_names)
        for node in graph.nodes:
            names.update(node.inputs)
        return names

    while True:
        consumed = consumed_names()
        dead = [
            idx
            for idx in sorted(chain_node_candidates)
            if idx < len(graph.nodes)
            and graph.nodes[idx].op_type in _ALIAS_OPS
            and all(out not in consumed for out in graph.nodes[idx].outputs)
        ]
        if not dead:
            break
        dead_set = set(dead)
        report.removed_nodes.extend(
            graph.nodes[i].name or f"node[{i}]" for i in dead
        )
        graph.nodes = [n for i, n in enumerate(graph.nodes) if i not in dead_set]
        chain_node_candidates = {
            _remap_index(i, dead) for i in chain_node_candidates - dead_set
        }

    consumed = consumed_names()
    kept_initializers = []
    for tensor in graph.initializers:
        if tensor.name in quantized_sources and tensor.name not in consumed:
            report.removed_initializers.append(tensor.name)
        else:
            kept_initializers.append(tensor)
    graph.initializers = kept_initializers


def _remap_index(idx: int, removed: List[int]) -> int:
    shift = sum(1 for r in removed if r < idx)
    return idx - shift
