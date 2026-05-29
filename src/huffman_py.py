"""
Pure-Python Huffman coding fallback.

The original repo expects `zgls_utils.huffman.from_frequencies`, which is backed by
native code. This module provides a compatible `huffman.from_frequencies` API so the
project can run without compiling extensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Dict, Optional, TypeVar, Union

T = TypeVar("T")


@dataclass(order=True)
class _Node:
    weight: float
    idx: int
    sym: Optional[int] = field(default=None, compare=False)
    left: Optional["_Node"] = field(default=None, compare=False)
    right: Optional["_Node"] = field(default=None, compare=False)


def _build_tree(weights: Dict[int, float]) -> _Node:
    heap: list[_Node] = []
    for sym, w in weights.items():
        heappush(heap, _Node(float(w), idx=sym, sym=sym))

    if not heap:
        raise ValueError("Empty frequency table.")
    if len(heap) == 1:
        # Single-symbol edge case: give it a 1-bit code.
        only = heappop(heap)
        return _Node(only.weight, idx=only.idx + 1, sym=None, left=only, right=None)

    next_idx = len(heap)
    while len(heap) > 1:
        # Match the original C++ implementation: pop the two smallest nodes,
        # but place the larger one on the left so that code "0" follows the
        # higher-probability branch when `larger_as_zero=True`.
        right = heappop(heap)
        left = heappop(heap)
        parent = _Node(left.weight + right.weight, idx=next_idx, sym=None, left=left, right=right)
        next_idx += 1
        heappush(heap, parent)
    return heap[0]


def _walk_codes(node: _Node, prefix: str, out: Dict[int, str], larger_as_zero: bool) -> None:
    if node.sym is not None:
        out[node.sym] = prefix or "0"
        return
    left_bit, right_bit = ("0", "1") if larger_as_zero else ("1", "0")
    if node.left is not None:
        _walk_codes(node.left, prefix + left_bit, out, larger_as_zero)
    if node.right is not None:
        _walk_codes(node.right, prefix + right_bit, out, larger_as_zero)


class huffman:
    @staticmethod
    def from_frequencies(
        freqs: Dict[T, Union[int, float]], larger_as_zero: bool = True
    ) -> Dict[T, str]:
        """
        Build a Huffman code table.

        Args:
            freqs: mapping from symbol -> weight
            larger_as_zero: kept for API compatibility; not used in this fallback.
        """
        # Stable symbol ordering: map user keys to integer symbols.
        keys: list[T] = list(freqs.keys())
        idx_freqs: Dict[int, float] = {i: float(freqs[k]) for i, k in enumerate(keys)}
        tree = _build_tree(idx_freqs)
        idx2code: Dict[int, str] = {}
        _walk_codes(tree, "", idx2code, larger_as_zero)
        return {keys[i]: idx2code[i] for i in range(len(keys))}
