#!/usr/bin/env python3
r"""
Offline GGUF surgery for a standalone qwen4exp MTP sidecar vs an expert-pruned trunk.

Pipeline (see each subcommand for details):
    p0_probe   read the sidecar schema, prove which tensors carry the expert axis
    p1_mask    recover the surviving expert indices (S) from an authoritative manifest,
               or fall back to slab-signature matching against an unpruned reference
    p2_plan    plan the byte surgery and assert post-slice invariants (writes nothing)
    p3_apply   gather expert slabs as raw bytes (no dequant/requant) and stream a new GGUF
    p4_verify  re-read the output, assert shapes/counts, prove bit-exact gathers,
               and print a llama-cli speculative-decoding benchmark playbook

Only gguf (gguf-py), numpy, struct and hashlib are used. struct is needed to fingerprint
file headers without pulling tensor data; hashlib proves the byte gathers survived the rewrite.

Ground truth is the data: every asserted constant (arch, block counts, expert counts, tensor
keys, tensor types, byte strides) is read from the files, never assumed from a spec.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Sequence


import logging

import numpy as np


# Necessary to load the local gguf package (same idiom as scripts/gguf_new_metadata.py)
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gguf

from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType, Keys, GGUFValueType  # noqa: F401

logger = None  # set in main()


def kv_of(reader: gguf.GGUFReader) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, field in reader.fields.items():
        if name.startswith("GGUF."):
            continue
        out[name] = field.contents()
    return out


def kv_get(kv: dict[str, Any], key: str, default: Any = None) -> Any:
    return kv[key] if key in kv else default


def arch_key(arch: str, suffix: str) -> str:
    return f"{arch}.{suffix}"


# ---------------------------------------------------------------------------
# expert axis / byte-slab math
# ---------------------------------------------------------------------------
# GGUF stores ne[0] as the contiguous axis. create_tensor() in src/models/qwen4exp.cpp
# builds { n_embd, n_expert } for the router and { n_ff_exp, n_embd, n_expert } for the
# expert weights, so the expert axis is always the LAST gguf dim: it is the outermost
# axis in file order, and one expert slab is one contiguous run of bytes.


def slab_math(ne: Sequence[int], qt: GGMLQuantizationType) -> tuple[int, int, int, int]:
    block_size, type_size = GGML_QUANT_SIZES[qt]
    ne = [int(d) for d in ne]
    slab_elems = 1
    for d in ne[:-1]:
        slab_elems *= d
    if slab_elems % block_size:
        raise ValueError(f"slab of {slab_elems} elems is not a whole number of {qt.name} blocks "
                         f"(block_size={block_size}); raw byte gather is impossible")
    n_slabs = ne[-1]
    slab_bytes = slab_elems * type_size // block_size
    total = slab_bytes * n_slabs
    block, ts = GGML_QUANT_SIZES[qt]
    n_elems = 1
    for d in ne:
        n_elems *= d
    if total != n_elems * ts // block:
        raise ValueError(f"slab math disagrees with element math for {qt.name}")
    return slab_elems, slab_bytes, n_slabs, total


def expert_tensors(reader: gguf.GGUFReader, n_expert: int,
                   layers: set[int] | None = None) -> dict[str, dict[str, Any]]:
    """Key-name based: qwen4exp.cpp creates the expert axis only on ffn_gate_inp (router)
    and ffn_gate/up/down_exps. Dimension equality alone is unsafe (attn_k/attn_v/indexer
    projections in this arch are also 512 wide)."""
    out: dict[str, dict[str, Any]] = {}
    for t in reader.tensors:
        if not any(t.name.endswith("." + s) for s in (ROUTER_SUFFIX,) + EXPERT_TENSOR_SUFFIXES):
            continue
        if layers is not None:
            if not t.name.startswith("blk.") or int(t.name.split(".")[1]) not in layers:
                continue
        ne = [int(d) for d in t.shape]
        if ne[-1] != n_expert:
            raise ValueError(f"{t.name}: expected last dim {n_expert}, got {ne}")
        slab_elems, slab_bytes, n_slabs, total = slab_math(ne, t.tensor_type)
        if total != t.n_bytes:
            raise ValueError(f"{t.name}: byte math {slab_bytes} x {n_slabs} != {t.n_bytes}")
        out[t.name] = {"ne": ne, "ne_numpy": list(reversed(ne)), "type": t.tensor_type.name,
                       "qt": t.tensor_type, "slab_elems": slab_elems, "slab_bytes": slab_bytes,
                       "n_slabs": n_slabs, "n_bytes": t.n_bytes, "data_offset": t.data_offset}
    return out



def nextn_layers(reader: gguf.GGUFReader) -> set[int]:
    out = set()
    for t in reader.tensors:
        if ".nextn." in t.name:
            out.add(int(t.name.split(".")[1]))
    return out


def spec_layers_without_nextn(reader: gguf.GGUFReader, nextn: set[int]) -> set[int]:
    seen = set()
    for t in reader.tensors:
        if t.name.startswith("blk."):
            seen.add(int(t.name.split(".")[1]))
    return seen - nextn


def header_sha256(path: Path, data_offset: int) -> str:
    n = max(64, int(data_offset))
    with open(path, "rb") as f:
        h = hashlib.sha256()
        left = n
        while left > 0:
            b = f.read(min(1 << 20, left))
            if not b:
                break
            h.update(b)
            left -= len(b)
        return h.hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    r = gguf.GGUFReader(path)
    return {
        "path": str(path),
        "bytes": int(Path(path).stat().st_size),
        "header_sha256": header_sha256(Path(path), r.data_offset),
        "alignment": int(r.alignment),
        "n_tensors": len(r.tensors),
        "data_offset": int(r.data_offset),
    }


def pool_readers(paths: Sequence[Path]) -> tuple[gguf.GGUFReader, dict[str, ReaderTensor], dict[str, Path], dict[str, Any]]:
    """Merge sharded GGUF (e.g. ...-00001-of-00002.gguf) into one tensor name -> tensor map."""
    primary = None
    tensors: dict[str, ReaderTensor] = {}
    owner: dict[str, Path] = {}
    for p in paths:
        r = gguf.GGUFReader(p)
        if primary is None or len(r.fields) > len(primary.fields):
            primary = r
        for t in r.tensors:
            if t.name in tensors:
                raise ValueError(f"tensor {t.name!r} is duplicated across shards")
            tensors[t.name] = t
            owner[t.name] = Path(p)
    kv = kv_of(primary)
    return primary, tensors, owner, kv



def slab_signatures(tensors: dict[str, ReaderTensor], owner: dict[str, Path], key: str,
                    nb_blocks: int) -> np.ndarray:
    t = tensors[key]
    ne = [int(d) for d in t.shape]
    _, slab_bytes, n_slabs, _ = slab_math(ne, t.tensor_type)
    win_bytes, tail, _ = slab_windows(ne, t.tensor_type, nb_blocks)
    block_size, type_size = GGML_QUANT_SIZES[t.tensor_type]
    qcls = getattr(gguf.quants, t.tensor_type.name)
    mm = np.memmap(owner[key], dtype=np.uint8, mode="r")
    out = np.empty((n_slabs, 2 * nb_blocks * block_size), dtype=np.float64)
    for i in range(n_slabs):
        base = t.data_offset + i * slab_bytes
        parts = []
        for off in (0, tail):
            raw = np.asarray(mm[base + off: base + off + win_bytes])
            parts.append(qcls.dequantize(raw.reshape(nb_blocks, type_size)).astype(np.float64))
        out[i] = np.concatenate(parts)
    return out



# ---------------------------------------------------------------------------
# p0: schema probe
# ---------------------------------------------------------------------------

EXPERT_TENSOR_SUFFIXES = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
ROUTER_SUFFIX = "ffn_gate_inp.weight"


def describe_sidecar(path: Path) -> dict[str, Any]:
    r = gguf.GGUFReader(path)
    kv = kv_of(r)
    arch = kv_get(kv, Keys.General.ARCHITECTURE)
    rep: dict[str, Any] = {"path": str(path), "arch": arch, "kv": kv, "fail": []}
    if arch != "qwen4exp":
        rep["fail"].append(f"general.architecture = {arch!r}, expected 'qwen4exp'")
        return rep

    n_block = kv_get(kv, arch_key(arch, "block_count"))
    n_nextn = kv_get(kv, arch_key(arch, "nextn_predict_layers"))
    n_expert = kv_get(kv, arch_key(arch, "expert_count"))
    n_used = kv_get(kv, arch_key(arch, "expert_used_count"))
    rep.update(block_count=n_block, nextn_predict_layers=n_nextn, expert_count=n_expert,
               expert_used_count=n_used)

    nextn = nextn_layers(r)
    rep["nextn_layers"] = sorted(nextn)
    if n_nextn is not None and len(nextn) != n_nextn:
        rep["fail"].append(f"blk.N.nextn.* tensors live in {sorted(nextn)}, metadata says {n_nextn} layer(s)")
    if not nextn:
        rep["fail"].append("no blk.N.nextn.* tensors: this file has no speculative block")

    if n_block is not None and len(nextn) == 1:
        spec = next(iter(nextn))
        rep["spec_layer"] = spec
        if spec != n_block - 1:
            rep["fail"].append(f"speculative block is {spec}, block_count is {n_block} (expected block_count - 1)")
        for key in ("attention.compress_ratios", "attention.recurrent_layers"):
            arr = kv_get(kv, arch_key(arch, key))
            if arr is not None and len(arr) != n_block:
                rep["fail"].append(f"{arch}.{key} has {len(arr)} entries, block_count is {n_block}")

    if n_expert is not None:
        all_carriers = expert_tensors(r, n_expert)
        layers = {int(n.split(".")[1]) for n in all_carriers}
        if nextn and not layers <= nextn:
            rep["fail"].append(f"expert-bearing tensors are in layers {sorted(layers)}, "
                               f"nextn layers are {sorted(nextn)}: this file is not a standalone sidecar")
        rep["expert_tensors"] = {k: v for k, v in all_carriers.items()
                                 if int(k.split(".")[1]) in (nextn or layers)}
        rep["nextn_tensors"] = [t.name for t in r.tensors if ".nextn." in t.name]
        rep["untouched"] = [t.name for t in r.tensors if t.name not in rep["expert_tensors"]]
        if n_used is not None and n_used > n_expert:
            rep["fail"].append(f"expert_used_count {n_used} > expert_count {n_expert}")
        for name, info in sorted(rep["expert_tensors"].items()):
            if info["n_bytes"] != info["slab_bytes"] * info["n_slabs"]:
                rep["fail"].append(f"{name}: n_bytes {info['n_bytes']} != {info['slab_bytes']} x {info['n_slabs']}")
    return rep


def header_probe(path: Path, nbytes: int = 96 << 20) -> dict[str, Any]:
    """Cheap range read: magic, version, tensor/KV counts and the header size, via struct only."""
    with open(path, "rb") as f:
        head = f.read(nbytes)
    if head[:4] != b"GGUF":
        raise SystemExit(f"{path}: not a GGUF file (magic {head[:4]!r})")
    version, n_tensors, n_kv = struct.unpack_from("<IQQ", head, 4)
    off = 4 + 4 + 8 + 8
    fmt = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 4, 11: 4, 12: 4}

    def skip_str(b: bytes, o: int) -> int:
        n, = struct.unpack_from("<Q", b, o)
        if o + 8 + n > len(b):
            raise SystemExit(f"{path}: KV section is larger than the {len(b)} byte range read")
        return o + 8 + n

    def skip_val(b: bytes, o: int, t: int) -> int:
        if t == 8:
            return skip_str(b, o)
        if t == 9:
            et, = struct.unpack_from("<I", b, o)
            o += 4
            c, = struct.unpack_from("<Q", b, o)
            o += 8
            for _ in range(c):
                o = skip_val(b, o, et)
            return o
        return o + fmt[t]

    for _ in range(n_kv):
        off = skip_str(head, off)
        t, = struct.unpack_from("<I", head, off)
        off += 4
        off = skip_val(head, off, t)
    return {"magic": head[:4].decode(), "version": version, "n_tensors": n_tensors,
            "n_kv": n_kv, "kv_end": off}

    def skip_val(b: bytes, o: int, t: int) -> int:
        if t == 8:
            return skip_str(b, o)
        if t == 9:
            et, = struct.unpack_from("<I", b, o)
            o += 4
            c, = struct.unpack_from("<Q", b, o)
            o += 8
            for _ in range(c):
                o = skip_val(b, o, et)
            return o
        return o + fmt[t]

    for _ in range(n_kv):
        off = skip_str(head, off)
        t, = struct.unpack_from("<I", head, off)
        off += 4
        off = skip_val(head, off, t)
    return {"magic": head[:4].decode(), "version": version, "n_tensors": n_tensors,
            "n_kv": n_kv, "kv_end": off}


def cmd_p0_probe(mtp: Path) -> int:
    cheap = header_probe(mtp)
    print(f"== p0_probe {mtp}")
    print(f"header probe: magic GGUF version={cheap['version']} n_tensors={cheap['n_tensors']} "
          f"n_kv={cheap['n_kv']} kv_bytes_end={cheap['kv_end']}")
    rep = describe_sidecar(mtp)
    print(f"arch={rep['arch']} block_count={rep.get('block_count')} "
          f"nextn_predict_layers={rep.get('nextn_predict_layers')} "
          f"expert_count={rep.get('expert_count')} expert_used_count={rep.get('expert_used_count')}")
    fp = file_fingerprint(mtp)
    print(f"file bytes={fp['bytes']} alignment={fp['alignment']} n_tensors={fp['n_tensors']} "
          f"data_offset={fp['data_offset']}")
    r = gguf.GGUFReader(mtp)
    types: dict[str, int] = {}
    for t in r.tensors:
        types[t.tensor_type.name] = types.get(t.tensor_type.name, 0) + 1
    print("tensor types:", dict(sorted(types.items(), key=lambda x: -x[1])))
    print(f"speculative layer: {rep.get('spec_layer')} (nextn layers {rep.get('nextn_layers')})")
    exp = rep.get("expert_tensors", {})
    print(f"tensors carrying the expert axis ({rep.get('expert_count')} slabs):")
    for name, info in sorted(exp.items()):
        print(f"  {name}  gguf ne={info['ne']} (expert axis = ne[{len(info['ne']) - 1}])  "
              f"numpy shape={info['ne_numpy']}  type={info['type']}")
        print(f"    slab = {info['slab_elems']} elems = {info['slab_bytes']} bytes; "
              f"{info['n_slabs']} slabs; total {info['n_bytes']} bytes")
    print(f"nextn tensors (never sliced): {rep.get('nextn_tensors')}")
    print(f"untouched tensors: {len(rep.get('untouched', []))}")
    print(f"imatrix keys: {[k for k in rep['kv'] if 'imatrix' in k]}")
    for line in rep["fail"]:
        print("FAIL:", line)
    if rep["fail"]:
        return 1
    print("schema check: OK")
    return 0




# ---------------------------------------------------------------------------
# mask sources (priority 1: release manifest, priority 2: slab signatures)
# ---------------------------------------------------------------------------


def manifest_discovery(trunk_paths: Sequence[Path]) -> list[Path]:
    found: list[Path] = []
    for p in trunk_paths:
        for base in (p.parent, p.parent.parent, p.parent / "tensor-allocation"):
            for pat in ("*rco-allocation.txt", "*allocation*.txt"):
                for cand in Path(base).glob(pat):
                    if cand.suffix.lower() == ".txt" and cand not in found:
                        found.append(cand)
    return sorted(found)


def parse_manifest(path: Path) -> dict[str, Any]:
    """Release allocation manifest: '<blk.N.ffn_x_exps.weight>[<orig idx>]: <TYPE>'; a pruned
    expert carries an empty type, and indices are those of the unpruned model."""
    import re
    line_re = re.compile(r"^(blk\.\d+)\.ffn_(gate|up|down)_exps\.weight\[(\d+)\]:\s*(\S*)\s*$")
    per_layer: dict[int, dict[str, set[int]]] = {}
    n_indexed = 0
    n_other = 0
    for raw in open(path, encoding="utf-8", errors="replace"):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        m = line_re.match(s)
        if not m:
            n_other += 1
            continue
        n_indexed += 1
        blk, which, idx, ty = m.groups()
        layer = int(blk.split(".")[1])
        d = per_layer.setdefault(layer, {w: set() for w in ("gate", "up", "down")})
        if ty:
            d[which].add(int(idx))
    layers = {}
    notes: list[str] = []
    for layer in sorted(per_layer):
        sets = per_layer[layer]
        sizes = sorted(len(v) for v in sets.values())
        if not (sets["gate"] == sets["up"] == sets["down"]):
            notes.append(f"layer {layer}: gate/up/down kept-sets differ (sizes {sizes})")
        layers[layer] = sorted(sets["gate"])
    return {"path": str(path), "indexed_lines": n_indexed, "other_lines": n_other,
            "layers": layers, "notes": notes}


def slab_windows(ne: Sequence[int], qt: GGMLQuantizationType, nb_blocks: int) -> tuple[int, int, int]:
    block_size, type_size = GGML_QUANT_SIZES[qt]
    _, slab_bytes, _, _ = slab_math(ne, qt)
    win_bytes = nb_blocks * type_size
    if 2 * win_bytes > slab_bytes:
        raise ValueError(f"slab of {slab_bytes} bytes cannot hold two windows of {win_bytes}")
    return win_bytes, slab_bytes - win_bytes, nb_blocks * block_size





def self_router_mask(path: Path, key: str, keep: int) -> tuple[list[int], list[float]]:
    """Heuristic mask from the sidecar's own router: a column norm is the only offline proxy
    for how often an expert is routed; it is not a measured routing rate."""
    t = next(x for x in gguf.GGUFReader(path).tensors if x.name == key)
    if t.data.dtype != np.float32:
        raise SystemExit(f"{key}: router must be F32 for a column-norm mask, got {t.data.dtype}")
    ne = [int(d) for d in t.shape]
    cols = np.asarray(t.data).reshape(ne[1], ne[0])
    norms = np.linalg.norm(cols, axis=1)
    order = np.argsort(-norms, kind="stable")
    keep = min(int(keep), len(order))
    return sorted(int(i) for i in order[:keep]), [float(v) for v in norms]


def vote_mask(layer_masks: dict[int, list[int]], keep: int) -> tuple[list[int] | None, list[int]]:
    votes: dict[int, int] = {}
    for kept in layer_masks.values():
        for i in kept:
            votes[i] = votes.get(i, 0) + 1
    order = sorted(votes, key=lambda i: (-votes[i], i))
    if len(order) < keep:
        return None, sorted(set(range(max(order) + 1)) - set(order))
    cut = votes[order[keep - 1]]
    tie = [i for i, v in votes.items() if v == cut]
    if len(tie) > 1:
        return None, sorted(tie)
    return sorted(order[:keep]), []


def resolve_global(layer_masks: dict[int, list[int]], policy: str, sidecar_experts: int | None,
                   mask_arg: list[int] | None, layer: int | None, keep: int | None) -> dict[str, Any]:
    sets = {k: set(v) for k, v in layer_masks.items()}
    inter = sorted(set.intersection(*sets.values())) if sets else []
    union = sorted(set.union(*sets.values())) if sets else []
    out: dict[str, Any] = {"intersection": inter, "union": union,
                           "force_kept": inter,
                           "force_killed": sorted(set(range(union[-1] + 1)) - set(union)) if union else [],
                           "divergent": sorted(set(union) - set(inter))}
    if not sets:
        out["status"] = "no-layer-masks"
        return out
    if all(s == sets[min(sets)] for s in sets.values()):
        out["status"] = "agree"
        out["global_S"] = sorted(sets[min(sets)])
        return out
    out["status"] = "diverge"
    if policy == "layer":
        if layer is None or layer not in layer_masks:
            out["status"] = "error"
            return out
        out["global_S"] = layer_masks[layer]
        out["status"] = "validated"
        return out
    if policy == "vote":
        k = keep or len(union) // 2
        s, tie = vote_mask(layer_masks, k)
        if s is None:
            out["ambiguous"] = tie
            out["status"] = "ambiguous"
            return out
        out["global_S"] = s
        out["status"] = "validated"
    return out


def cmd_p1_emit(out_json: Path, trunk_paths: Sequence[Path], source: dict[str, Any] | None,
                layer_masks: dict[int, list[int]], global_S: list[int] | None,
                extra: dict[str, Any], notes: Sequence[str], sidecar_experts: int | None) -> int:
    status = "validated" if global_S is not None else "unresolved"
    payload = {
        "tool": "mtp_prune_surgery p1_mask",
        "schema": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": status,
        "source": source,
        "trunk_files": [file_fingerprint(p) for p in trunk_paths],
        "expert_axis_convention": "last gguf dim (ne[-1]); one slab is contiguous bytes",
        "per_layer_masks": {str(k): v for k, v in layer_masks.items()},
        "global_S": global_S,
        "intersection": extra.get("intersection"),
        "union": extra.get("union"),
        "divergent": extra.get("divergent"),
        "force_kept": extra.get("force_kept"),
        "force_killed": extra.get("force_killed"),
        "ambiguous": extra.get("ambiguous") or (extra.get("divergent") if global_S is None else []),
        "sidecar_expert_count": sidecar_experts,
        "notes": list(notes),
    }
    out_json.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {out_json}")
    return 0 if global_S is not None else 2



def cmd_p1_mask(trunk_paths: Sequence[Path], out_json: Path, args, mask_arg: list[int] | None) -> int:
    print(f"== p1_mask trunk files: {[str(p) for p in trunk_paths]}")
    _primary, trunk_tensors, _owner, trunk_kv = pool_readers(trunk_paths)
    arch = kv_get(trunk_kv, Keys.General.ARCHITECTURE)
    if arch is None:
        print("FAIL: trunk has no general.architecture")
        return 1
    trunk_experts = kv_get(trunk_kv, arch_key(arch, "expert_count"))
    trunk_blocks = kv_get(trunk_kv, arch_key(arch, "block_count"))
    n_carriers = sum(1 for n in trunk_tensors if any(n.endswith(s) for s in EXPERT_TENSOR_SUFFIXES))
    print(f"trunk: arch={arch} block_count={trunk_blocks} expert_count={trunk_experts} "
          f"expert-bearing tensors={n_carriers}")
    for suffix in EXPERT_TENSOR_SUFFIXES:
        key = f"blk.0.{suffix}"
        if key in trunk_tensors:
            ne = [int(d) for d in trunk_tensors[key].shape]
            print(f"  carrier {key}: gguf ne={ne} type={trunk_tensors[key].tensor_type.name}")

    sidecar_experts = None
    if args.sidecar:
        rep = describe_sidecar(Path(args.sidecar))
        sidecar_experts = rep.get("expert_count")
        print(f"sidecar: expert_count={sidecar_experts} spec_layer={rep.get('spec_layer')}")

    layer_masks: dict[int, list[int]] = {}
    source: dict[str, Any] | None = None
    notes: list[str] = []

    if args.policy == "identity":
        if sidecar_experts is None:
            print("FAIL: --policy identity needs --sidecar")
            return 1
        return cmd_p1_emit(out_json, trunk_paths, {"kind": "identity"}, {},
                           list(range(sidecar_experts)), {}, notes, sidecar_experts)
    if args.policy == "explicit":
        if not mask_arg:
            print("FAIL: --policy explicit needs --mask or --mask-file")
            return 1
        return cmd_p1_emit(out_json, trunk_paths, {"kind": "explicit"}, {},
                           sorted(set(mask_arg)), {}, notes, sidecar_experts)
    if args.policy == "self-router":
        if not args.sidecar:
            print("FAIL: --policy self-router needs --sidecar")
            return 1
        rep = describe_sidecar(Path(args.sidecar))
        router = f"blk.{rep.get('spec_layer')}.{ROUTER_SUFFIX}"
        keep = args.keep or trunk_experts
        s, _norms = self_router_mask(Path(args.sidecar), router, keep)
        notes.append(f"router column-norm proxy on {router}: a heuristic, not a measured routing rate")
        return cmd_p1_emit(out_json, trunk_paths, {"kind": "self-router", "router": router},
                           {}, s, {}, notes, sidecar_experts)

    manifests = [args.manifest] if args.manifest else manifest_discovery(trunk_paths)
    if manifests:
        man = parse_manifest(Path(manifests[0]))
        layer_masks = man["layers"]
        source = {"kind": "manifest", "path": man["path"], "indexed_lines": man["indexed_lines"]}
        notes += man["notes"]
        sizes = sorted({len(v) for v in layer_masks.values()})
        print(f"manifest {man['path']}: {len(layer_masks)} layers, kept sizes {sizes}")
        if len(sizes) == 1 and trunk_experts is not None and sizes[0] != trunk_experts:
            notes.append(f"manifest kept count {sizes[0]} != trunk expert_count {trunk_experts}")
    elif args.reference:
        _r, ref_tensors, _ro, ref_kv = pool_readers(args.reference)
        print(f"reference expert_count = {kv_get(ref_kv, arch_key(arch, 'expert_count'))}")
        sample = args.sample_layers if args.sample_layers else [0, max(0, (trunk_blocks or 1) - 1)]
        layer_masks, more = reference_maps(trunk_tensors, ref_tensors, sample,
                                           args.block_window, args.margin)
        notes += more
        source = {"kind": "slab-signature", "reference": [str(p) for p in args.reference],
                  "sample_layers": sample, "block_window": args.block_window, "margin": args.margin}
    elif sidecar_experts is not None and trunk_experts == sidecar_experts:
        notes.append("trunk and sidecar expert_count match: no slicing required")
        return cmd_p1_emit(out_json, trunk_paths, {"kind": "count-match"}, {},
                           list(range(sidecar_experts)), {}, notes, sidecar_experts)
    else:
        print("FAIL: no manifest and no --reference: the surviving indices are not recoverable")
        print("  (compaction leaves no zeroed slabs to detect). Provide --manifest, --reference,")
        print("  --policy identity (keep all) or --policy explicit --mask ...")
        return 1

    res = resolve_global(layer_masks, args.policy, sidecar_experts, mask_arg, args.layer, args.keep)
    if res["status"] in ("agree", "validated"):
        print(f"global mask resolved (source status {res['status']})")
    else:
        print(f"{len(layer_masks)} per-layer masks do NOT agree")
        print(f"intersection={len(res['intersection'])} union={len(res['union'])} "
              f"divergent={len(res['divergent'])} (ambiguous set is non-empty)")
        for n in notes:
            print("note:", n)
        print("global_S is not derivable: pick --policy layer --layer N / self-router / identity / explicit")
    extra = {k: v for k, v in res.items() if k not in ("status", "global_S")}
    return cmd_p1_emit(out_json, trunk_paths, source, layer_masks, res.get("global_S"), extra, notes, sidecar_experts)



# ---------------------------------------------------------------------------
# p2: surgery plan (no writes)
# ---------------------------------------------------------------------------


def load_mask(mask_json: Path) -> dict[str, Any]:
    payload = json.loads(mask_json.read_text(encoding="utf-8"))
    if payload.get("status") != "validated" or payload.get("global_S") is None:
        raise SystemExit(f"mask {mask_json} is not validated (status={payload.get('status')}): "
                         "resolve a policy with p1_mask first")
    return payload


def plan_surgery(mtp: Path, S: Sequence[int]) -> dict[str, Any]:
    rep = describe_sidecar(mtp)
    if rep["fail"]:
        raise SystemExit("sidecar schema check failed: " + "; ".join(rep["fail"]))
    arch = rep["arch"]
    kv = rep["kv"]
    n_expert_old = rep["expert_count"]
    spec = rep["spec_layer"]
    carriers = rep["expert_tensors"]

    S = sorted(set(int(i) for i in S))
    if len(S) != len(set(S)) or any(i < 0 for i in S):
        raise SystemExit("mask must be non-negative and strictly increasing")
    if S[-1] >= n_expert_old:
        raise SystemExit(f"max(mask)={S[-1]} >= expert_count={n_expert_old}")

    used = kv_get(kv, arch_key(arch, "expert_used_count"))
    if used is not None and used > len(S):
        raise SystemExit(f"expert_used_count {used} > new expert_count {len(S)}")

    plan = {}
    for name, info in sorted(carriers.items()):
        ne_new = list(info["ne"])
        ne_new[-1] = len(S)
        _, slab_bytes, _, total = slab_math(ne_new, info["qt"])
        if slab_bytes != info["slab_bytes"]:
            raise SystemExit(f"{name}: slab stride must not change under slicing")
        # gguf-py byte helpers work in numpy order (contiguous axis last), while the reader
        # reports gguf order (contiguous axis first)
        byte_shape = gguf.quants.quant_shape_to_byte_shape(list(reversed(ne_new)), info["qt"])
        plan[name] = {"ne_old": info["ne"], "ne_new": ne_new, "type": info["type"],
                      "n_bytes_old": info["n_bytes"], "n_bytes_new": total,
                      "byte_shape": [int(d) for d in byte_shape], "slab_bytes": slab_bytes}

    untouched = [t.name for t in gguf.GGUFReader(mtp).tensors if t.name not in carriers]
    return {"rep": rep, "S": S, "plan": plan, "untouched": untouched}


def cmd_p2_plan(mtp: Path, mask_json: Path) -> int:
    payload = load_mask(mask_json)
    S = payload["global_S"]
    got = plan_surgery(mtp, S)
    rep = got["rep"]
    arch = rep["arch"]
    kv = rep["kv"]
    print(f"== p2_plan {mtp} <- {mask_json}")
    print(f"source mask: {payload['source']}")
    print(f"sidecar: expert_count={rep['expert_count']} block_count={rep['block_count']} "
          f"nextn_predict_layers={rep['nextn_predict_layers']} spec_layer={rep['spec_layer']}")
    layer_sizes = sorted({len(v) for v in payload["per_layer_masks"].values()}) if payload["per_layer_masks"] else []
    print(f"trunk per-layer kept count: {layer_sizes} (target expert_count after surgery = {len(S)})")
    print(f"mask: len={len(S)} strictly_increasing={S == sorted(S)} min={min(S)} max={max(S)} "
          f"identity={S == list(range(len(S)))}")
    if layer_sizes and len(S) != layer_sizes[0]:
        print(f"WARN: len(mask) {len(S)} != trunk per-layer kept count {layer_sizes[0]}")
    if rep["expert_count"] is not None and len(S) == rep["expert_count"]:
        print("slicing is a no-op here: the mask keeps every expert (metadata rewrite only)")
    for name, info in sorted(got["plan"].items()):
        print(f"  {name}: {info['ne_old']} -> {info['ne_new']}  {info['n_bytes_old']} -> "
              f"{info['n_bytes_new']} bytes  type={info['type']} (unchanged)")
        print(f"    slab {info['slab_bytes']} bytes x {len(S)} slabs; gguf byte shape {info['byte_shape']}")
    print(f"untouched tensors: {len(got['untouched'])} (of them {len(rep['nextn_tensors'])} are nextn.*)")
    saved = sum(v["n_bytes_old"] - v["n_bytes_new"] for v in got["plan"].values())
    print(f"projected size delta: -{saved} bytes ({saved / 1e6:.0f} MiB smaller)")
    print("no bytes written (p2_plan is dry)")
    return 0



# ---------------------------------------------------------------------------
# p3: bit-exact byte slab gather + GGUF re-write
# ---------------------------------------------------------------------------


def gather_hash(mm: np.ndarray, data_offset: int, slab_bytes: int, n_old: int, S: Sequence[int]) -> str:
    h = hashlib.sha256()
    view = mm[data_offset:data_offset + slab_bytes * n_old].reshape(n_old, slab_bytes)
    for i in S:
        h.update(np.ascontiguousarray(view[i]).tobytes())
    return h.hexdigest()


def cmd_p3_apply(mtp: Path, mask_json: Path, out_path: Path, args) -> int:
    payload = load_mask(mask_json)
    S = payload["global_S"]
    got = plan_surgery(mtp, S)
    rep = got["rep"]
    arch = rep["arch"]
    kv = rep["kv"]
    r = gguf.GGUFReader(mtp)
    plan = got["plan"]
    new_expert = len(S)

    overrides: dict[str, Any] = {arch_key(arch, "expert_count"): new_expert}
    label = kv_get(kv, Keys.General.SIZE_LABEL)
    if label is not None and "x" in label:
        head, _, tail = label.partition("x")
        if head.isdigit():
            overrides[Keys.General.SIZE_LABEL] = f"{new_expert}x{tail}"
            print(f"size_label: {label!r} -> {overrides[Keys.General.SIZE_LABEL]!r}")

    drop = [k for k in kv if "imatrix" in k]
    if drop:
        print(f"dropping stale imatrix keys: {drop}")
    split_keys = [k for k in kv if k.startswith("split.")]
    if split_keys:
        raise SystemExit(f"input is a shard ({split_keys}); merge shards with gguf-split first")
    for k in (arch_key(arch, "attention.compress_ratios"), arch_key(arch, "attention.recurrent_layers")):
        if k in kv:
            print(f"preserving verbatim {k}: {len(kv[k])} entries (block_count={rep['block_count']})")

    if args.dry_run:
        print(f"dry run: would write {len(r.tensors)} tensors, "
              f"{sum(t.n_bytes for t in r.tensors) + sum(v['n_bytes_new'] - v['n_bytes_old'] for v in plan.values())} bytes")
        return 0

    writer = gguf.GGUFWriter(str(out_path), arch, endianess=r.endianess)
    writer.data_alignment = r.alignment
    for name, field in r.fields.items():
        if name.startswith("GGUF.") or name == Keys.General.ARCHITECTURE:
            continue
        if name in drop:
            continue
        vtype = field.types[0]
        sub = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        val = field.contents()
        if name in overrides:
            new = overrides.pop(name)
            print(f"override {name}: {val!r} -> {new!r}")
            writer.add_key_value(name, new, vtype, sub_type=sub)
        elif val is not None:
            writer.add_key_value(name, val, vtype, sub_type=sub)
    for name, val in overrides.items():
        writer.add_key_value(name, val, gguf.GGUFValueType.INT32)
        print(f"added missing key {name} = {val}")

    for t in r.tensors:
        if t.name in plan:
            p = plan[t.name]
            writer.add_tensor_info(t.name, p["byte_shape"], np.uint8, p["n_bytes_new"], t.tensor_type)
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.n_bytes, t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    mm = np.memmap(mtp, dtype=np.uint8, mode="r")
    records = {}
    for t in r.tensors:
        if t.name in plan:
            p = plan[t.name]
            n_old = p["ne_old"][-1]
            view = mm[t.data_offset:t.data_offset + p["slab_bytes"] * n_old].reshape(n_old, p["slab_bytes"])
            sel = np.ascontiguousarray(view[np.asarray(S, dtype=np.int64)])
            if sel.nbytes != p["n_bytes_new"]:
                raise SystemExit(f"{t.name}: gathered {sel.nbytes} bytes, planned {p['n_bytes_new']}")
            h = hashlib.sha256()
            for row in sel:
                h.update(np.ascontiguousarray(row).tobytes())
            records[t.name] = {"ne_new": p["ne_new"], "n_bytes_new": p["n_bytes_new"],
                               "sha256": h.hexdigest()}
            writer.write_tensor_data(sel, tensor_endianess=r.endianess)
        else:
            writer.write_tensor_data(t.data, tensor_endianess=r.endianess)
    writer.close()

    side = out_path.with_name(out_path.name + ".surgery.json")
    side.write_text(json.dumps({
        "tool": "mtp_prune_surgery p3_apply",
        "schema": 1,
        "source": str(mtp),
        "source_header_sha256": file_fingerprint(mtp)["header_sha256"],
        "mask": str(mask_json),
        "slabs": records,
        "S": S,
        "alignment": int(r.alignment),
    }, indent=1), encoding="utf-8")
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes) and provenance {side}")
    return 0


# ---------------------------------------------------------------------------
# p4: post-surgery verification
# ---------------------------------------------------------------------------


def hash_tensor(path: Path, name: str) -> str:
    t = next(x for x in gguf.GGUFReader(path).tensors if x.name == name)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    h = hashlib.sha256()
    view = mm[t.data_offset:t.data_offset + t.n_bytes]
    step = 1 << 20
    for i in range(0, view.nbytes, step):
        h.update(np.ascontiguousarray(view[i:i + step]).tobytes())
    return h.hexdigest()



def cmd_p4_verify(out_path: Path, trunk_paths: Sequence[Path], args) -> int:
    print(f"== p4_verify {out_path} against trunk {[str(p) for p in trunk_paths]}")
    r = gguf.GGUFReader(out_path)
    kv = kv_of(r)
    arch = kv_get(kv, Keys.General.ARCHITECTURE)
    checks: list[bool] = []

    def ck(cond: bool, msg: str):
        checks.append(bool(cond))
        print(("  ok   " if cond else "  FAIL ") + msg)

    _primary, _trunk_tensors, _owner, trunk_kv = pool_readers(trunk_paths)
    tarch = kv_get(trunk_kv, Keys.General.ARCHITECTURE)
    trunk_experts = kv_get(trunk_kv, arch_key(tarch, "expert_count")) if tarch else None
    side = Path(str(out_path) + ".surgery.json")
    proven = json.loads(side.read_text(encoding="utf-8")) if side.exists() else None
    expected = trunk_experts if trunk_experts is not None else (len(proven["S"]) if proven else None)
    if proven and trunk_experts is not None and len(proven["S"]) != trunk_experts:
        print(f"  note S keeps {len(proven['S'])} experts while the trunk has {trunk_experts}: "
              "this is the un-aligned control config, the shape asserts below are expected to fail")

    ck(tarch == arch, f"arch matches trunk: sidecar={arch} trunk={tarch}")
    ck(expected is not None, f"target expert count known: {expected}")
    n_expert = kv_get(kv, arch_key(arch, "expert_count"))
    ck(n_expert is not None and n_expert == expected, f"header {arch}.expert_count = {n_expert} == {expected}")
    ck(kv_get(kv, arch_key(arch, "expert_used_count"), 0) <= (n_expert or 0), "expert_used_count <= expert_count")
    nextn = nextn_layers(r)
    ck(len(nextn) == kv_get(kv, arch_key(arch, "nextn_predict_layers"), -1),
       f"nextn_predict_layers = {kv_get(kv, arch_key(arch, 'nextn_predict_layers'))}, nextn tensors in {sorted(nextn)}")
    ck(len(nextn) == 1, "exactly one speculative block")
    spec = next(iter(nextn)) if len(nextn) == 1 else None
    n_block = kv_get(kv, arch_key(arch, "block_count"))
    ck(spec == (n_block or 0) - 1, f"speculative block {spec} == block_count {n_block} - 1")
    for k in ("attention.compress_ratios", "attention.recurrent_layers"):
        arr = kv_get(kv, arch_key(arch, k))
        if arr is not None:
            ck(len(arr) == n_block, f"{arch}.{k} still has block_count entries ({len(arr)})")
    ck(not any(k.startswith("split.") for k in kv), "output is a single file (no split.* keys)")
    carriers = expert_tensors(r, n_expert or 0)
    ck(len(carriers) == 4, f"exactly 4 expert-axis tensors: {sorted(carriers)}")
    for name, info in sorted(carriers.items()):
        ck(info["ne"][-1] == n_expert, f"{name} expert axis ne[-1] = {info['ne'][-1]} == {n_expert}")
        ck(info["n_bytes"] == info["slab_bytes"] * info["n_slabs"],
           f"{name} byte math {info['slab_bytes']} x {info['n_slabs']} == {info['n_bytes']}")
    ck(not any(".nextn." in n for n in carriers), "no nextn tensor carries an expert axis")
    for k in ("vocab_size", "embedding_length", "hyper_connection.count"):
        a, b = kv_get(kv, arch_key(arch, k)), kv_get(trunk_kv, arch_key(tarch, k))
        if a is not None and b is not None:
            ck(a == b, f"sidecar/trunk agree on {k} ({a} == {b})")
    hc = kv_get(kv, arch_key(arch, "hyper_connection.count")) or 1
    hc_t = kv_get(trunk_kv, arch_key(tarch, "hyper_connection.count")) or 1
    e_a = kv_get(kv, arch_key(arch, "embedding_length")) or 0
    e_t = kv_get(trunk_kv, arch_key(tarch, "embedding_length")) or 0
    ck(e_a * hc == e_t * hc_t, f"n_embd_out agree (the only cross-file check in common_speculative.cpp): "
                               f"sidecar {e_a * hc} == trunk {e_t * hc_t}")

    if proven:
        for name, rec in sorted(proven["slabs"].items()):
            ck(hash_tensor(out_path, name) == rec["sha256"], f"{name} output bytes match recorded sha256")
        src = Path(proven["source"])
        if src.exists():
            rs = gguf.GGUFReader(src)
            mm = np.memmap(src, dtype=np.uint8, mode="r")
            for name, rec in sorted(proven["slabs"].items()):
                t = next(x for x in rs.tensors if x.name == name)
                ne = [int(d) for d in t.shape]
                _, slab_bytes, n_slabs, _ = slab_math(ne, t.tensor_type)
                ck(gather_hash(mm, t.data_offset, slab_bytes, n_slabs, proven["S"]) == rec["sha256"],
                   f"{name} re-gather from the stock sidecar is bit-identical")
        else:
            print(f"  note source {src} absent: gather chain not re-verified")
    else:
        print("  note no .surgery.json next to the output: byte-exactness not re-verified")

    failed = checks.count(False)
    print(f"verification: {len(checks) - failed}/{len(checks)} checks passed")
    print_benchmark_playbook(out_path, trunk_paths)
    return 1 if failed else 0



def print_benchmark_playbook(out_path: Path, trunk_paths: Sequence[Path]) -> None:
    trunk = str(trunk_paths[0])
    print("\n=== llama-cli benchmark playbook (3-way acceptance-rate test) ===")
    print("# The counters are printed by common/speculative.cpp (SPC_TRC -> LOG_TRC), so add")
    print("# -v / --log-verbose; grep 'statistics' for #gen drafts, #acc drafts, #acc tokens,")
    print("# and '#acc rate/pos = (...)' which is the per-position acceptance rate.")
    print("""
TRUNK_PRUNED="%s"
MTP_SURGERY="%s"
MTP_STOCK="mtp-Qwen3.8-Flash-Next-Q4_0.gguf"
TRUNK_STOCK="<stock-512-expert-trunk-00001-of-0000N.gguf>"

bench() {
  local tag="$1" trunk="$2" side="$3"
  llama-cli -m "$trunk" --spec-draft-model "$side" --spec-type draft-mtp \\
      --spec-draft-n-max 4 -ngl 99 -ngld 99 -p "Write a short function that parses an ISO-8601 date." \\
      -n 256 -v 2>&1 | tee "spec-$tag.log" \\
      | grep -E 'statistics .*draft-mtp|draft-mtp is not usable|n_embd_out'
}

bench baseline "$TRUNK_STOCK" "$MTP_STOCK"
bench target   "$TRUNK_PRUNED"  "$MTP_SURGERY"
bench control  "$TRUNK_PRUNED"  "$MTP_STOCK"

for f in spec-baseline.log spec-target.log spec-control.log; do
  grep -E 'statistics .*draft-mtp' "$f"
done

# expected reading of the three runs:
#   baseline : the reference acceptance rate at 512 experts on both sides
#   target   : the engineered pair; compare #acc rate/pos and #mean acc len to baseline
#   control  : if this runs at all, the loader accepted a 512-expert sidecar against a
#              256-expert trunk, i.e. common_speculative_mtp_is_available only compares
#              n_embd_out and n_layer_nextn, and the slicing was a size choice, not a fix
""" % (trunk, out_path))


def main() -> None:
    ap = argparse.ArgumentParser(description="qwen4exp MTP sidecar expert-surgery tool")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p0 = sp.add_parser("p0_probe")
    p0.add_argument("mtp", type=Path)

    p1 = sp.add_parser("p1_mask")
    p1.add_argument("trunk", type=Path, nargs="+")
    p1.add_argument("--manifest", type=Path)
    p1.add_argument("--reference", type=Path, nargs="+")
    p1.add_argument("--sidecar", type=Path)
    p1.add_argument("--policy", default="auto",
                    choices=("auto", "manifest", "layer", "vote", "identity", "self-router", "explicit"))
    p1.add_argument("--mask", type=str)
    p1.add_argument("--mask-file", type=Path)
    p1.add_argument("--layer", type=int)
    p1.add_argument("--keep", type=int)
    p1.add_argument("--sample-layers", type=int, nargs="+")
    p1.add_argument("--block-window", type=int, default=8)
    p1.add_argument("--margin", type=float, default=0.25)
    p1.add_argument("-o", "--out", type=Path, default=Path("mask.json"))

    p2 = sp.add_parser("p2_plan")
    p2.add_argument("mtp", type=Path)
    p2.add_argument("mask", type=Path)

    p3 = sp.add_parser("p3_apply")
    p3.add_argument("mtp", type=Path)
    p3.add_argument("mask", type=Path)
    p3.add_argument("-o", "--out", type=Path, required=True)
    p3.add_argument("--dry-run", action="store_true")

    p4 = sp.add_parser("p4_verify")
    p4.add_argument("out", type=Path)
    p4.add_argument("trunk", type=Path, nargs="+")

    args = ap.parse_args()
    mask_arg: list[int] | None = None
    if getattr(args, "mask", None):
        mask_arg = [int(x) for x in args.mask.split(",") if x.strip() != ""]
    elif getattr(args, "mask_file", None):
        mask_arg = [int(x) for x in Path(args.mask_file).read_text(encoding="utf-8").replace(",", " ").split()]

    if args.cmd == "p0_probe":
        return cmd_p0_probe(args.mtp)
    if args.cmd == "p1_mask":
        return cmd_p1_mask(args.trunk, args.out, args, mask_arg)
    if args.cmd == "p2_plan":
        return cmd_p2_plan(args.mtp, args.mask)
    if args.cmd == "p3_apply":
        return cmd_p3_apply(args.mtp, args.mask, args.out, args)
    return cmd_p4_verify(args.out, args.trunk, args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())

