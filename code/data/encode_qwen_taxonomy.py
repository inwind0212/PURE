#!/usr/bin/env python3
"""Encode Overture 2024 POI text with Qwen3-Embedding-8B.

The text is always constructed from source fields as:
    A place of {category_leaf}, a type of {category_top}, named {name}.

The pre-existing description column is not used for non-China inputs. The China
source has no name column, so only the POI name is extracted from its Chinese
description; category_top and category_leaf still come from explicit columns.
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import sentence_transformers
import torch
import transformers
from sentence_transformers import SentenceTransformer


TEXT_FORMULA = "A place of {category_leaf}, a type of {category_top}, named {name}."
CHINA_PREFIX = "这个地点是"
CHINA_NAME_END = "，属于"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poi-parquet", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--read-batch-size", type=int, default=16384)
    parser.add_argument("--emb-dim", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--prompt-name", default="query")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def clean(value, field, row):
    if value is None:
        raise ValueError(f"Null {field} at row {row}")
    value = str(value).strip()
    if not value:
        raise ValueError(f"Empty {field} at row {row}")
    return value


def china_name(description, row):
    description = clean(description, "description", row)
    if not description.startswith(CHINA_PREFIX) or CHINA_NAME_END not in description:
        raise ValueError(f"Cannot extract China POI name at row {row}: {description[:120]!r}")
    name = description[len(CHINA_PREFIX):].split(CHINA_NAME_END, 1)[0].strip()
    return clean(name, "name", row)


def make_texts(batch, row_start, china_mode):
    data = batch.to_pydict()
    tops = data["category_top"]
    leaves = data["category_leaf"]
    if china_mode:
        names = [china_name(v, row_start + i) for i, v in enumerate(data["description"])]
    else:
        names = data["name"]
    texts = []
    for i, (name, top, leaf) in enumerate(zip(names, tops, leaves)):
        row = row_start + i
        name = clean(name, "name", row)
        top = clean(top, "category_top", row)
        leaf = clean(leaf, "category_leaf", row)
        texts.append(TEXT_FORMULA.format(category_leaf=leaf, category_top=top, name=name))
    return texts


def write_json(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main():
    args = parse_args()
    source = Path(args.poi_parquet).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{args.out_name}.npy"
    partial_path = out_dir / f"{args.out_name}.partial.npy"
    progress_path = out_dir / f"{args.out_name}.progress.json"
    meta_path = out_dir / f"{args.out_name}.meta.json"

    if args.skip_existing and final_path.exists() and meta_path.exists():
        print(f"SKIP complete output: {final_path}", flush=True)
        return

    parquet = pq.ParquetFile(source)
    total_rows = parquet.metadata.num_rows
    fields = set(parquet.schema_arrow.names)
    china_mode = "name" not in fields
    required = {"category_top", "category_leaf", "description" if china_mode else "name"}
    missing = sorted(required - fields)
    if missing:
        raise ValueError(f"Missing required columns in {source}: {missing}")
    columns = ["category_top", "category_leaf", "description" if china_mode else "name"]

    next_row = 0
    if partial_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("source") != str(source) or progress.get("total_rows") != total_rows:
            raise RuntimeError("Existing partial output does not match this source")
        next_row = int(progress["next_row"])
        output = np.lib.format.open_memmap(partial_path, mode="r+", dtype=np.float32,
                                           shape=(total_rows, args.emb_dim))
        print(f"RESUME row {next_row:,}/{total_rows:,}", flush=True)
    else:
        if partial_path.exists() or progress_path.exists():
            raise RuntimeError("Incomplete partial state; remove both partial and progress files")
        output = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32,
                                           shape=(total_rows, args.emb_dim))
        write_json(progress_path, {
            "source": str(source), "total_rows": total_rows, "next_row": 0,
            "text_formula": TEXT_FORMULA,
        })

    model = SentenceTransformer(
        args.model,
        device=args.device,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    model.max_seq_length = args.max_seq_length

    seen = 0
    started = time.time()
    first_text = None
    last_text = None
    for record_batch in parquet.iter_batches(batch_size=args.read_batch_size, columns=columns):
        batch_rows = record_batch.num_rows
        batch_end = seen + batch_rows
        if batch_end <= next_row:
            seen = batch_end
            continue
        if seen < next_row:
            record_batch = record_batch.slice(next_row - seen)
            seen = next_row
        texts = make_texts(record_batch, seen, china_mode)
        if first_text is None:
            first_text = texts[0]
            print(f"FORMULA: {TEXT_FORMULA}", flush=True)
            print(f"FIRST: {first_text}", flush=True)
        embeddings = model.encode(
            texts,
            batch_size=args.batch_size,
            prompt_name=args.prompt_name,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        if embeddings.shape[1] < args.emb_dim:
            raise RuntimeError(f"Model returned {embeddings.shape[1]} dimensions, need {args.emb_dim}")
        embeddings = np.asarray(embeddings[:, :args.emb_dim], dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if not np.all(np.isfinite(norms)) or np.any(norms == 0):
            raise RuntimeError(f"Invalid embedding norm near row {seen}")
        embeddings /= norms
        output[seen:seen + len(texts)] = embeddings
        output.flush()
        seen += len(texts)
        last_text = texts[-1]
        write_json(progress_path, {
            "source": str(source), "total_rows": total_rows, "next_row": seen,
            "text_formula": TEXT_FORMULA,
        })
        elapsed = time.time() - started
        rate = (seen - next_row) / elapsed if elapsed else 0.0
        eta = (total_rows - seen) / rate if rate else float("inf")
        print(f"PROGRESS {seen:,}/{total_rows:,} ({100 * seen / total_rows:.2f}%) "
              f"rate={rate:.1f} rows/s eta={eta / 3600:.2f}h", flush=True)

    if seen != total_rows:
        raise RuntimeError(f"Encoded {seen} rows, expected {total_rows}")
    del output
    os.replace(partial_path, final_path)
    progress_path.unlink()
    stat = source.stat()
    write_json(meta_path, {
        "status": "complete",
        "source": str(source),
        "source_size_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "rows": total_rows,
        "embedding_dimension": args.emb_dim,
        "dtype": "float32",
        "model": args.model,
        "prompt_name": args.prompt_name,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "text_formula": TEXT_FORMULA,
        "preexisting_description_used": False,
        "china_name_extracted_from_description": china_mode,
        "first_text": first_text,
        "last_text": last_text,
        "normalization": "MRL prefix truncation followed by L2 normalization",
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
        },
        "completed_unix_time": time.time(),
    })
    print(f"COMPLETE {final_path} shape=({total_rows}, {args.emb_dim})", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr, flush=True)
        raise
