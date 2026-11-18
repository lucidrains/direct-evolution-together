"""Stage bundled frontend scripts and simulation assets into a flat Nginx-ready directory."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
from pathlib import Path
import shutil

PKG_DIR = Path(__file__).resolve().parent

PUBLIC_ASSET_SUFFIXES = frozenset(
    {".js", ".mjs", ".css", ".xml", ".wasm", ".json", ".bin", ".stl", ".obj", ".png"}
)
PRECOMPRESS_SUFFIXES = frozenset(
    {".js", ".mjs", ".css", ".xml", ".wasm", ".json", ".bin", ".stl", ".obj"}
)


def iter_source_assets(pkg_dir: Path = PKG_DIR) -> dict[str, Path]:
    """Return mapping of relative POSIX URL path -> source file (`frontend/` overrides `assets/`)."""
    collected: dict[str, Path] = {}
    for base_dir in (pkg_dir / "assets", pkg_dir / "frontend"):
        if not base_dir.is_dir():
            continue
        root = base_dir.resolve()
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in PUBLIC_ASSET_SUFFIXES:
                continue
            rel = path.relative_to(root).as_posix()
            collected[rel] = path
    return collected


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_gzip_sidecar(target: Path, *, compresslevel: int = 9) -> Path | None:
    """Write `<target>.gz` with identical mtime for Nginx `gzip_static on;` if smaller than raw."""
    raw = target.read_bytes()
    if len(raw) < 256:
        return None
    gz = gzip.compress(raw, compresslevel=compresslevel, mtime=0)
    gz_path = target.with_name(target.name + ".gz")
    if len(gz) >= len(raw):
        if gz_path.exists():
            gz_path.unlink()
        return None
    tmp_gz = gz_path.with_name(gz_path.name + ".tmp")
    tmp_gz.write_bytes(gz)
    st = target.stat()
    os.utime(tmp_gz, ns=(st.st_atime_ns, st.st_mtime_ns))
    tmp_gz.replace(gz_path)
    return gz_path


def stage_public_assets(
    dest_dir: str | Path,
    *,
    precompress: bool = True,
    pkg_dir: Path = PKG_DIR,
) -> dict[str, int]:
    """Sync all `/assets/*` files into `dest_dir` with optional `.gz` sidecars for Nginx `gzip_static`."""
    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    sources = iter_source_assets(pkg_dir)

    expected_rels: set[str] = set()
    copied = 0
    gzipped = 0

    for rel, src in sources.items():
        expected_rels.add(rel)
        out = dest / Path(*rel.split("/"))
        out.parent.mkdir(parents=True, exist_ok=True)
        src_st = src.stat()
        needs_copy = (
            not out.is_file()
            or out.stat().st_size != src_st.st_size
            or _sha256_file(out) != _sha256_file(src)
        )
        if needs_copy:
            tmp = out.with_name(out.name + ".tmp")
            shutil.copy2(src, tmp)
            tmp.replace(out)
            copied += 1

        if precompress and out.suffix.lower() in PRECOMPRESS_SUFFIXES:
            gz_path = out.with_name(out.name + ".gz")
            expected_rels.add(rel + ".gz")
            out_st = out.stat()
            if needs_copy or not gz_path.is_file() or gz_path.stat().st_mtime_ns != out_st.st_mtime_ns:
                if write_gzip_sidecar(out) is not None:
                    gzipped += 1
                else:
                    expected_rels.discard(rel + ".gz")

    removed = 0
    for existing in sorted(dest.rglob("*"), reverse=True):
        if existing.is_file():
            rel_existing = existing.relative_to(dest).as_posix()
            if rel_existing not in expected_rels:
                existing.unlink()
                removed += 1
        elif existing.is_dir() and existing != dest:
            try:
                existing.rmdir()
            except OSError:
                pass

    return {
        "total": len(sources),
        "copied": copied,
        "gzipped": gzipped,
        "removed": removed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage direct-evolution-together public assets for Nginx")
    parser.add_argument("dest", help="Target directory to populate (aliased to /assets/ in Nginx)")
    parser.add_argument("--no-gzip", action="store_true", help="Skip generating .gz sidecars")
    args = parser.parse_args()
    stats = stage_public_assets(args.dest, precompress=not args.no_gzip)
    print(
        f"assets: {stats['total']} staged ({stats['copied']} updated, {stats['gzipped']} precompressed, {stats['removed']} pruned)"
    )


if __name__ == "__main__":
    main()
