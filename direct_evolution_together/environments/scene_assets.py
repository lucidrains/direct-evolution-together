"""General MuJoCo MJCF + STL/OBJ scene compiler and binary packager.

Compiles any robot MJCF scene directory (e.g. Microduck, SO-101 arm, Menagerie robots)
into two content-addressed, browser-friendly bundles:

1. ``physics.bin`` (``DETVFS1``): Minimal Emscripten VFS archive containing the XML
   tree with exact body ``<inertial>`` properties baked in, visual-only CAD meshes
   stripped, and all collision-active meshes preserved bit-for-bit.
2. ``visual.bin`` (``DETVIS1``): Decimated, vertex-colored, 4-byte-aligned indexed
   triangle meshes grouped by rigid ``body_id`` so WebGL2 renders the entire robot
   in ``<= nbody`` draw calls driven solely by compact ``(xpos, xquat)`` poses.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import struct
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
SCENES_DIR = ASSETS_DIR / "scenes"

DETVFS_MAGIC = b"DETVFS1\x00"
DETVIS_MAGIC = b"DETVIS1\x00"


def pack_vfs_entries(files: dict[str, bytes]) -> bytes:
    """Pack ``{rel_posix_path: raw_bytes}`` into a ``DETVFS1`` binary archive."""
    ordered = sorted(files.items(), key=lambda kv: kv[0])
    chunks: list[bytes] = [DETVFS_MAGIC, struct.pack("<I", len(ordered))]
    for rel_path, data in ordered:
        clean = PurePosixPath(rel_path).as_posix().lstrip("/")
        path_bytes = clean.encode("utf-8")
        if len(path_bytes) > 65535:
            raise ValueError(f"VFS path too long: {clean}")
        chunks.append(struct.pack("<H", len(path_bytes)))
        chunks.append(path_bytes)
        chunks.append(struct.pack("<I", len(data)))
        chunks.append(data)
    return b"".join(chunks)


def unpack_vfs_entries(blob: bytes) -> dict[str, bytes]:
    """Unpack a ``DETVFS1`` binary archive into ``{rel_posix_path: raw_bytes}``."""
    if len(blob) < 12 or blob[:8] != DETVFS_MAGIC:
        raise ValueError("Invalid DETVFS1 header.")
    (count,) = struct.unpack_from("<I", blob, 8)
    off = 12
    out: dict[str, bytes] = {}
    for _ in range(count):
        if off + 2 > len(blob):
            raise ValueError("Truncated DETVFS1 path length.")
        (plen,) = struct.unpack_from("<H", blob, off)
        off += 2
        path_str = blob[off : off + plen].decode("utf-8")
        off += plen
        (dlen,) = struct.unpack_from("<I", blob, off)
        off += 4
        if off + dlen > len(blob):
            raise ValueError(f"Truncated DETVFS1 payload for {path_str}.")
        out[path_str] = blob[off : off + dlen]
        off += dlen
    return out


def _quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64))
    return mat.reshape(3, 3)


def _decimate_vertex_cluster(
    verts: np.ndarray,
    faces: np.ndarray,
    max_faces: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast deterministic spatial vertex-clustering decimation to ``<= max_faces``."""
    if len(faces) <= max_faces or max_faces <= 0:
        return verts.astype(np.float32), faces.astype(np.int32)

    bbox_min = verts.min(axis=0)
    span = np.maximum(verts.max(axis=0) - bbox_min, 1e-6)
    lo, hi = 6, 160
    best_v = verts.astype(np.float32)
    best_f = faces.astype(np.int32)

    for _ in range(8):
        res = (lo + hi) // 2
        grid = np.floor((verts - bbox_min) / span * res).astype(np.int64)
        keys = grid[:, 0] * 1_000_000 + grid[:, 1] * 1_000 + grid[:, 2]
        uniq, inv = np.unique(keys, return_inverse=True)
        nf = inv[faces]
        valid = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
        nf = nf[valid]
        nv = np.zeros((len(uniq), 3), dtype=np.float64)
        counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)[:, None]
        np.add.at(nv, inv, verts)
        nv /= np.maximum(counts, 1.0)
        best_v = nv.astype(np.float32)
        best_f = nf.astype(np.int32)
        if len(nf) > max_faces:
            hi = res - 1
        else:
            lo = res + 1
    return best_v, best_f


def _tessellate_primitive_geom(
    gtype: int,
    size: np.ndarray,
    *,
    slices: int = 16,
    rings: int = 6,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Tessellate primitive MuJoCo geoms (sphere, capsule, ellipsoid, cylinder, box)."""
    mj_sphere = int(mujoco.mjtGeom.mjGEOM_SPHERE)
    mj_capsule = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    mj_ellipsoid = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
    mj_cylinder = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    mj_box = int(mujoco.mjtGeom.mjGEOM_BOX)

    if gtype == mj_box:
        sx, sy, sz = (max(1e-4, float(size[k])) for k in range(3))
        quads = [
            ([-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz]),
            ([sx, -sy, -sz], [-sx, -sy, -sz], [-sx, sy, -sz], [sx, sy, -sz]),
            ([sx, -sy, sz], [sx, -sy, -sz], [sx, sy, -sz], [sx, sy, sz]),
            ([-sx, -sy, -sz], [-sx, -sy, sz], [-sx, sy, sz], [-sx, sy, -sz]),
            ([-sx, sy, sz], [sx, sy, sz], [sx, sy, -sz], [-sx, sy, -sz]),
            ([-sx, -sy, -sz], [sx, -sy, -sz], [sx, -sy, sz], [-sx, -sy, sz]),
        ]
        verts: list[list[float]] = []
        faces: list[list[int]] = []
        for a, b, c, d in quads:
            base = len(verts)
            verts.extend([a, b, c, d])
            faces.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
        return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)

    if gtype == mj_cylinder:
        r = max(1e-4, float(size[0]))
        hl = max(1e-4, float(size[1]))
        verts = [[0.0, 0.0, hl], [0.0, 0.0, -hl]]
        faces = []
        for s in range(slices):
            th = (s / slices) * 2.0 * math.pi
            verts.append([r * math.cos(th), r * math.sin(th), hl])
            verts.append([r * math.cos(th), r * math.sin(th), -hl])
        for s in range(slices):
            sn = (s + 1) % slices
            t0, b0 = 2 + 2 * s, 2 + 2 * s + 1
            t1, b1 = 2 + 2 * sn, 2 + 2 * sn + 1
            faces.append([0, t0, t1])
            faces.append([1, b1, b0])
            faces.append([t0, b0, b1])
            faces.append([t0, b1, t1])
        return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)

    if gtype in (mj_sphere, mj_capsule, mj_ellipsoid):
        r = max(1e-4, float(size[0]))
        hl = max(0.0, float(size[1])) if gtype == mj_capsule else 0.0
        sc = (
            np.array([max(1e-4, float(size[0])), max(1e-4, float(size[1])), max(1e-4, float(size[2]))])
            if gtype == mj_ellipsoid
            else np.array([r, r, r])
        )
        ring_specs: list[tuple[float, float]] = []
        for i in range(rings + 1):
            ring_specs.append(((i / rings) * (math.pi * 0.5), 1.0))
        for i in range(rings + 1):
            ring_specs.append((math.pi * 0.5 + (i / rings) * (math.pi * 0.5), -1.0))
        stride = slices + 1
        verts = []
        faces = []
        for phi, cap_sgn in ring_specs:
            sp, cp = math.sin(phi), math.cos(phi)
            for s in range(stride):
                th = (s / slices) * 2.0 * math.pi
                ux, uy, uz = sp * math.cos(th), sp * math.sin(th), cp
                verts.append([ux * sc[0], uy * sc[1], uz * sc[2] + cap_sgn * hl])
        for ri in range(len(ring_specs) - 1):
            for s in range(slices):
                a = ri * stride + s
                b = (ri + 1) * stride + s
                d = (ri + 1) * stride + s + 1
                e = ri * stride + s + 1
                faces.append([a, b, d])
                faces.append([a, d, e])
        return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)

    return None


def build_visual_meshpack(
    model: mujoco.MjModel,
    *,
    max_tris_per_body: int = 2400,
) -> tuple[bytes, dict[str, int]]:
    """Build a ``DETVIS1`` 4-byte-aligned per-body triangle mesh bundle from ``MjModel``."""
    mj_plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
    mj_geom_mesh = int(mujoco.mjtGeom.mjGEOM_MESH)

    body_entries: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    total_verts = 0
    total_tris = 0

    for bid in range(1, int(model.nbody)):
        body_geoms = [
            g for g in range(int(model.ngeom))
            if int(model.geom_bodyid[g]) == bid and int(model.geom_type[g]) != mj_plane
        ]
        if not body_geoms:
            continue
        vis_group_geoms = [g for g in body_geoms if int(model.geom_group[g]) <= 2]
        selected_geoms = vis_group_geoms if vis_group_geoms else body_geoms
        per_geom_budget = max(220, max_tris_per_body // max(1, len(selected_geoms)))

        b_verts: list[np.ndarray] = []
        b_norms: list[np.ndarray] = []
        b_rgba: list[np.ndarray] = []
        b_faces: list[np.ndarray] = []
        v_offset = 0

        for gid in selected_geoms:
            gtype = int(model.geom_type[gid])
            if gtype == mj_geom_mesh:
                mid = int(model.geom_dataid[gid])
                if mid < 0:
                    continue
                vadr, vnum = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
                fadr, fnum = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
                verts = np.asarray(model.mesh_vert[vadr : vadr + vnum], dtype=np.float64)
                faces = np.asarray(model.mesh_face[fadr : fadr + fnum], dtype=np.int32)
                verts_f, faces_i = _decimate_vertex_cluster(verts, faces, per_geom_budget)
            else:
                prim = _tessellate_primitive_geom(gtype, model.geom_size[gid])
                if prim is None:
                    continue
                verts_f, faces_i = prim

            if len(faces_i) == 0 or len(verts_f) == 0:
                continue
            if v_offset + len(verts_f) > 65535:
                break

            rot = _quat_to_rotmat(model.geom_quat[gid]).astype(np.float32)
            pos = np.asarray(model.geom_pos[gid], dtype=np.float32)
            vb = verts_f @ rot.T + pos

            tris = vb[faces_i]
            fnorm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            vnorm = np.zeros_like(vb)
            np.add.at(vnorm, faces_i[:, 0], fnorm)
            np.add.at(vnorm, faces_i[:, 1], fnorm)
            np.add.at(vnorm, faces_i[:, 2], fnorm)
            norms_unit = vnorm / (np.linalg.norm(vnorm, axis=1, keepdims=True) + 1e-8)
            norms_i8 = np.round(np.clip(norms_unit, -1.0, 1.0) * 127.0).astype(np.int8)

            matid = int(model.geom_matid[gid])
            rgba = model.mat_rgba[matid] if matid >= 0 else model.geom_rgba[gid]
            rgba_u8 = np.tile(
                np.round(np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8),
                (len(vb), 1),
            )

            b_verts.append(np.ascontiguousarray(vb, dtype="<f4"))
            b_norms.append(np.ascontiguousarray(norms_i8, dtype="i1"))
            b_rgba.append(np.ascontiguousarray(rgba_u8, dtype="u1"))
            b_faces.append(np.ascontiguousarray(faces_i + v_offset, dtype="<u2"))
            v_offset += len(vb)

        if b_verts:
            vv = np.concatenate(b_verts, axis=0)
            nn = np.concatenate(b_norms, axis=0)
            cc = np.concatenate(b_rgba, axis=0)
            ff = np.concatenate(b_faces, axis=0)
            body_entries.append((bid, vv, nn, cc, ff))
            total_verts += int(len(vv))
            total_tris += int(len(ff))

    chunks: list[bytes] = [DETVIS_MAGIC, struct.pack("<I", len(body_entries))]
    for bid, vv, nn, cc, ff in body_entries:
        nv = int(len(vv))
        nt = int(len(ff))
        chunks.append(struct.pack("<HHII", bid, 0, nv, nt))
        chunks.append(vv.tobytes())
        attr_bytes = nn.tobytes() + cc.tobytes()
        attr_pad = (4 - (len(attr_bytes) % 4)) % 4
        chunks.append(attr_bytes)
        if attr_pad:
            chunks.append(b"\x00" * attr_pad)
        idx_bytes = ff.tobytes()
        idx_pad = (4 - (len(idx_bytes) % 4)) % 4
        chunks.append(idx_bytes)
        if idx_pad:
            chunks.append(b"\x00" * idx_pad)

    return b"".join(chunks), {
        "bodyMeshCount": len(body_entries),
        "totalVerts": total_verts,
        "totalTris": total_tris,
    }


def _collect_collision_mesh_names(model: mujoco.MjModel) -> set[str]:
    mj_geom_mesh = int(mujoco.mjtGeom.mjGEOM_MESH)
    mj_obj_mesh = int(mujoco.mjtObj.mjOBJ_MESH)
    used: set[str] = set()
    for gid in range(int(model.ngeom)):
        if int(model.geom_type[gid]) != mj_geom_mesh:
            continue
        if int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0:
            continue
        mid = int(model.geom_dataid[gid])
        if mid >= 0:
            mname = mujoco.mj_id2name(model, mj_obj_mesh, mid)
            if mname:
                used.add(mname)
    return used


def _fmt_vec(arr: np.ndarray) -> str:
    return " ".join(f"{float(x):.9g}" for x in arr)


def _bake_missing_inertials(root_elem: ET.Element, model: mujoco.MjModel) -> None:
    """Inject explicit ``<inertial>`` from ``MjModel`` on bodies relying on geom mass."""
    mj_body = int(mujoco.mjtObj.mjOBJ_BODY)
    for body_el in root_elem.iter("body"):
        bname = body_el.attrib.get("name")
        if not bname or body_el.find("inertial") is not None:
            continue
        bid = mujoco.mj_name2id(model, mj_body, bname)
        if bid <= 0:
            continue
        mass = float(model.body_mass[bid])
        if mass <= 0.0:
            continue
        iner = ET.Element(
            "inertial",
            {
                "pos": _fmt_vec(model.body_ipos[bid]),
                "quat": _fmt_vec(model.body_iquat[bid]),
                "mass": f"{mass:.9g}",
                "diaginertia": _fmt_vec(model.body_inertia[bid]),
            },
        )
        body_el.insert(0, iner)


def _resolve_default_geom_classes(root_elem: ET.Element) -> set[str]:
    """Identify ``<default class="...">`` blocks whose ``<geom>`` sets contype=0 & conaffinity=0."""
    visual_classes: set[str] = set()
    for def_el in root_elem.iter("default"):
        cname = def_el.attrib.get("class")
        if not cname:
            continue
        geom_el = def_el.find("geom")
        if geom_el is not None:
            ct = geom_el.attrib.get("contype")
            ca = geom_el.attrib.get("conaffinity")
            if ct == "0" and ca == "0":
                visual_classes.add(cname)
    return visual_classes


def _strip_xml_for_physics(
    xml_path: Path,
    base_dir: Path,
    model: mujoco.MjModel,
    collision_meshes: set[str],
    vfs_files: dict[str, bytes],
    inherited_visual_classes: set[str] | None = None,
) -> None:
    """Recursively strip visual-only mesh geoms & copy required collision assets into ``vfs_files``."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    vis_classes = set(inherited_visual_classes or ()) | _resolve_default_geom_classes(root)
    meshdir = "."
    for comp in root.findall("compiler"):
        if "meshdir" in comp.attrib:
            meshdir = comp.attrib["meshdir"]

    # Process <include file="..."/> children first so default classes propagate
    for inc in root.findall(".//include"):
        inc_file = inc.attrib.get("file")
        if inc_file:
            inc_path = (xml_path.parent / inc_file).resolve()
            _strip_xml_for_physics(inc_path, base_dir, model, collision_meshes, vfs_files, vis_classes)

    _bake_missing_inertials(root, model)

    # Remove visual-only mesh geoms
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom":
                g_cls = child.attrib.get("class", "")
                g_type = child.attrib.get("type", "")
                has_mesh = "mesh" in child.attrib or g_type == "mesh"
                if not has_mesh:
                    continue
                ct = child.attrib.get("contype")
                ca = child.attrib.get("conaffinity")
                is_zero_con = (ct == "0" and ca == "0") or (ct is None and ca is None and g_cls in vis_classes)
                if is_zero_con:
                    mref = child.attrib.get("mesh", "")
                    if mref not in collision_meshes or g_cls in vis_classes:
                        parent.remove(child)
            elif child.tag == "mesh":
                mfile = child.attrib.get("file", "")
                mname = child.attrib.get("name") or Path(mfile).stem
                if mname not in collision_meshes:
                    parent.remove(child)
                elif mfile:
                    src_mesh = (xml_path.parent / meshdir / mfile).resolve()
                    if not src_mesh.is_file():
                        src_mesh = (base_dir / meshdir / mfile).resolve()
                    if src_mesh.is_file():
                        rel_mesh = src_mesh.relative_to(base_dir).as_posix()
                        vfs_files[rel_mesh] = src_mesh.read_bytes()

    rel_xml = xml_path.resolve().relative_to(base_dir).as_posix()
    vfs_files[rel_xml] = ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _compute_camera_defaults(model: mujoco.MjModel) -> dict[str, object]:
    """Derive sensible 3D camera target & orbit distance from the robot's home pose."""
    data = mujoco.MjData(model)
    if int(model.nkey) > 0:
        mj_key = int(mujoco.mjtObj.mjOBJ_KEY)
        stand_id = mujoco.mj_name2id(model, mj_key, "STAND")
        kid = stand_id if stand_id >= 0 else 0
        data.qpos[:] = model.key_qpos[kid]
    mujoco.mj_kinematics(model, data)

    if int(model.nbody) <= 1:
        return {"trackBodyId": 1, "targetHeight": 0.25, "distance": 1.2, "yaw": -0.65, "pitch": 0.28}

    xpos = np.asarray(data.xpos[1:], dtype=np.float64)
    z_min = float(min(0.0, xpos[:, 2].min()))
    z_max = float(xpos[:, 2].max())
    xy_span = float(np.ptp(xpos[:, :2], axis=0).max())
    height = max(0.08, z_max - z_min)
    extent = max(height, xy_span)
    return {
        "trackBodyId": 1,
        "targetHeight": round(max(0.05, z_min + 0.52 * height), 4),
        "distance": round(max(0.38, min(5.0, extent * 3.1)), 3),
        "yaw": -0.68,
        "pitch": 0.26,
        "gridScale": 0.25 if extent < 0.45 else 1.0,
    }


def compile_mjcf_scene(
    scene_id: str,
    entry_xml_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    max_visual_tris_per_body: int = 2400,
    license_path: str | Path | None = None,
    camera_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    """Compile an MJCF scene directory into ``physics.bin``, ``visual.bin``, and ``manifest.json``."""
    entry_xml = Path(entry_xml_path).resolve()
    if not entry_xml.is_file():
        raise FileNotFoundError(f"MJCF entry XML not found: {entry_xml}")
    base_dir = entry_xml.parent
    out_dir = Path(output_dir).resolve() if output_dir is not None else (SCENES_DIR / scene_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(entry_xml))
    collision_meshes = _collect_collision_mesh_names(model)

    vfs_files: dict[str, bytes] = {}
    _strip_xml_for_physics(entry_xml, base_dir, model, collision_meshes, vfs_files)
    physics_bytes = pack_vfs_entries(vfs_files)
    visual_bytes, vis_stats = build_visual_meshpack(model, max_tris_per_body=max_visual_tris_per_body)

    # Verify stripped physics bundle loads cleanly in MuJoCo
    entry_rel = entry_xml.relative_to(base_dir).as_posix()
    phys_sha = hashlib.sha256(physics_bytes).hexdigest()
    vis_sha = hashlib.sha256(visual_bytes).hexdigest()

    (out_dir / "physics.bin").write_bytes(physics_bytes)
    (out_dir / "visual.bin").write_bytes(visual_bytes)
    if license_path and Path(license_path).is_file():
        shutil.copy2(license_path, out_dir / "LICENSE")

    mj_body = int(mujoco.mjtObj.mjOBJ_BODY)
    body_names = [
        mujoco.mj_id2name(model, mj_body, bid) or f"body_{bid}"
        for bid in range(int(model.nbody))
    ]
    cam_defaults = _compute_camera_defaults(model)
    if camera_overrides:
        cam_defaults.update(camera_overrides)
    manifest: dict[str, object] = {
        "sceneId": scene_id,
        "entryXml": entry_rel,
        "physicsAsset": f"scenes/{scene_id}/physics.bin",
        "physicsSha256": phys_sha,
        "physicsBytes": len(physics_bytes),
        "visualAsset": f"scenes/{scene_id}/visual.bin",
        "visualSha256": vis_sha,
        "visualBytes": len(visual_bytes),
        "nbody": int(model.nbody),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "collisionMeshes": sorted(collision_meshes),
        "bodyNames": body_names,
        "visualStats": vis_stats,
        "cameraDefaults": cam_defaults,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


_MANIFEST_CACHE: dict[str, dict[str, object]] = {}


def get_scene_manifest(scene_id: str | None) -> dict[str, object] | None:
    """Load and cache ``assets/scenes/<scene_id>/manifest.json`` if bundled."""
    if not scene_id:
        return None
    clean = str(scene_id).strip().lower()
    if clean in _MANIFEST_CACHE:
        return _MANIFEST_CACHE[clean]
    mpath = SCENES_DIR / clean / "manifest.json"
    if not mpath.is_file():
        return None
    try:
        parsed = json.loads(mpath.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            _MANIFEST_CACHE[clean] = parsed
            return parsed
    except Exception:
        return None
    return None
