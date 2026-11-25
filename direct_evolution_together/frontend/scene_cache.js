// Content-addressed IndexedDB + in-memory cache for heavy MuJoCo scene packs
// (DETVFS1 physics.bin, DETVIS1 visual.bin) and shared PopuLoRA base models.
(function (global) {
  "use strict";

  const DB_NAME = "det_scene_assets_v2";
  const STORE_NAME = "blobs";
  const memCache = new Map();
  const inFlight = new Map();
  let dbPromise = null;

  function openDb() {
    if (dbPromise) return dbPromise;
    if (typeof indexedDB === "undefined") {
      dbPromise = Promise.resolve(null);
      return dbPromise;
    }
    dbPromise = new Promise((resolve) => {
      try {
        const req = indexedDB.open(DB_NAME, 1);
        req.onupgradeneeded = () => {
          const db = req.result;
          if (!db.objectStoreNames.contains(STORE_NAME)) {
            db.createObjectStore(STORE_NAME);
          }
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => resolve(null);
      } catch {
        resolve(null);
      }
    });
    return dbPromise;
  }

  async function idbGet(key) {
    const db = await openDb();
    if (!db) return undefined;
    return new Promise((resolve) => {
      try {
        const tx = db.transaction(STORE_NAME, "readonly");
        const req = tx.objectStore(STORE_NAME).get(key);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => resolve(undefined);
      } catch {
        resolve(undefined);
      }
    });
  }

  async function idbSet(key, val) {
    const db = await openDb();
    if (!db) return;
    return new Promise((resolve) => {
      try {
        const tx = db.transaction(STORE_NAME, "readwrite");
        tx.objectStore(STORE_NAME).put(val, key);
        tx.oncomplete = () => resolve();
        tx.onerror = () => resolve();
      } catch {
        resolve();
      }
    });
  }

  function fetchBinaryAsset(url, cacheKey) {
    const key = cacheKey || `bin:${url}`;
    if (memCache.has(key)) return Promise.resolve(memCache.get(key));
    if (inFlight.has(key)) return inFlight.get(key);

    const p = (async () => {
      const hit = await idbGet(key);
      if (hit instanceof ArrayBuffer && hit.byteLength > 0) {
        memCache.set(key, hit);
        return hit;
      }
      const resp = await fetch(url, { cache: "force-cache" });
      if (!resp.ok) {
        throw new Error(`Failed to fetch scene asset ${url} (${resp.status})`);
      }
      const buf = await resp.arrayBuffer();
      memCache.set(key, buf);
      idbSet(key, buf).catch(() => {});
      return buf;
    })();

    inFlight.set(key, p);
    return p.finally(() => inFlight.delete(key));
  }

  function parseVisualMeshpack(arrayBuffer, manifest) {
    const bytes = new Uint8Array(arrayBuffer);
    if (bytes.byteLength < 12) throw new Error("Truncated DETVIS1 meshpack.");
    const magic = String.fromCharCode(
      bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6]
    );
    if (magic !== "DETVIS1") throw new Error("Invalid DETVIS1 header.");

    const view = new DataView(arrayBuffer);
    const numBodies = view.getUint32(8, true);
    let off = 12;
    const bodyMeshes = [];

    for (let i = 0; i < numBodies; i++) {
      const bodyId = view.getUint16(off, true);
      const numVerts = view.getUint32(off + 4, true);
      const numTris = view.getUint32(off + 8, true);
      off += 12;

      const posBytes = numVerts * 12;
      const pos = new Float32Array(arrayBuffer, off, numVerts * 3);
      off += posBytes;

      const normBytes = numVerts * 3;
      const norm = new Int8Array(arrayBuffer, off, normBytes);
      off += normBytes;

      const rgbaBytes = numVerts * 4;
      const rgba = new Uint8Array(arrayBuffer, off, rgbaBytes);
      off += rgbaBytes;

      const attrPad = (4 - ((normBytes + rgbaBytes) % 4)) % 4;
      off += attrPad;

      const idxBytes = numTris * 6;
      const idx = new Uint16Array(arrayBuffer, off, numTris * 3);
      off += idxBytes;

      const idxPad = (4 - (idxBytes % 4)) % 4;
      off += idxPad;

      bodyMeshes.push({ bodyId, numVerts, numTris, pos, norm, rgba, idx });
    }

    return {
      scene_id: manifest ? manifest.scene_id : "scene",
      sha256: manifest ? manifest.visual_sha256 : "",
      nbody: manifest ? manifest.nbody : numBodies + 1,
      camera_defaults: (manifest && manifest.camera_defaults) || {
        track_body_id: 1,
        target_height: 0.14,
        distance: 0.75,
        yaw: -0.68,
        pitch: 0.26,
        grid_scale: 0.25,
      },
      bodyMeshes,
    };
  }

  async function loadSceneVisualPack(basePath, manifest) {
    if (!manifest || !manifest.visual_asset) return null;
    const ver = (manifest.visual_sha256 || "").slice(0, 12);
    const parsedKey = `vis_parsed:${manifest.scene_id}:${ver}`;
    if (memCache.has(parsedKey)) {
      const hit = memCache.get(parsedKey);
      if (manifest.camera_defaults) hit.camera_defaults = manifest.camera_defaults;
      return hit;
    }
    const url = `${basePath}/assets/${manifest.visual_asset}${ver ? `?v=${ver}` : ""}`;
    const buf = await fetchBinaryAsset(url, `vis_bin:${manifest.scene_id}:${ver}`);
    const parsed = parseVisualMeshpack(buf, manifest);
    memCache.set(parsedKey, parsed);
    return parsed;
  }

  async function loadScenePhysicsBuffer(basePath, manifest) {
    if (!manifest || !manifest.physics_asset) return null;
    const ver = (manifest.physics_sha256 || "").slice(0, 12);
    const url = `${basePath}/assets/${manifest.physics_asset}${ver ? `?v=${ver}` : ""}`;
    return fetchBinaryAsset(url, `phys_bin:${manifest.scene_id}:${ver}`);
  }

  async function loadPopuloraBaseModel(basePath, taskId, baseHash = "", baseSeed = 1) {
    const cleanHash = /^[0-9a-f]{8,64}$/i.test(String(baseHash || "")) ? String(baseHash) : "";
    const key = `populora_base:${taskId}:${cleanHash || baseSeed}`;
    if (memCache.has(key)) return memCache.get(key);
    if (inFlight.has(key)) return inFlight.get(key);

    const p = (async () => {
      const hit = await idbGet(key);
      if (hit && typeof hit === "object" && hit.tensors && (hit.obs_dim || hit.obsDim)) {
        memCache.set(key, hit);
        return hit;
      }
      if (cleanHash) {
        try {
          const staticResp = await fetch(`${basePath}/media/base_models/${cleanHash}.json?v=2`, {
            cache: "force-cache",
          });
          if (staticResp.ok) {
            const parsed = await staticResp.json();
            if (parsed && parsed.tensors && (parsed.obs_dim || parsed.obsDim)) {
              memCache.set(key, parsed);
              idbSet(key, parsed).catch(() => {});
              return parsed;
            }
          }
        } catch {}
      }
      const qs = new URLSearchParams({ task: taskId, seed: String(baseSeed || 1) });
      if (cleanHash) qs.set("hash", cleanHash);
      const resp = await fetch(`${basePath}/api/sim/base-model/?${qs.toString()}`, {
        cache: "force-cache",
      });
      if (!resp.ok) {
        throw new Error(`Failed to fetch PopuLoRA base model (${resp.status})`);
      }
      const parsed = await resp.json();
      memCache.set(key, parsed);
      idbSet(key, parsed).catch(() => {});
      return parsed;
    })();

    inFlight.set(key, p);
    return p.finally(() => inFlight.delete(key));
  }

  global.DetSceneCache = {
    fetchBinaryAsset,
    parseVisualMeshpack,
    loadSceneVisualPack,
    loadScenePhysicsBuffer,
    loadPopuloraBaseModel,
  };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = global.DetSceneCache;
  }
})(typeof window !== "undefined" ? window : globalThis);
