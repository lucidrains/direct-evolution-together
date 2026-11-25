// Dedicated simulation & evaluation Web Worker for direct-evolution-together.
// Runs CartPole (pure JS), HalfCheetah (MuJoCo WASM), Microduck (PopuLoRA + MuJoCo WASM VFS),
// and general declarative MJCF scenes (`DETVFS1` physics packs) off the UI thread.

let taskSpecs = {};
let basePath = "";
let mujocoWasmUrl = "";
let cheetahXmlUrl = "";

let mujocoPromise = null;
let cheetahModel = null;
let cheetahData = null;

const mountedScenes = new Map(); // sceneId -> { model, data, meta }
const physicsBufCache = new Map(); // sceneId -> ArrayBuffer
const baseModelCache = new Map(); // `${taskId}:${baseHash}` -> parsed base model

// Precomputed O(1) IEEE-754 float16 -> float32 lookup table
const F16_TO_F32 = (() => {
  const tbl = new Float32Array(65536);
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  for (let h = 0; h < 65536; h++) {
    const s = (h & 0x8000) << 16;
    const e = (h >> 10) & 0x1f;
    const m = h & 0x03ff;
    if (e === 0) {
      if (m === 0) {
        u32[0] = s;
      } else {
        let mm = m;
        let ee = -14;
        while ((mm & 0x0400) === 0) {
          mm <<= 1;
          ee--;
        }
        mm &= 0x03ff;
        u32[0] = s | ((ee + 127) << 23) | (mm << 13);
      }
    } else if (e === 31) {
      u32[0] = s | 0x7f800000 | (m << 13);
    } else {
      u32[0] = s | ((e - 15 + 127) << 23) | (m << 13);
    }
    tbl[h] = f32[0];
  }
  return tbl;
})();

function decodeBase64Bytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function decodeF16B64(b64) {
  const bytes = decodeBase64Bytes(b64);
  const n = bytes.byteLength >> 1;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const lo = bytes[i * 2];
    const hi = bytes[i * 2 + 1];
    out[i] = F16_TO_F32[lo | (hi << 8)];
  }
  return out;
}

function decodeF32B64(b64) {
  const bytes = decodeBase64Bytes(b64);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 2).slice();
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function linspace(lo, hi, count) {
  if (count <= 1) return [0.0];
  const out = new Array(count);
  for (let i = 0; i < count; i++) {
    out[i] = lo + ((hi - lo) * i) / (count - 1);
  }
  return out;
}

function hashText(text, seed) {
  let h = seed >>> 0;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

function genomeFingerprint(genome) {
  if (!genome) return "0000000000000000";
  if (genome.kind === "populora" && typeof genome.hash === "string" && genome.hash.length >= 16) {
    return genome.hash.slice(0, 16);
  }
  const w = Array.isArray(genome.weights) ? genome.weights : [];
  const b = Array.isArray(genome.biases) ? genome.biases : [];
  const a = Array.isArray(genome.activations) ? genome.activations : [];
  const text = w.join(",") + "|" + b.join(",") + "|" + a.join(",");
  return (
    hashText(text, 0x811c9dc5).toString(16).padStart(8, "0") +
    hashText(text, 0x9e3779b9).toString(16).padStart(8, "0")
  );
}

function applyActivation(z, code) {
  switch (code) {
    case 0: return Math.tanh(z);
    case 1: return Math.sin(z * 1.5);
    case 2: return Math.exp(-z * z);
    case 3: return Math.abs(z) - 0.5;
    default: return z > 0 ? z : 0.1 * z;
  }
}

function evalCppnPoint(genome, x0, x1, x2, x3, h1, h2, h3, outRgb) {
  const w = genome.weights;
  const b = genome.biases;
  const acts = genome.activations;

  for (let j = 0; j < 12; j++) {
    const wo = j * 4;
    const z = w[wo] * x0 + w[wo + 1] * x1 + w[wo + 2] * x2 + w[wo + 3] * x3 + b[j];
    h1[j] = applyActivation(z, acts[j]);
  }
  for (let j = 0; j < 12; j++) {
    const wo = 48 + j * 12;
    let z = b[12 + j];
    for (let i = 0; i < 12; i++) z += w[wo + i] * h1[i];
    h2[j] = applyActivation(z, acts[12 + j]);
  }
  for (let j = 0; j < 12; j++) {
    const wo = 192 + j * 12;
    let z = b[24 + j];
    for (let i = 0; i < 12; i++) z += w[wo + i] * h2[i];
    h3[j] = applyActivation(z, acts[24 + j]);
  }
  for (let c = 0; c < 3; c++) {
    const wo = 336 + c * 12;
    let z = b[36 + c];
    for (let i = 0; i < 12; i++) z += w[wo + i] * h3[i];
    outRgb[c] = 0.5 + 0.5 * Math.tanh(z);
  }
}

const substrateSpecCache = new Map();
function getSubstrateSpec(dims, layerCoords) {
  const key = dims.join(",") + "|" + JSON.stringify(layerCoords || []);
  if (substrateSpecCache.has(key)) return substrateSpecCache.get(key);
  const numLayers = dims.length - 1;
  const zs = linspace(-1.0, 1.0, numLayers + 1);
  const layerXs = dims.map((d, idx) =>
    Array.isArray(layerCoords) && Array.isArray(layerCoords[idx]) && layerCoords[idx].length === d
      ? layerCoords[idx]
      : linspace(-1.0, 1.0, d)
  );
  const queries = [];
  const layers = [];
  for (let l = 0; l < numLayers; l++) {
    const dIn = dims[l];
    const dOut = dims[l + 1];
    const zMid = 0.5 * (zs[l] + zs[l + 1]);
    const wIdx = new Int32Array(dIn * dOut);
    for (let i = 0; i < dIn; i++) {
      for (let j = 0; j < dOut; j++) {
        wIdx[i * dOut + j] = queries.length;
        queries.push([layerXs[l][i], layerXs[l + 1][j], zMid, 1.0]);
      }
    }
    const bIdx = new Int32Array(dOut);
    for (let j = 0; j < dOut; j++) {
      bIdx[j] = queries.length;
      queries.push([0.0, layerXs[l + 1][j], zs[l + 1], -1.0]);
    }
    layers.push({ dIn, dOut, wIdx, bIdx });
  }
  const entry = { dims, queries, layers };
  substrateSpecCache.set(key, entry);
  return entry;
}

const decodedCache = new Map();
function decodeHyperNeat(genome, dims, layerCoords) {
  const fp = genomeFingerprint(genome);
  const key = `${dims.join(",")}:${JSON.stringify(layerCoords || [])}:${fp}`;
  if (decodedCache.has(key)) return decodedCache.get(key);

  const spec = getSubstrateSpec(dims, layerCoords);
  const h1 = new Float32Array(12);
  const h2 = new Float32Array(12);
  const h3 = new Float32Array(12);
  const outRgb = new Float32Array(3);
  const raw = new Float64Array(spec.queries.length);

  for (let q = 0; q < spec.queries.length; q++) {
    const pt = spec.queries[q];
    evalCppnPoint(genome, pt[0], pt[1], pt[2], pt[3], h1, h2, h3, outRgb);
    raw[q] = outRgb[0] * 2.0 - 1.0;
  }

  const decoded = spec.layers.map((layer) => {
    const wLen = layer.dIn * layer.dOut;
    const w = new Float64Array(wLen);
    let sum = 0.0;
    for (let k = 0; k < wLen; k++) {
      const val = raw[layer.wIdx[k]];
      w[k] = val;
      sum += val;
    }
    const mean = sum / wLen;
    let varSum = 0.0;
    for (let k = 0; k < wLen; k++) {
      const diff = w[k] - mean;
      varSum += diff * diff;
    }
    const std = Math.sqrt(varSum / wLen);
    const scale = Math.sqrt(1.0 / layer.dIn) / (std + 1e-6);
    for (let k = 0; k < wLen; k++) w[k] = (w[k] - mean) * scale;
    const b = new Float64Array(layer.dOut);
    for (let j = 0; j < layer.dOut; j++) b[j] = raw[layer.bIdx[j]] * 0.1;
    return { dIn: layer.dIn, dOut: layer.dOut, w, b };
  });

  if (decodedCache.size > 1000) decodedCache.clear();
  decodedCache.set(key, decoded);
  return decoded;
}

const _polBufA = new Float64Array(64);
const _polBufB = new Float64Array(64);
const _polOut = { action: _polBufA, activations: null };

function forwardPolicy(decoded, obs, recordActs = false) {
  let h = obs;
  if (!recordActs) {
    for (let l = 0; l < decoded.length; l++) {
      const layer = decoded[l];
      const next = l % 2 === 0 ? _polBufA : _polBufB;
      for (let j = 0; j < layer.dOut; j++) {
        let z = layer.b[j];
        const w = layer.w;
        for (let i = 0; i < layer.dIn; i++) {
          z += h[i] * w[i * layer.dOut + j];
        }
        next[j] = Math.tanh(z);
      }
      h = next;
    }
    _polOut.action = h;
    _polOut.activations = null;
    return _polOut;
  }
  const acts = [Array.from(obs, (v) => Math.round(v * 1000) / 1000)];
  for (let l = 0; l < decoded.length; l++) {
    const layer = decoded[l];
    const next = new Float64Array(layer.dOut);
    for (let j = 0; j < layer.dOut; j++) {
      let z = layer.b[j];
      for (let i = 0; i < layer.dIn; i++) {
        z += h[i] * layer.w[i * layer.dOut + j];
      }
      next[j] = Math.tanh(z);
    }
    h = next;
    acts.push(Array.from(h, (v) => Math.round(v * 1000) / 1000));
  }
  return { action: h, activations: acts };
}

// --- PopuLoRA Policy Runtime (GRUMLPPolicy & MLPPolicy) ---

async function ensurePopuLoraBaseModel(taskId, genome, suppliedBaseModel = null) {
  const baseHash =
    (genome && (genome.base_hash || genome.baseHash))
    || (suppliedBaseModel && (suppliedBaseModel.base_hash || suppliedBaseModel.baseHash))
    || "default";
  const cacheKey = `${taskId}:${baseHash}`;
  if (baseModelCache.has(cacheKey)) return baseModelCache.get(cacheKey);

  let raw = suppliedBaseModel;
  if (!raw || !raw.tensors) {
    const qs = new URLSearchParams({ task: taskId, seed: String(genome?.base_seed || genome?.baseSeed || 1) });
    const reqHash = genome?.base_hash || genome?.baseHash;
    if (reqHash) qs.set("hash", reqHash);
    const url = new URL(`${basePath}/api/sim/base-model/?${qs.toString()}`, self.location.href).href;
    const resp = await fetch(url, { cache: "force-cache" });
    if (!resp.ok) throw new Error(`Failed to fetch PopuLoRA base model (${resp.status})`);
    raw = await resp.json();
  }

  const decodedTensors = {};
  for (const [tname, entry] of Object.entries(raw.tensors || {})) {
    decodedTensors[tname] = entry.dtype === "f32" ? decodeF32B64(entry.b64) : decodeF16B64(entry.b64);
  }
  const obsDim = Number(raw.obs_dim ?? raw.obsDim);
  const actionDim = Number(raw.action_dim ?? raw.actionDim);
  if (!Number.isFinite(obsDim) || obsDim <= 0 || !Number.isFinite(actionDim) || actionDim <= 0) {
    throw new Error(`Invalid PopuLoRA base model dimensions (obs_dim=${obsDim}, action_dim=${actionDim})`);
  }
  const parsed = {
    arch: raw.arch || "gru_mlp",
    obs_dim: obsDim,
    action_dim: actionDim,
    hidden_dim: Number(raw.hidden_dim ?? raw.hiddenDim ?? 128),
    head_dim: Number(raw.head_dim ?? raw.headDim ?? 128),
    low_rank: Number(raw.low_rank ?? raw.lowRank ?? 8),
    lora_targets: raw.lora_targets || raw.loraTargets || [],
    tensors: decodedTensors,
  };
  baseModelCache.set(cacheKey, parsed);
  return parsed;
}

const mergedPopuLoraCache = new Map();

function compilePopuLoraIndividual(baseModel, genome) {
  const fp = (genome && genome.hash) || genomeFingerprint(genome);
  const cacheKey = `${baseModel.arch}:${fp}`;
  if (mergedPopuLoraCache.has(cacheKey)) return mergedPopuLoraCache.get(cacheKey);

  const r = Number(genome.low_rank ?? genome.lowRank ?? baseModel.low_rank ?? 8);
  const adapters = genome.adapters || {};

  // Merge W_eff[o, i] = W_base[o, i] + sum_k(up[o, k] * down[i, k]) (row-major [outDim, inDim])
  function mergeLinear(path, outDim, inDim, hasBias = true) {
    const baseW = baseModel.tensors[`${path}.weight`];
    const baseB = hasBias ? baseModel.tensors[`${path}.bias`] : null;
    const w = baseW ? baseW.slice() : new Float32Array(outDim * inDim);
    const key = path.replace(/\./g, "_");
    const ad = adapters[key];
    if (ad && ad.down && ad.up) {
      const down = decodeF16B64(ad.down); // [inDim, r]
      const up = decodeF16B64(ad.up);     // [outDim, r]
      for (let o = 0; o < outDim; o++) {
        const uOff = o * r;
        const wRow = o * inDim;
        for (let i = 0; i < inDim; i++) {
          const dOff = i * r;
          let sum = 0.0;
          for (let k = 0; k < r; k++) sum += up[uOff + k] * down[dOff + k];
          w[wRow + i] += sum;
        }
      }
    }
    return { w, b: baseB, in_dim: inDim, out_dim: outDim };
  }

  let compiled;
  if (baseModel.arch === "weight_only_mlp") {
    const H = baseModel.hidden_dim;
    const D = baseModel.head_dim;
    compiled = {
      arch: "weight_only_mlp",
      obs_dim: baseModel.obs_dim,
      action_dim: baseModel.action_dim,
      l0: mergeLinear("l0", H, baseModel.obs_dim + 1, false),
      l1: mergeLinear("l1", D, H + 1, false),
      head: mergeLinear("head", baseModel.action_dim, D + 1, false),
    };
  } else if (baseModel.arch === "mlp") {
    const H = baseModel.hidden_dim;
    const D = baseModel.head_dim;
    compiled = {
      arch: "mlp",
      obs_dim: baseModel.obs_dim,
      action_dim: baseModel.action_dim,
      enc0: mergeLinear("encoder.0", H, baseModel.obs_dim, true),
      enc2: mergeLinear("encoder.2", D, H, true),
      head: mergeLinear("head", baseModel.action_dim, D, true),
    };
  } else {
    const H = baseModel.hidden_dim;
    const D = baseModel.head_dim;
    const initMem = baseModel.tensors["init_memory"] || new Float32Array(H);
    compiled = {
      arch: "gru_mlp",
      obs_dim: baseModel.obs_dim,
      action_dim: baseModel.action_dim,
      hidden_dim: H,
      head_dim: D,
      initMem,
      enc0: mergeLinear("net.encoder.0", H, baseModel.obs_dim, true),
      enc2: mergeLinear("net.encoder.2", H, H, true),
      ih: mergeLinear("net.cell.linear_ih", 3 * H, H, true),
      hh: mergeLinear("net.cell.linear_hh", 3 * H, H, false),
      dec0: mergeLinear("net.decoder.0", D, H, true),
      dec2: mergeLinear("net.decoder.2", baseModel.action_dim, D, true),
    };
  }

  if (mergedPopuLoraCache.size > 256) mergedPopuLoraCache.clear();
  mergedPopuLoraCache.set(cacheKey, compiled);
  return compiled;
}

function matVecLinear(layer, x, out, applyTanh) {
  const { w, b, in_dim: inDim, out_dim: outDim } = layer;
  for (let o = 0; o < outDim; o++) {
    let acc = b ? b[o] : 0.0;
    const row = o * inDim;
    for (let i = 0; i < inDim; i++) acc += w[row + i] * x[i];
    out[o] = applyTanh ? Math.tanh(acc) : acc;
  }
}

function matVecLinearAct(layer, x, out, actKind) {
  const { w, b, in_dim: inDim, out_dim: outDim } = layer;
  for (let o = 0; o < outDim; o++) {
    let acc = b ? b[o] : 0.0;
    const row = o * inDim;
    for (let i = 0; i < inDim; i++) acc += w[row + i] * x[i];
    if (actKind === "relu") out[o] = acc > 0.0 ? acc : 0.0;
    else if (actKind === "tanh") out[o] = Math.tanh(acc);
    else out[o] = acc;
  }
}

function createPopuLoraRunner(compiled) {
  if (compiled.arch === "weight_only_mlp") {
    const x0 = new Float32Array(compiled.obs_dim + 1);
    const h0 = new Float32Array(compiled.l0.out_dim);
    const x1 = new Float32Array(compiled.l0.out_dim + 1);
    const h1 = new Float32Array(compiled.l1.out_dim);
    const x2 = new Float32Array(compiled.l1.out_dim + 1);
    const act = new Float32Array(compiled.action_dim);
    x0[compiled.obs_dim] = 1.0;
    x1[compiled.l0.out_dim] = 1.0;
    x2[compiled.l1.out_dim] = 1.0;
    return {
      reset() {},
      step(obs, recordActs = false) {
        for (let i = 0; i < compiled.obs_dim; i++) x0[i] = obs[i];
        matVecLinearAct(compiled.l0, x0, h0, "relu");
        x1.set(h0, 0);
        matVecLinearAct(compiled.l1, x1, h1, "relu");
        x2.set(h1, 0);
        matVecLinearAct(compiled.head, x2, act, "tanh");
        const acts = recordActs
          ? [
              Array.from(obs.slice ? obs.slice(0, 16) : Array.from(obs).slice(0, 16), (v) => Math.round(v * 1000) / 1000),
              Array.from(h0.subarray(0, 16), (v) => Math.round(v * 1000) / 1000),
              Array.from(h1.subarray(0, 16), (v) => Math.round(v * 1000) / 1000),
              Array.from(act, (v) => Math.round(v * 1000) / 1000),
            ]
          : null;
        return { action: act, activations: acts };
      },
    };
  }
  if (compiled.arch === "mlp") {
    const h0 = new Float32Array(compiled.enc0.out_dim);
    const h1 = new Float32Array(compiled.enc2.out_dim);
    const act = new Float32Array(compiled.action_dim);
    return {
      reset() {},
      step(obs, recordActs = false) {
        matVecLinear(compiled.enc0, obs, h0, true);
        matVecLinear(compiled.enc2, h0, h1, true);
        matVecLinear(compiled.head, h1, act, true);
        const acts = recordActs
          ? [
              Array.from(obs.subarray(0, 16), (v) => Math.round(v * 1000) / 1000),
              Array.from(h0.subarray(0, 16), (v) => Math.round(v * 1000) / 1000),
              Array.from(h1.subarray(0, 16), (v) => Math.round(v * 1000) / 1000),
              Array.from(act, (v) => Math.round(v * 1000) / 1000),
            ]
          : null;
        return { action: act, activations: acts };
      },
    };
  }

  const H = compiled.hidden_dim;
  const mem = new Float32Array(H);
  const e0 = new Float32Array(H);
  const e1 = new Float32Array(H);
  const gIh = new Float32Array(3 * H);
  const gHh = new Float32Array(3 * H);
  const d0 = new Float32Array(compiled.head_dim);
  const act = new Float32Array(compiled.action_dim);

  return {
    reset() {
      mem.set(compiled.initMem.subarray(0, H));
    },
    step(obs, recordActs = false) {
      matVecLinear(compiled.enc0, obs, e0, true);
      matVecLinear(compiled.enc2, e0, e1, true);
      matVecLinear(compiled.ih, e1, gIh, false);
      matVecLinear(compiled.hh, mem, gHh, false);
      for (let j = 0; j < H; j++) {
        const rGate = 1.0 / (1.0 + Math.exp(-(gIh[j] + gHh[j])));
        const zGate = 1.0 / (1.0 + Math.exp(-(gIh[H + j] + gHh[H + j])));
        const nCand = Math.tanh(gIh[2 * H + j] + rGate * gHh[2 * H + j]);
        mem[j] = (1.0 - zGate) * nCand + zGate * mem[j];
      }
      matVecLinear(compiled.dec0, mem, d0, true);
      matVecLinear(compiled.dec2, d0, act, true); // SquashedGaussian mean = tanh(raw_mean)
      const acts = recordActs
        ? [
            Array.from(obs.subarray(0, 16), (v) => Math.round(v * 1000) / 1000),
            Array.from(e1.subarray(0, 16), (v) => Math.round(v * 1000) / 1000),
            Array.from(mem.subarray(0, 16), (v) => Math.round(v * 1000) / 1000),
            Array.from(d0.subarray(0, 16), (v) => Math.round(v * 1000) / 1000),
            Array.from(act, (v) => Math.round(v * 1000) / 1000),
          ]
        : null;
      return { action: act, activations: acts };
    },
  };
}

// --- Shared MuJoCo WASM + DETVFS1 Scene Loader ---

async function ensureMujocoWasm() {
  if (!mujocoWasmUrl) throw new Error("MuJoCo WASM URL not configured.");
  if (!mujocoPromise) {
    const absWasmUrl = new URL(mujocoWasmUrl, self.location.href).href;
    mujocoPromise = (async () => {
      const mod = await import(absWasmUrl);
      const loadFn = mod.default || mod;
      return await loadFn();
    })();
  }
  return mujocoPromise;
}

function mountDetVfsArchive(mujoco, sceneId, arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  if (bytes.byteLength < 12) throw new Error("Truncated DETVFS1 archive.");
  const magic = String.fromCharCode(
    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6]
  );
  if (magic !== "DETVFS1") throw new Error("Invalid DETVFS1 header.");

  const view = new DataView(arrayBuffer);
  const fileCount = view.getUint32(8, true);
  const textDecoder = new TextDecoder("utf-8");
  const rootDir = `/scenes/${sceneId}`;
  mujoco.FS.mkdirTree(rootDir);

  let off = 12;
  for (let i = 0; i < fileCount; i++) {
    const plen = view.getUint16(off, true);
    off += 2;
    const relPath = textDecoder.decode(bytes.subarray(off, off + plen));
    off += plen;
    const dlen = view.getUint32(off, true);
    off += 4;
    const fullPath = `${rootDir}/${relPath}`;
    const slash = fullPath.lastIndexOf("/");
    if (slash > 0) {
      mujoco.FS.mkdirTree(fullPath.slice(0, slash));
    }
    mujoco.FS.writeFile(fullPath, bytes.subarray(off, off + dlen));
    off += dlen;
  }
  return rootDir;
}

async function ensureMujocoScene(sceneId, manifest, suppliedPhysicsBuf = null) {
  if (mountedScenes.has(sceneId)) {
    const cached = mountedScenes.get(sceneId);
    return { mujoco: await ensureMujocoWasm(), ...cached };
  }
  if (!manifest || !manifest.physics_asset || !manifest.entry_xml) {
    throw new Error(`Missing scene manifest for '${sceneId}'.`);
  }
  if (suppliedPhysicsBuf instanceof ArrayBuffer && suppliedPhysicsBuf.byteLength > 0) {
    physicsBufCache.set(sceneId, suppliedPhysicsBuf);
  }
  let bufPromise = physicsBufCache.has(sceneId)
    ? Promise.resolve(physicsBufCache.get(sceneId))
    : (async () => {
        const ver = (manifest.physics_sha256 || "").slice(0, 12);
        const relUrl = `${basePath}/assets/${manifest.physics_asset}${ver ? `?v=${ver}` : ""}`;
        const absUrl = new URL(relUrl, self.location.href).href;
        const resp = await fetch(absUrl, { cache: "force-cache" });
        if (!resp.ok) throw new Error(`Failed to load ${sceneId} physics pack (${resp.status})`);
        const ab = await resp.arrayBuffer();
        physicsBufCache.set(sceneId, ab);
        return ab;
      })();

  const [mujoco, physBuf] = await Promise.all([ensureMujocoWasm(), bufPromise]);
  if (!mountedScenes.has(sceneId)) {
    const rootDir = mountDetVfsArchive(mujoco, sceneId, physBuf);
    const model = mujoco.MjModel.from_xml_path(`${rootDir}/${manifest.entry_xml}`);
    const data = new mujoco.MjData(model);
    mountedScenes.set(sceneId, { model, data, initialized: false, meta: {} });
  }
  return { mujoco, ...mountedScenes.get(sceneId) };
}

function captureSceneBodyPoses(model, data) {
  const nb = model.nbody - 1;
  const row = new Array(nb * 7);
  const xp = data.xpos;
  const xq = data.xquat;
  for (let b = 1; b < model.nbody; b++) {
    const outO = (b - 1) * 7;
    const pO = b * 3;
    const qO = b * 4;
    row[outO] = Math.round(xp[pO] * 10000) / 10000;
    row[outO + 1] = Math.round(xp[pO + 1] * 10000) / 10000;
    row[outO + 2] = Math.round(xp[pO + 2] * 10000) / 10000;
    row[outO + 3] = Math.round(xq[qO] * 10000) / 10000;
    row[outO + 4] = Math.round(xq[qO + 1] * 10000) / 10000;
    row[outO + 5] = Math.round(xq[qO + 2] * 10000) / 10000;
    row[outO + 6] = Math.round(xq[qO + 3] * 10000) / 10000;
  }
  return row;
}

function readActuatorCtrlBounds(model, fallbackAbs = 10.0) {
  const ctrlMin = new Float64Array(model.nu);
  const ctrlMax = new Float64Array(model.nu);
  for (let a = 0; a < model.nu; a++) {
    const limited = model.actuator_ctrllimited ? model.actuator_ctrllimited[a] : 1;
    ctrlMin[a] = limited ? model.actuator_ctrlrange[a * 2] : -fallbackAbs;
    ctrlMax[a] = limited ? model.actuator_ctrlrange[a * 2 + 1] : fallbackAbs;
  }
  return { ctrlMin, ctrlMax };
}

function isMjStateFinite(model, qpos, qvel) {
  for (let k = 0; k < model.nq; k++) {
    if (!Number.isFinite(qpos[k])) return false;
  }
  if (qvel) {
    for (let k = 0; k < model.nv; k++) {
      if (!Number.isFinite(qvel[k])) return false;
    }
  }
  return true;
}

function tailPercentile95(samples, skipHead = 20) {
  if (!samples.length) return 0.0;
  const tail = samples.slice(Math.min(skipHead, Math.max(0, samples.length - 1)));
  if (!tail.length) return 0.0;
  const sorted = tail.slice().sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(0.95 * (sorted.length - 1)))];
}

function subsampleTrajectory(rows, numSub = 16) {
  const out = [];
  if (!rows.length) return out;
  for (let k = 0; k < numSub; k++) {
    const rIdx = Math.round((k * (rows.length - 1)) / Math.max(1, numSub - 1));
    out.push(rows[rIdx].map((v) => Math.round(v * 10000) / 10000));
  }
  return out;
}

// --- Microduck 14-DOF Bipedal Gait Environment (exact match to populora/train_microduck.py) ---

function ensureMicroduckSceneMeta(mujoco, entry, envParams) {
  if (entry.initialized) return entry.meta;
  const { model } = entry;
  const p = envParams || {};
  const physicsDt = p.physics_dt || 0.004;
  const kpMult = p.kp_mult ?? 3.0;
  const kv = p.kv ?? 0.1;

  model.opt.timestep = physicsDt;
  for (let a = 0; a < model.nu; a++) {
    model.actuator_gainprm[a * 10] *= kpMult;
    model.actuator_biasprm[a * 10 + 1] = -model.actuator_gainprm[a * 10];
    model.actuator_biasprm[a * 10 + 2] = -kv;
  }

  const objKey = mujoco.mjtObj.mjOBJ_KEY.value ?? mujoco.mjtObj.mjOBJ_KEY;
  const objGeom = mujoco.mjtObj.mjOBJ_GEOM.value ?? mujoco.mjtObj.mjOBJ_GEOM;
  const objSite = mujoco.mjtObj.mjOBJ_SITE.value ?? mujoco.mjtObj.mjOBJ_SITE;
  const objBody = mujoco.mjtObj.mjOBJ_BODY.value ?? mujoco.mjtObj.mjOBJ_BODY;

  const standIdx = Math.max(0, mujoco.mj_name2id(model, objKey, "STAND"));
  const homeQpos = Float64Array.from(model.key_qpos.subarray(standIdx * model.nq, (standIdx + 1) * model.nq));
  const homeCtrl = Float64Array.from(model.key_ctrl.subarray(standIdx * model.nu, (standIdx + 1) * model.nu));

  const floorGeom = mujoco.mj_name2id(model, objGeom, "floor");
  const leftFootGeom = mujoco.mj_name2id(model, objGeom, "left_foot_collision");
  const rightFootGeom = mujoco.mj_name2id(model, objGeom, "right_foot_collision");
  const leftFootSite = mujoco.mj_name2id(model, objSite, "left_foot");
  const rightFootSite = mujoco.mj_name2id(model, objSite, "right_foot");

  const footBodies = new Set([model.geom_bodyid[leftFootGeom], model.geom_bodyid[rightFootGeom]]);
  const nonFootBodies = [];
  for (let b = 1; b < model.nbody; b++) {
    if (!footBodies.has(b)) nonFootBodies.push(b);
  }

  entry.meta = {
    objBody,
    homeQpos,
    homeCtrl,
    floorGeom,
    footGeoms: [leftFootGeom, rightFootGeom],
    footSites: [leftFootSite, rightFootSite],
    nonFootBodies,
    velLocalBuf: new Float64Array(6),
    velWorldBuf: new Float64Array(6),
    obsBuf: new Float32Array(55),
  };
  entry.initialized = true;
  return entry.meta;
}

function matchWorkerHyperparamOption(hSpec, rawVal) {
  const norm = String(rawVal ?? "").trim().toLowerCase().replace(/-/g, "_");
  if (!norm || !Array.isArray(hSpec.options)) return "";
  for (const opt of hSpec.options) {
    const val = String(opt.value || "").trim().toLowerCase();
    const aliases = Array.isArray(opt.aliases) ? opt.aliases.map((a) => String(a).trim().toLowerCase()) : [];
    if (norm === val || aliases.includes(norm)) return val;
  }
  return "";
}

function resolveSceneTaskConfig(taskId, envParamsOverride = null) {
  const spec = taskSpecs[taskId] || {};
  let merged = spec.env_params ? { ...spec.env_params } : {};
  const ov = envParamsOverride && typeof envParamsOverride === "object" ? envParamsOverride : {};
  const hSpecs = Array.isArray(spec.hyperparam_specs) ? spec.hyperparam_specs : [];
  let selectedVariant = "";
  let variantManifest = null;

  if (spec.variants && typeof spec.variants === "object") {
    for (const h of hSpecs) {
      const keys = [h.key, ...(Array.isArray(h.aliases) ? h.aliases : [])].filter(Boolean);
      const raw = keys.map((k) => ov[k]).find((v) => v !== undefined && v !== null && v !== "")
        ?? merged[h.key]
        ?? h.default;
      const matched = matchWorkerHyperparamOption(h, raw);
      if (matched && spec.variants[matched]) {
        selectedVariant = matched;
        const vObj = { ...spec.variants[matched] };
        variantManifest = vObj.scene_manifest || null;
        delete vObj.scene_manifest;
        merged = { ...merged, ...vObj };
        break;
      }
    }
  }

  for (const h of hSpecs) {
    const keys = [h.key, ...(Array.isArray(h.aliases) ? h.aliases : [])].filter(Boolean);
    const raw = keys.map((k) => ov[k]).find((v) => v !== undefined && v !== null && v !== "");
    if (raw === undefined) continue;
    const hType = String(h.type || "select").toLowerCase();
    if (hType === "select") {
      const matched = matchWorkerHyperparamOption(h, raw);
      if (matched) merged[h.key] = matched;
    } else if (hType === "float" || hType === "number" || hType === "int") {
      let num = Number(raw);
      if (Number.isFinite(num)) {
        if (typeof h.min === "number") num = Math.max(h.min, num);
        if (typeof h.max === "number") num = Math.min(h.max, num);
        merged[h.key] = hType === "int" ? Math.round(num) : Math.round(num * 10000) / 10000;
      }
    } else if (hType === "bool") {
      merged[h.key] = raw === true || raw === "true" || raw === "1";
    }
  }

  const sceneId = String(merged.scene_id || spec.scene_id || taskId);
  const manifest = variantManifest || spec.scene_manifest || null;
  return { variant: selectedVariant, scene_id: sceneId, manifest, env_params: merged };
}

async function microduckRollout(
  genome,
  seed = 0,
  maxSteps = 400,
  nudge = null,
  includeActs = false,
  capturePoses = true,
  poseStride = 2,
  suppliedBaseModel = null,
  suppliedPhysicsBuf = null,
  envParamsOverride = null
) {
  const { scene_id: sceneId, manifest, env_params: envParams } = resolveSceneTaskConfig("microduck", envParamsOverride);

  const [{ mujoco, model, data }, baseModel] = await Promise.all([
    ensureMujocoScene(sceneId, manifest, suppliedPhysicsBuf),
    ensurePopuLoraBaseModel("microduck", genome, suppliedBaseModel),
  ]);
  const sceneEntry = mountedScenes.get(sceneId);
  const meta = ensureMicroduckSceneMeta(mujoco, sceneEntry, envParams);
  const compiledPolicy = compilePopuLoraIndividual(baseModel, genome);
  const policyRunner = createPopuLoraRunner(compiledPolicy);
  policyRunner.reset();

  const nSubsteps = envParams.n_substeps || 5;
  const ctrlDt = envParams.ctrl_dt || 0.02;
  const actionScale = envParams.action_scale ?? 0.5;
  const cmd = envParams.command || [0.15, 0.0];
  const cmdSpeed = Math.hypot(cmd[0], cmd[1]);
  const hasCmd = cmdSpeed > 0.01;
  const cmdDirX = hasCmd ? cmd[0] / cmdSpeed : 1.0;
  const cmdDirY = hasCmd ? cmd[1] / cmdSpeed : 0.0;

  const progressWeight = envParams.progress_weight ?? 350.0;
  const clearanceWeight = envParams.clearance_weight ?? 900.0;
  const clearanceTarget = envParams.clearance_target ?? 0.008;
  const footDragWeight = envParams.foot_drag_weight ?? 160.0;
  const altStepBonus = envParams.alt_step_bonus ?? 400.0;
  const swayWeight = envParams.sway_weight ?? 20.0;
  const bodyAngVelWeight = envParams.body_ang_vel_weight ?? 0.04;
  const actionRateWeight = envParams.action_rate_weight ?? 0.10;
  const fallPenalty = envParams.fall_penalty ?? 3500.0;
  const minTrunkZ = envParams.min_trunk_z ?? 0.04;
  const minUpright = envParams.min_uprightness ?? 0.35;
  const legJoints = envParams.leg_joints || [0, 1, 2, 3, 4, 9, 10, 11, 12, 13];

  mujoco.mj_resetData(model, data);
  let qpos = data.qpos;
  let qvel = data.qvel;
  let ctrl = data.ctrl;

  const rng = mulberry32(seed);
  qpos.set(meta.homeQpos);
  qpos[2] += rng() * 0.002;
  for (let j = 0; j < model.nu; j++) qpos[7 + j] += (rng() * 2.0 - 1.0) * 0.005;
  for (let v = 0; v < model.nv; v++) qvel[v] = (rng() * 2.0 - 1.0) * 0.01;
  ctrl.set(meta.homeCtrl);
  mujoco.mj_forward(model, data);
  if (qpos.byteLength === 0) {
    qpos = data.qpos;
    qvel = data.qvel;
    ctrl = data.ctrl;
  }

  if (nudge === "fwd") qvel[0] += 0.32;
  else if (nudge === "back") qvel[0] -= 0.28;
  else if (nudge === "lat") qvel[1] += 0.28;

  const startX = qpos[0];
  const startY = qpos[1];
  const lastAction = new Float32Array(model.nu);
  const prevFootXY = [
    [data.site_xpos[meta.footSites[0] * 3], data.site_xpos[meta.footSites[0] * 3 + 1]],
    [data.site_xpos[meta.footSites[1] * 3], data.site_xpos[meta.footSites[1] * 3 + 1]],
  ];
  const maxFootProj = [
    (prevFootXY[0][0] - startX) * cmdDirX + (prevFootXY[0][1] - startY) * cmdDirY,
    (prevFootXY[1][0] - startX) * cmdDirX + (prevFootXY[1][1] - startY) * cmdDirY,
  ];
  const tdProj = [maxFootProj[0], maxFootProj[1]];
  const liftedAdv = [0.0, 0.0];
  const swingSteps = [0, 0];
  const swingPeak = [0.0, 0.0];
  let lastFoot = -1;
  let maxDeff = 0.0;
  let ssupSteps = 0;
  let totalReturn = 0.0;
  let minRoll = Infinity;
  let maxRoll = -Infinity;
  const footClearsMm = [];
  const rawGaitRows = [];
  let dispIntX = 0.0;
  let dispIntY = 0.0;

  const fillObs = () => {
    const obs = meta.obsBuf;
    const xmOff = 1 * 9; // TRUNK_BODY = 1
    obs[0] = -data.xmat[xmOff + 6];
    obs[1] = -data.xmat[xmOff + 7];
    obs[2] = -data.xmat[xmOff + 8];

    mujoco.mj_objectVelocity(model, data, meta.objBody, 1, meta.velLocalBuf, 1);
    mujoco.mj_objectVelocity(model, data, meta.objBody, 1, meta.velWorldBuf, 0);
    if (qpos.byteLength === 0) {
      qpos = data.qpos;
      qvel = data.qvel;
      ctrl = data.ctrl;
    }

    obs[3] = meta.velLocalBuf[3];
    obs[4] = meta.velLocalBuf[4];
    obs[5] = meta.velLocalBuf[5];
    obs[6] = meta.velLocalBuf[0];
    obs[7] = meta.velLocalBuf[1];
    obs[8] = meta.velLocalBuf[2];
    obs[9] = meta.velWorldBuf[3];
    obs[10] = meta.velWorldBuf[4];

    for (let j = 0; j < 14; j++) obs[11 + j] = qpos[7 + j] - meta.homeQpos[7 + j];
    for (let j = 0; j < 14; j++) obs[25 + j] = qvel[6 + j];
    for (let j = 0; j < 14; j++) obs[39 + j] = lastAction[j];
    obs[53] = cmd[0];
    obs[54] = cmd[1];
    return obs;
  };

  const bodyPoses = capturePoses ? [captureSceneBodyPoses(model, data)] : null;
  const vxTrace = capturePoses ? [0.0] : null;
  const stepTrace = capturePoses ? [[0.0, 0.0, 0.0, 0.0, 0.0]] : null;
  const stepActs = includeActs ? [] : null;

  let steps = 0;
  let terminated = false;

  for (let s = 0; s < maxSteps; s++) {
    const obs = fillObs();
    // Record 20-D gait trajectory row matching train_microduck.py:extract_gait_trajectories
    dispIntX += obs[9] * ctrlDt;
    dispIntY += obs[10] * ctrlDt;
    const gaitRow = new Array(20);
    gaitRow[0] = dispIntX;
    gaitRow[1] = dispIntY;
    gaitRow[2] = obs[9];
    gaitRow[3] = obs[10];
    gaitRow[4] = obs[0];
    gaitRow[5] = obs[1];
    gaitRow[6] = obs[2];
    gaitRow[7] = obs[6];
    gaitRow[8] = obs[7];
    gaitRow[9] = obs[8];
    for (let lj = 0; lj < 10; lj++) gaitRow[10 + lj] = obs[11 + legJoints[lj]];
    rawGaitRows.push(gaitRow);

    const { action, activations } = policyRunner.step(obs, includeActs);
    if (includeActs && (s % Math.max(1, poseStride) === 0)) stepActs.push(activations);

    let actRateAcc = 0.0;
    for (let j = 0; j < 14; j++) {
      const aRaw = Number.isFinite(action[j]) ? action[j] : 0.0;
      const aEff = Math.max(-1.0, Math.min(1.0, aRaw));
      const diff = aEff - lastAction[j];
      actRateAcc += diff * diff;
      ctrl[j] = meta.homeCtrl[j] + actionScale * aEff;
      lastAction[j] = aEff;
    }
    const actionRate = actRateAcc / 14.0;

    for (let sub = 0; sub < nSubsteps; sub++) mujoco.mj_step(model, data);
    if (qpos.byteLength === 0) {
      qpos = data.qpos;
      qvel = data.qvel;
      ctrl = data.ctrl;
    }
    steps++;

    if (!isMjStateFinite(model, qpos, qvel)) {
      totalReturn += -10.0 - fallPenalty;
      terminated = true;
      break;
    }

    const xmOff = 9;
    const uprightness = data.xmat[xmOff + 8];
    const rollDeg = (Math.atan2(data.xmat[xmOff + 7], uprightness) * 180.0) / Math.PI;
    if (rollDeg < minRoll) minRoll = rollDeg;
    if (rollDeg > maxRoll) maxRoll = rollDeg;

    let minNonFootZ = Infinity;
    for (let bi = 0; bi < meta.nonFootBodies.length; bi++) {
      const bz = data.xpos[meta.nonFootBodies[bi] * 3 + 2];
      if (bz < minNonFootZ) minNonFootZ = bz;
    }
    const floorClip = Math.max(0.0, -minNonFootZ);
    terminated = uprightness < minUpright || qpos[2] < minTrunkZ || floorClip > 0.015;

    const cvelOff = 6; // TRUNK_BODY = 1 -> 1 * 6
    const vWx = data.cvel[cvelOff + 3];
    const vWy = data.cvel[cvelOff + 4];
    const dispX = qpos[0] - startX;
    const dispY = qpos[1] - startY;

    const vProj = hasCmd ? vWx * cmdDirX + vWy * cmdDirY : 0.0;
    const vLat = Math.hypot(vWx - vProj * cmdDirX, vWy - vProj * cmdDirY);
    const forwardDist = hasCmd ? dispX * cmdDirX + dispY * cmdDirY : 0.0;
    const latDrift = Math.hypot(dispX - forwardDist * cmdDirX, dispY - forwardDist * cmdDirY);
    const headingCos = hasCmd ? data.xmat[xmOff] * cmdDirX + data.xmat[xmOff + 3] * cmdDirY : 1.0;

    const dEff = Math.min(0.22, Math.max(0.0, forwardDist - 0.03 - 0.55 * latDrift));
    const frontierVel = Math.max(0.0, dEff - maxDeff) / ctrlDt;
    if (dEff > maxDeff) maxDeff = dEff;

    mujoco.mj_objectVelocity(model, data, meta.objBody, 1, meta.velLocalBuf, 1);
    const rollYawSq = meta.velLocalBuf[0] * meta.velLocalBuf[0] + meta.velLocalBuf[2] * meta.velLocalBuf[2];

    const poseGate = Math.exp(
      -12.0 * Math.max(0.0, 1.0 - uprightness) -
      10.0 * Math.max(0.0, 1.0 - headingCos) -
      55.0 * (latDrift * latDrift)
    );
    const swayGate = Math.exp(-10.0 * (vLat * vLat) - 0.06 * rollYawSq);

    let stepReward =
      poseGate * Math.sqrt(swayGate) * (progressWeight * frontierVel + 0.25 * progressWeight * Math.min(vProj, cmdSpeed)) -
      swayWeight * (vLat * vLat + 0.80 * latDrift * latDrift) -
      bodyAngVelWeight * rollYawSq -
      actionRateWeight * actionRate;

    let stepClearMm = 0.0;
    if (hasCmd && clearanceWeight > 0.0) {
      const c0 = mujoco.mj_geomDistance(model, data, meta.floorGeom, meta.footGeoms[0], 0.10, null);
      const c1 = mujoco.mj_geomDistance(model, data, meta.floorGeom, meta.footGeoms[1], 0.10, null);
      if (qpos.byteLength === 0) {
        qpos = data.qpos;
        qvel = data.qvel;
        ctrl = data.ctrl;
      }
      const c = [c0, c1];

      const fxy0 = [data.site_xpos[meta.footSites[0] * 3], data.site_xpos[meta.footSites[0] * 3 + 1]];
      const fxy1 = [data.site_xpos[meta.footSites[1] * 3], data.site_xpos[meta.footSites[1] * 3 + 1]];
      const fspd0 = Math.hypot(fxy0[0] - prevFootXY[0][0], fxy0[1] - prevFootXY[0][1]) / ctrlDt;
      const fspd1 = Math.hypot(fxy1[0] - prevFootXY[1][0], fxy1[1] - prevFootXY[1][1]) / ctrlDt;
      prevFootXY[0] = fxy0;
      prevFootXY[1] = fxy1;

      const fp0 = (fxy0[0] - startX) * cmdDirX + (fxy0[1] - startY) * cmdDirY - 0.50 * latDrift;
      const fp1 = (fxy1[0] - startX) * cmdDirX + (fxy1[1] - startY) * cmdDirY - 0.50 * latDrift;
      const fp = [fp0, fp1];
      const fvel0 = Math.max(0.0, fp0 - maxFootProj[0]) / ctrlDt;
      const fvel1 = Math.max(0.0, fp1 - maxFootProj[1]) / ctrlDt;
      if (fp0 > maxFootProj[0]) maxFootProj[0] = fp0;
      if (fp1 > maxFootProj[1]) maxFootProj[1] = fp1;

      const u0 = Math.max(0.0, Math.min(1.0, (c0 + 0.0003) / clearanceTarget));
      const u1 = Math.max(0.0, Math.min(1.0, (c1 + 0.0003) / clearanceTarget));
      const oppG0 = Math.max(0.0, Math.min(1.0, 1.0 - Math.max(0.0, c1 - 0.0005) / 0.0015));
      const oppG1 = Math.max(0.0, Math.min(1.0, 1.0 - Math.max(0.0, c0 - 0.0005) / 0.0015));
      const sw0 = u0 * oppG0 * fvel0;
      const sw1 = u1 * oppG1 * fvel1;

      const prevMin = Math.min(liftedAdv[0], liftedAdv[1]);
      liftedAdv[0] += sw0 * ctrlDt;
      liftedAdv[1] += sw1 * ctrlDt;
      const symVel = (Math.min(liftedAdv[0], liftedAdv[1]) - prevMin) / ctrlDt;

      const cScale = clearanceWeight / 900.0;
      const dragRamp = Math.max(0.0, Math.min(1.0, (forwardDist - 0.16) / 0.12));
      const dragTerm = (1.0 - u0) * (1.0 - u0) * fspd0 + (1.0 - u1) * (1.0 - u1) * fspd1;
      stepReward += poseGate * clearanceWeight * (sw0 + sw1 + 1.8 * symVel) - cScale * footDragWeight * dragRamp * dragTerm;

      for (let i = 0; i < 2; i++) {
        if (c[i] > 0.0010 && c[1 - i] <= 0.0008) {
          swingSteps[i] += 1;
          if (c[i] > swingPeak[i]) swingPeak[i] = c[i];
        } else if (c[i] <= 0.0006) {
          if (
            swingSteps[i] >= 2 &&
            swingPeak[i] >= 0.0015 &&
            fp[i] > Math.max(fp[1 - i] + 0.004, tdProj[i] + 0.008) &&
            lastFoot !== i
          ) {
            stepReward +=
              poseGate *
              cScale *
              altStepBonus *
              Math.min(swingSteps[i] / 5.0, 1.0) *
              Math.min(swingPeak[i] / clearanceTarget, 1.0);
            tdProj[i] = fp[i];
            lastFoot = i;
          }
          swingSteps[i] = 0;
          swingPeak[i] = 0.0;
        }
      }

      const oppGr0 = c1 <= 0.0008;
      const oppGr1 = c0 <= 0.0008;
      if ((c0 > 0.0010 && oppGr0) || (c1 > 0.0010 && oppGr1)) ssupSteps += 1;
      const cl0 = oppGr0 ? Math.max(0.0, c0) * 1000.0 : 0.0;
      const cl1 = oppGr1 ? Math.max(0.0, c1) * 1000.0 : 0.0;
      stepClearMm = Math.max(cl0, cl1);
      footClearsMm.push([cl0, cl1]);
    }

    if (terminated) stepReward -= fallPenalty;
    totalReturn += stepReward;

    if (capturePoses && (s % Math.max(1, poseStride) === 0 || terminated || s === maxSteps - 1)) {
      bodyPoses.push(captureSceneBodyPoses(model, data));
      vxTrace.push(Math.round(vProj * 1000) / 1000);
      stepTrace.push([
        Math.round(dispX * 1000) / 1000,
        Math.round(totalReturn * 10) / 10,
        Math.round(stepClearMm * 10) / 10,
        Math.round(vProj * 100) / 100,
        Math.round((100.0 * ssupSteps) / Math.max(1, steps)),
      ]);
    }

    if (terminated) break;
  }

  const dx = qpos[0] - startX;
  const absY = Math.abs(qpos[1] - startY);
  const rollPtp = minRoll <= maxRoll ? maxRoll - minRoll : 0.0;
  const ssupPct = (100.0 * ssupSteps) / Math.max(1, steps);

  const clearMm = tailPercentile95(footClearsMm.map((r) => Math.max(r[0], r[1])), 20);
  const minLrMm = Math.min(
    tailPercentile95(footClearsMm.map((r) => r[0]), 20),
    tailPercentile95(footClearsMm.map((r) => r[1]), 20)
  );
  const trajSample = subsampleTrajectory(rawGaitRows, 16);

  return {
    seed,
    steps,
    terminated,
    distance: dx,
    abs_y: absY,
    roll_deg: rollPtp,
    clear_mm: clearMm,
    min_lr_mm: minLrMm,
    ssup_pct: ssupPct,
    return: totalReturn,
    pose_stride: Math.max(1, poseStride),
    body_poses: bodyPoses,
    vx_trace: vxTrace,
    stepTrace,
    stepActivations: stepActs,
    traj_sample: trajSample,
  };
}

async function evaluateMicroduck(genome, seeds, baseModel = null, physicsBuf = null, envParams = null) {
  const spec = taskSpecs.microduck || { max_steps: 400 };
  const maxSteps = spec.max_steps || 400;
  let sumRet = 0.0;
  let sumSteps = 0.0;
  let minSteps = maxSteps;
  let sumX = 0.0;
  let sumAbsY = 0.0;
  let sumClear = 0.0;
  let sumMinLr = 0.0;
  let sumSsup = 0.0;
  let sumRoll = 0.0;
  let primaryRollout = null;

  for (let i = 0; i < seeds.length; i++) {
    const ro = await microduckRollout(
      genome,
      seeds[i],
      maxSteps,
      null,
      false,
      i === 0,
      2,
      baseModel,
      physicsBuf,
      envParams
    );
    if (i === 0) primaryRollout = ro;
    sumRet += ro.return;
    sumSteps += ro.steps;
    if (ro.steps < minSteps) minSteps = ro.steps;
    sumX += ro.distance;
    sumAbsY += ro.abs_y;
    sumClear += ro.clear_mm;
    sumMinLr += ro.min_lr_mm;
    sumSsup += ro.ssup_pct;
    sumRoll += ro.roll_deg;
  }

  const n = Math.max(1, seeds.length);
  const meanRet = Math.round((sumRet / n) * 10) / 10;
  const meanX = Math.round((sumX / n) * 1000) / 1000;
  const meanSteps = Math.round((sumSteps / n) * 10) / 10;
  const clearMm = Math.round((sumClear / n) * 10) / 10;
  const minLrMm = Math.round((sumMinLr / n) * 10) / 10;
  const ssupPct = Math.round(sumSsup / n);
  const rollDeg = Math.round((sumRoll / n) * 10) / 10;
  const absY = Math.round((sumAbsY / n) * 1000) / 1000;

  const solved = minSteps >= maxSteps && meanRet >= 180.0 && meanX >= 0.35;
  const tier = solved ? "solved" : (minSteps >= maxSteps * 0.75 && meanRet >= 20.0 ? "mid" : "");
  const dispTxt = `${meanRet >= 0 ? "+" : ""}${meanRet.toFixed(0)}`;

  return {
    summary: {
      task: "microduck",
      seeds,
      return: meanRet,
      mean_distance: meanX,
      mean_steps: meanSteps,
      min_steps: minSteps,
      clear_mm: clearMm,
      min_lr_mm: minLrMm,
      ssup_pct: ssupPct,
      roll_deg: rollDeg,
      abs_y: absY,
      solved,
      primary_value: meanRet,
      primary_display: `${dispTxt} (${meanX >= 0 ? "+" : ""}${meanX.toFixed(2)}m)`,
      badge_text: `${dispTxt} · ${meanX >= 0 ? "+" : ""}${meanX.toFixed(2)}m`,
      tier,
      secondary_text: `steps ${minSteps}/${maxSteps} · clr ${clearMm.toFixed(1)}mm · ssup ${ssupPct}%`,
    },
    traj_sample: primaryRollout ? primaryRollout.traj_sample : null,
    primary_rollout: primaryRollout,
  };
}

// --- Unitree Go2 12-DOF Quadrupedal Locomotion Environment (PopuLoRA + MuJoCo WASM) ---

function ensureUnitreeGo2SceneMeta(mujoco, entry, envParams) {
  if (entry.initialized) return entry.meta;
  const { model } = entry;
  const p = envParams || {};
  model.opt.timestep = p.physics_dt || 0.004;

  const objKey = mujoco.mjtObj.mjOBJ_KEY.value ?? mujoco.mjtObj.mjOBJ_KEY;
  const objGeom = mujoco.mjtObj.mjOBJ_GEOM.value ?? mujoco.mjtObj.mjOBJ_GEOM;
  const objSite = mujoco.mjtObj.mjOBJ_SITE.value ?? mujoco.mjtObj.mjOBJ_SITE;
  const objBody = mujoco.mjtObj.mjOBJ_BODY.value ?? mujoco.mjtObj.mjOBJ_BODY;

  const homeIdx = Math.max(0, mujoco.mj_name2id(model, objKey, "home"));
  const homeQpos = Float64Array.from(model.key_qpos.subarray(homeIdx * model.nq, (homeIdx + 1) * model.nq));
  const homeCtrl = Float64Array.from(model.key_ctrl.subarray(homeIdx * model.nu, (homeIdx + 1) * model.nu));

  const floorGeom = mujoco.mj_name2id(model, objGeom, "floor");
  const footGeoms = ["FL", "FR", "RL", "RR"].map((n) => mujoco.mj_name2id(model, objGeom, n));
  const footSites = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"].map((n) => mujoco.mj_name2id(model, objSite, n));
  const footBodySet = new Set(footGeoms.map((gid) => model.geom_bodyid[gid]));
  const nonFootBodies = [];
  for (let b = 1; b < model.nbody; b++) {
    if (!footBodySet.has(b)) nonFootBodies.push(b);
  }

  const { ctrlMin, ctrlMax } = readActuatorCtrlBounds(model, 10.0);

  entry.meta = {
    objBody,
    homeQpos,
    homeCtrl,
    floorGeom,
    footGeoms,
    footSites,
    footBodySet,
    nonFootBodies,
    ctrlMin,
    ctrlMax,
    velLocalBuf: new Float64Array(6),
    velWorldBuf: new Float64Array(6),
    obsBuf: new Float32Array(49),
  };
  entry.initialized = true;
  return entry.meta;
}

async function unitreeGo2Rollout(
  genome,
  seed = 0,
  maxSteps = 400,
  nudge = null,
  includeActs = false,
  capturePoses = true,
  poseStride = 2,
  suppliedBaseModel = null,
  suppliedPhysicsBuf = null,
  envParamsOverride = null
) {
  const { scene_id: sceneId, manifest, env_params: envParams } = resolveSceneTaskConfig("unitree_go2", envParamsOverride);

  const [{ mujoco, model, data }, baseModel] = await Promise.all([
    ensureMujocoScene(sceneId, manifest, suppliedPhysicsBuf),
    ensurePopuLoraBaseModel("unitree_go2", genome, suppliedBaseModel),
  ]);
  const sceneEntry = mountedScenes.get(sceneId);
  const meta = ensureUnitreeGo2SceneMeta(mujoco, sceneEntry, envParams);
  const compiledPolicy = compilePopuLoraIndividual(baseModel, genome);
  const policyRunner = createPopuLoraRunner(compiledPolicy);
  policyRunner.reset();

  const nSubsteps = envParams.n_substeps || 5;
  const ctrlDt = envParams.ctrl_dt || 0.02;
  const actionScale = envParams.action_scale ?? 0.42;
  const angVelObsScale = envParams.ang_vel_obs_scale ?? 0.25;
  const jointVelObsScale = envParams.joint_vel_obs_scale ?? 0.05;
  const cmd = envParams.command || [0.65, 0.0];
  const cmdVx = cmd[0];
  const footRadius = envParams.foot_radius ?? 0.022;
  const nominalBaseZ = envParams.nominal_base_z ?? 0.268;
  const minBaseZ = envParams.min_base_z ?? 0.145;
  const minUpright = envParams.min_uprightness ?? 0.45;

  const signedProgressW = envParams.signed_progress_weight ?? 140.0;
  const highWaterW = envParams.high_water_weight ?? 60.0;
  const velTrackW = envParams.vel_track_weight ?? 16.0;
  const clearanceW = envParams.clearance_weight ?? 24.0;
  const clearanceTarget = envParams.clearance_target ?? 0.055;
  const stepStrideW = envParams.step_stride_weight ?? 35.0;
  const footSlipW = envParams.foot_slip_weight ?? 1.2;
  const trotBonusW = envParams.trot_bonus_weight ?? 8.0;
  const orientW = envParams.orient_weight ?? 8.0;
  const lateralDriftW = envParams.lateral_drift_weight ?? 12.0;
  const heightW = envParams.height_weight ?? 20.0;
  const bodyAngVelW = envParams.body_ang_vel_weight ?? 0.35;
  const vertVelW = envParams.vert_vel_weight ?? 1.5;
  const hipPoseW = envParams.hip_pose_weight ?? 3.0;
  const actionRateW = envParams.action_rate_weight ?? 0.08;
  const fallPenalty = envParams.fall_penalty ?? 120.0;
  const hipJoints = envParams.hip_joints || [0, 3, 6, 9];

  mujoco.mj_resetData(model, data);
  let qpos = data.qpos;
  let qvel = data.qvel;
  let ctrl = data.ctrl;

  const rng = mulberry32(seed);
  qpos.set(meta.homeQpos);
  qpos[2] += (rng() * 2.0 - 1.0) * 0.002;
  for (let j = 0; j < 12; j++) qpos[7 + j] += (rng() * 2.0 - 1.0) * 0.01;
  for (let v = 0; v < model.nv; v++) qvel[v] = (rng() * 2.0 - 1.0) * 0.02;
  ctrl.set(meta.homeCtrl);
  mujoco.mj_forward(model, data);
  if (qpos.byteLength === 0) {
    qpos = data.qpos;
    qvel = data.qvel;
    ctrl = data.ctrl;
  }

  if (nudge === "fwd") qvel[0] += 0.65;
  else if (nudge === "back") qvel[0] -= 0.55;
  else if (nudge === "lat") qvel[1] += 0.55;

  const startX = qpos[0];
  const startY = qpos[1];
  let prevX = startX;
  let maxX = startX;
  const lastAction = new Float32Array(12);
  const prevFootX = new Float64Array(4);
  const prevFootY = new Float64Array(4);
  const tdFootX = new Float64Array(4);
  const airTime = new Float64Array(4);
  const swingPeak = new Float64Array(4);
  for (let f = 0; f < 4; f++) {
    const sOff = meta.footSites[f] * 3;
    prevFootX[f] = data.site_xpos[sOff];
    prevFootY[f] = data.site_xpos[sOff + 1];
    tdFootX[f] = data.site_xpos[sOff];
  }

  let totalReturn = 0.0;
  let trotSteps = 0;
  const swingClearMmSamples = [];
  const rawGaitRows = [];
  let dispIntX = 0.0;
  let dispIntY = 0.0;

  const fillObs = () => {
    const obs = meta.obsBuf;
    const xmOff = 1 * 9; // base = body 1
    obs[0] = -data.xmat[xmOff + 6];
    obs[1] = -data.xmat[xmOff + 7];
    obs[2] = -data.xmat[xmOff + 8];

    mujoco.mj_objectVelocity(model, data, meta.objBody, 1, meta.velLocalBuf, 1);
    mujoco.mj_objectVelocity(model, data, meta.objBody, 1, meta.velWorldBuf, 0);
    if (qpos.byteLength === 0) {
      qpos = data.qpos;
      qvel = data.qvel;
      ctrl = data.ctrl;
    }

    obs[3] = meta.velLocalBuf[3];
    obs[4] = meta.velLocalBuf[4];
    obs[5] = meta.velLocalBuf[5];
    obs[6] = meta.velLocalBuf[0] * angVelObsScale;
    obs[7] = meta.velLocalBuf[1] * angVelObsScale;
    obs[8] = meta.velLocalBuf[2] * angVelObsScale;
    obs[9] = meta.velWorldBuf[3];
    obs[10] = meta.velWorldBuf[4];

    for (let j = 0; j < 12; j++) obs[11 + j] = qpos[7 + j] - meta.homeQpos[7 + j];
    for (let j = 0; j < 12; j++) obs[23 + j] = qvel[6 + j] * jointVelObsScale;
    for (let j = 0; j < 12; j++) obs[35 + j] = lastAction[j];
    obs[47] = cmd[0];
    obs[48] = cmd[1];
    return obs;
  };

  const bodyPoses = capturePoses ? [captureSceneBodyPoses(model, data)] : null;
  const vxTrace = capturePoses ? [0.0] : null;
  const stepTrace = capturePoses ? [[0.0, 0.0, 0.0, 0.0, 4.0]] : null;
  const stepActs = includeActs ? [] : null;

  let steps = 0;
  let terminated = false;

  for (let s = 0; s < maxSteps; s++) {
    const obs = fillObs();
    dispIntX += obs[9] * ctrlDt;
    dispIntY += obs[10] * ctrlDt;
    const gaitRow = new Array(22);
    gaitRow[0] = dispIntX;
    gaitRow[1] = dispIntY;
    gaitRow[2] = obs[9];
    gaitRow[3] = obs[10];
    gaitRow[4] = obs[0];
    gaitRow[5] = obs[1];
    gaitRow[6] = obs[2];
    gaitRow[7] = obs[6];
    gaitRow[8] = obs[7];
    gaitRow[9] = obs[8];
    for (let j = 0; j < 12; j++) gaitRow[10 + j] = obs[11 + j];
    rawGaitRows.push(gaitRow);

    const { action, activations } = policyRunner.step(obs, includeActs);
    if (includeActs && (s % Math.max(1, poseStride) === 0)) stepActs.push(activations);

    let actRateAcc = 0.0;
    for (let j = 0; j < 12; j++) {
      const aRaw = Number.isFinite(action[j]) ? action[j] : 0.0;
      const aEff = Math.max(-1.0, Math.min(1.0, aRaw));
      const diff = aEff - lastAction[j];
      actRateAcc += diff * diff;
      const rawTarget = meta.homeCtrl[j] + actionScale * aEff;
      ctrl[j] = Math.max(meta.ctrlMin[j], Math.min(meta.ctrlMax[j], rawTarget));
      lastAction[j] = aEff;
    }
    const actionRate = actRateAcc / 12.0;

    for (let sub = 0; sub < nSubsteps; sub++) mujoco.mj_step(model, data);
    if (qpos.byteLength === 0) {
      qpos = data.qpos;
      qvel = data.qvel;
      ctrl = data.ctrl;
    }
    steps++;

    if (!isMjStateFinite(model, qpos, qvel)) {
      totalReturn -= fallPenalty;
      terminated = true;
      break;
    }

    const contacts = [false, false, false, false];
    for (let f = 0; f < 4; f++) {
      const distF = mujoco.mj_geomDistance(model, data, meta.floorGeom, meta.footGeoms[f], 0.04, null);
      contacts[f] = distF <= 0.0015;
    }
    let minNonFootZ = Infinity;
    for (let bi = 0; bi < meta.nonFootBodies.length; bi++) {
      const bz = data.xpos[meta.nonFootBodies[bi] * 3 + 2];
      if (bz < minNonFootZ) minNonFootZ = bz;
    }
    const bodyHit = minNonFootZ < 0.035;

    if (qpos.byteLength === 0) {
      qpos = data.qpos;
      qvel = data.qvel;
      ctrl = data.ctrl;
    }
    const baseZ = qpos[2];
    const xmOff = 9;
    const upZ = data.xmat[xmOff + 8];
    if (bodyHit || baseZ < minBaseZ || upZ < minUpright) {
      terminated = true;
    }

    mujoco.mj_objectVelocity(model, data, meta.objBody, 1, meta.velLocalBuf, 1);
    mujoco.mj_objectVelocity(model, data, meta.objBody, 1, meta.velWorldBuf, 0);
    if (qpos.byteLength === 0) {
      qpos = data.qpos;
      qvel = data.qvel;
      ctrl = data.ctrl;
    }
    const vxW = meta.velWorldBuf[3];
    const vyW = meta.velWorldBuf[4];
    const vzW = meta.velWorldBuf[5];
    const wxLoc = meta.velLocalBuf[0];
    const wyLoc = meta.velLocalBuf[1];

    const curX = qpos[0];
    const dxStep = curX - prevX;
    prevX = curX;
    const dNew = Math.max(0.0, curX - maxX);
    if (curX > maxX) maxX = curX;

    const rProg = signedProgressW * dxStep + highWaterW * dNew;
    const fwdGate = Math.max(0.0, Math.min(1.0, (vxW + 0.02) / 0.25));
    const vxErr = (vxW - cmdVx) * (vxW - cmdVx) + 1.5 * vyW * vyW;
    const rVel = velTrackW * Math.exp(-vxErr / 0.22) * ctrlDt;

    let rFeet = 0.0;
    let stepMaxClearMm = 0.0;
    let contactCount = 0;
    for (let f = 0; f < 4; f++) {
      const sOff = meta.footSites[f] * 3;
      const fx = data.site_xpos[sOff];
      const fy = data.site_xpos[sOff + 1];
      const fz = data.site_xpos[sOff + 2];
      const dfx = fx - prevFootX[f];
      const dfy = fy - prevFootY[f];
      const footV = Math.hypot(dfx, dfy) / ctrlDt;
      const footVx = dfx / ctrlDt;
      prevFootX[f] = fx;
      prevFootY[f] = fy;

      const clr = Math.max(0.0, fz - footRadius);
      if (contacts[f]) {
        contactCount++;
        const strideFwd = Math.max(0.0, fx - tdFootX[f]);
        if (airTime[f] >= 0.06 && airTime[f] <= 0.50 && strideFwd > 0.01 && vxW > 0.0) {
          rFeet += stepStrideW * Math.min(strideFwd, 0.18) * Math.min(swingPeak[f] / 0.03, 1.5);
        }
        airTime[f] = 0.0;
        swingPeak[f] = 0.0;
        tdFootX[f] = fx;
        rFeet -= footSlipW * footV * footV * ctrlDt;
      } else {
        airTime[f] += ctrlDt;
        if (clr > swingPeak[f]) swingPeak[f] = clr;
        const clrMm = clr * 1000.0;
        if (clrMm > stepMaxClearMm) stepMaxClearMm = clrMm;
        if (footVx > 0.02) {
          rFeet += clearanceW * Math.min(clr, clearanceTarget) * Math.min(footVx, 1.5) * ctrlDt * fwdGate;
        }
      }
    }
    swingClearMmSamples.push(stepMaxClearMm);

    const diag1 = contacts[0] && contacts[3] && !contacts[1] && !contacts[2];
    const diag2 = contacts[1] && contacts[2] && !contacts[0] && !contacts[3];
    if (diag1 || diag2) {
      trotSteps++;
      if (vxW > 0.08) rFeet += trotBonusW * ctrlDt;
    }

    const gx = -data.xmat[xmOff + 6];
    const gy = -data.xmat[xmOff + 7];
    const latDrift = Math.abs(qpos[1] - startY);
    let hipErr = 0.0;
    for (let h = 0; h < hipJoints.length; h++) {
      const hj = hipJoints[h];
      const dq = qpos[7 + hj] - meta.homeQpos[7 + hj];
      hipErr += dq * dq;
    }

    const rPen =
      orientW * (gx * gx + gy * gy) * ctrlDt +
      lateralDriftW * latDrift * ctrlDt +
      heightW * (baseZ - nominalBaseZ) * (baseZ - nominalBaseZ) * ctrlDt +
      bodyAngVelW * (wxLoc * wxLoc + wyLoc * wyLoc) * ctrlDt +
      vertVelW * vzW * vzW * ctrlDt +
      hipPoseW * hipErr * ctrlDt +
      actionRateW * actionRate;

    let stepReward = rProg + rVel + rFeet - rPen;
    if (terminated) stepReward -= fallPenalty;
    totalReturn += stepReward;

    const dispX = curX - startX;
    if (capturePoses && (s % Math.max(1, poseStride) === 0 || terminated || s === maxSteps - 1)) {
      bodyPoses.push(captureSceneBodyPoses(model, data));
      vxTrace.push(Math.round(vxW * 1000) / 1000);
      stepTrace.push([
        Math.round(dispX * 1000) / 1000,
        Math.round(totalReturn * 10) / 10,
        Math.round(stepMaxClearMm * 10) / 10,
        Math.round(vxW * 100) / 100,
        contactCount,
      ]);
    }

    if (terminated) break;
  }

  const dx = qpos[0] - startX;
  const absY = Math.abs(qpos[1] - startY);
  const meanVx = dx / Math.max(ctrlDt, steps * ctrlDt);
  const trotPct = (100.0 * trotSteps) / Math.max(1, steps);

  const clearMm = tailPercentile95(swingClearMmSamples, 20);
  const trajSample = subsampleTrajectory(rawGaitRows, 16);

  return {
    seed,
    steps,
    terminated,
    distance: dx,
    mean_vx: meanVx,
    abs_y: absY,
    clear_mm: clearMm,
    trot_pct: trotPct,
    return: totalReturn,
    pose_stride: Math.max(1, poseStride),
    body_poses: bodyPoses,
    vx_trace: vxTrace,
    stepTrace,
    stepActivations: stepActs,
    traj_sample: trajSample,
  };
}

async function evaluateUnitreeGo2(genome, seeds, baseModel = null, physicsBuf = null, envParams = null) {
  const spec = taskSpecs.unitree_go2 || { max_steps: 400 };
  const maxSteps = spec.max_steps || 400;
  let sumRet = 0.0;
  let sumSteps = 0.0;
  let minSteps = maxSteps;
  let sumX = 0.0;
  let sumVx = 0.0;
  let sumClear = 0.0;
  let sumTrot = 0.0;
  let primaryRollout = null;

  for (let i = 0; i < seeds.length; i++) {
    const ro = await unitreeGo2Rollout(
      genome,
      seeds[i],
      maxSteps,
      null,
      false,
      i === 0,
      2,
      baseModel,
      physicsBuf,
      envParams
    );
    if (i === 0) primaryRollout = ro;
    sumRet += ro.return;
    sumSteps += ro.steps;
    if (ro.steps < minSteps) minSteps = ro.steps;
    sumX += ro.distance;
    sumVx += ro.mean_vx;
    sumClear += ro.clear_mm;
    sumTrot += ro.trot_pct;
  }

  const n = Math.max(1, seeds.length);
  const meanRet = Math.round((sumRet / n) * 10) / 10;
  const meanX = Math.round((sumX / n) * 1000) / 1000;
  const meanVx = Math.round((sumVx / n) * 100) / 100;
  const meanSteps = Math.round((sumSteps / n) * 10) / 10;
  const clearMm = Math.round((sumClear / n) * 10) / 10;
  const trotPct = Math.round(sumTrot / n);

  const solved = minSteps >= maxSteps && meanRet >= 320.0 && meanX >= 1.5;
  const tier = solved ? "solved" : (minSteps >= maxSteps * 0.75 && meanRet >= 80.0 && meanX >= 0.25 ? "mid" : "");
  const dispTxt = `${meanRet >= 0 ? "+" : ""}${meanRet.toFixed(0)}`;

  return {
    summary: {
      task: "unitree_go2",
      seeds,
      return: meanRet,
      mean_distance: meanX,
      mean_vx: meanVx,
      mean_steps: meanSteps,
      min_steps: minSteps,
      clear_mm: clearMm,
      trot_pct: trotPct,
      solved,
      primary_value: meanRet,
      primary_display: `${dispTxt} (${meanX >= 0 ? "+" : ""}${meanX.toFixed(2)}m)`,
      badge_text: `${dispTxt} \u00b7 ${meanX >= 0 ? "+" : ""}${meanX.toFixed(2)}m`,
      tier,
      secondary_text: `steps ${minSteps}/${maxSteps} \u00b7 vx ${meanVx >= 0 ? "+" : ""}${meanVx.toFixed(2)}m/s \u00b7 clr ${clearMm.toFixed(1)}mm`,
    },
    traj_sample: primaryRollout ? primaryRollout.traj_sample : null,
    primary_rollout: primaryRollout,
  };
}

// --- SO-101 / SO-ARM102 6-DOF Robotic Arm Cube Pick & Lift Environment (PopuLoRA + MuJoCo WASM) ---

function resolveSo101Variant(envParamsOverride = null) {
  const { variant, scene_id: sceneId, manifest, env_params: envParams } = resolveSceneTaskConfig("so101", envParamsOverride);
  const armModel = String(variant || envParams.arm_model || sceneId || "so102");
  return { arm_model: armModel, scene_id: sceneId, manifest, env_params: envParams };
}

function ensureSo101SceneMeta(mujoco, entry, envParams) {
  if (entry.initialized) return entry.meta;
  const { model } = entry;
  const p = envParams || {};
  model.opt.timestep = p.physics_dt || 0.004;

  const objKey = mujoco.mjtObj.mjOBJ_KEY.value ?? mujoco.mjtObj.mjOBJ_KEY;
  const objGeom = mujoco.mjtObj.mjOBJ_GEOM.value ?? mujoco.mjtObj.mjOBJ_GEOM;
  const objSite = mujoco.mjtObj.mjOBJ_SITE.value ?? mujoco.mjtObj.mjOBJ_SITE;
  const objJnt = mujoco.mjtObj.mjOBJ_JOINT.value ?? mujoco.mjtObj.mjOBJ_JOINT;
  const objBody = mujoco.mjtObj.mjOBJ_BODY.value ?? mujoco.mjtObj.mjOBJ_BODY;
  const jntSlideVal = mujoco.mjtJoint?.mjJNT_SLIDE?.value ?? mujoco.mjtJoint?.mjJNT_SLIDE ?? 2;

  const homeIdx = Math.max(0, mujoco.mj_name2id(model, objKey, "home"));
  const homeQpos = Float64Array.from(model.key_qpos.subarray(homeIdx * model.nq, (homeIdx + 1) * model.nq));
  const homeCtrl = Float64Array.from(model.key_ctrl.subarray(homeIdx * model.nu, (homeIdx + 1) * model.nu));

  const fixedJawGeom = mujoco.mj_name2id(model, objGeom, "fixed_jaw_pad");
  const movingJawGeom = mujoco.mj_name2id(model, objGeom, "moving_jaw_pad");
  const cubeGeom = mujoco.mj_name2id(model, objGeom, "cube_col");
  const fixedTipSite = mujoco.mj_name2id(model, objSite, "fixed_tip");
  const movingTipSite = mujoco.mj_name2id(model, objSite, "moving_tip");

  let cubeJnt = mujoco.mj_name2id(model, objJnt, "cube_freejoint");
  if (cubeJnt < 0) {
    const cubeBody = mujoco.mj_name2id(model, objBody, "cube");
    cubeJnt = cubeBody >= 0 ? Number(model.body_jntadr[cubeBody]) : -1;
  }
  const cubeQposAdr = cubeJnt >= 0 ? Number(model.jnt_qposadr[cubeJnt]) : 6;
  const cubeDofAdr = cubeJnt >= 0 ? Number(model.jnt_dofadr[cubeJnt]) : 6;

  const gripperJnt = mujoco.mj_name2id(model, objJnt, "gripper");
  const gripperIsSlide = gripperJnt >= 0 && Number(model.jnt_type[gripperJnt]) === Number(jntSlideVal);
  const mirrorJnt = mujoco.mj_name2id(model, objJnt, "gripper_mirror");
  const mirrorQposAdr = mirrorJnt >= 0 ? Number(model.jnt_qposadr[mirrorJnt]) : -1;

  const { ctrlMin, ctrlMax } = readActuatorCtrlBounds(model, 3.14);

  entry.meta = {
    homeQpos,
    homeCtrl,
    fixedJawGeom,
    movingJawGeom,
    cubeGeom,
    fixedTipSite,
    movingTipSite,
    cubeQposAdr,
    cubeDofAdr,
    gripperIsSlide,
    mirrorQposAdr,
    ctrlMin,
    ctrlMax,
    obsBuf: new Float32Array(33),
  };
  entry.initialized = true;
  return entry.meta;
}

async function so101Rollout(
  genome,
  seed = 0,
  maxSteps = 250,
  nudge = null,
  includeActs = false,
  capturePoses = true,
  poseStride = 2,
  suppliedBaseModel = null,
  suppliedPhysicsBuf = null,
  envParamsOverride = null
) {
  const { arm_model: armModel, scene_id: sceneId, manifest, env_params: envParams } = resolveSo101Variant(envParamsOverride);

  const [{ mujoco, model, data }, baseModel] = await Promise.all([
    ensureMujocoScene(sceneId, manifest, suppliedPhysicsBuf),
    ensurePopuLoraBaseModel("so101", genome, suppliedBaseModel),
  ]);
  const sceneEntry = mountedScenes.get(sceneId);
  const meta = ensureSo101SceneMeta(mujoco, sceneEntry, envParams);
  const compiledPolicy = compilePopuLoraIndividual(baseModel, genome);
  const policyRunner = createPopuLoraRunner(compiledPolicy);
  policyRunner.reset();

  const nSubsteps = envParams.n_substeps || 5;
  const ctrlDt = envParams.ctrl_dt || 0.02;
  const actionScales =
    envParams.action_scales ||
    (meta.gripperIsSlide ? [0.65, 0.65, 0.65, 0.65, 0.55, 0.028] : [0.65, 0.65, 0.65, 0.65, 0.55, 0.95]);
  const goalPos = envParams.goal_pos || (meta.gripperIsSlide ? [0.238, 0.0, 0.14] : [0.23, 0.012, 0.14]);
  const cubeRestZ = envParams.cube_rest_z ?? 0.016;

  const reachW = envParams.reach_weight ?? 12.0;
  const reachProgW = envParams.reach_progress_weight ?? 45.0;
  const preGraspW = envParams.pre_grasp_align_weight ?? 5.0;
  const singleTouchW = envParams.single_touch_weight ?? 2.5;
  const pinchGraspW = envParams.pinch_grasp_weight ?? 16.0;
  const liftW = envParams.lift_weight ?? 220.0;
  const liftHighWaterW = envParams.lift_high_water_weight ?? 650.0;
  const goalHoldW = envParams.goal_hold_weight ?? 24.0;
  const knockPenW = envParams.knock_penalty_weight ?? 15.0;
  const floorScrapeW = envParams.floor_scrape_weight ?? 8.0;
  const actionRateW = envParams.action_rate_weight ?? 0.04;

  const cQ = meta.cubeQposAdr;
  const cV = meta.cubeDofAdr;

  mujoco.mj_resetData(model, data);
  let qpos = data.qpos;
  let qvel = data.qvel;
  let ctrl = data.ctrl;

  const rng = mulberry32(seed);
  qpos.set(meta.homeQpos);
  for (let j = 0; j < 6; j++) {
    const pertScale = j === 5 && meta.gripperIsSlide ? 0.0015 : 0.015;
    qpos[j] += (rng() * 2.0 - 1.0) * pertScale;
  }
  if (meta.gripperIsSlide) {
    qpos[5] = Math.max(meta.ctrlMin[5], Math.min(meta.ctrlMax[5], qpos[5]));
    if (meta.mirrorQposAdr >= 0) qpos[meta.mirrorQposAdr] = -qpos[5];
  }
  // Perturb initial cube XY slightly per seed for closed-loop reaching
  if (seed > 0) {
    qpos[cQ] += (rng() * 2.0 - 1.0) * 0.015;
    qpos[cQ + 1] += (rng() * 2.0 - 1.0) * 0.018;
  }
  if (nudge === "left") qpos[cQ + 1] += 0.032;
  else if (nudge === "right") qpos[cQ + 1] -= 0.032;
  else if (nudge === "back") qpos[cQ] += 0.030;

  ctrl.set(meta.homeCtrl);
  mujoco.mj_forward(model, data);
  qpos = data.qpos;
  qvel = data.qvel;
  ctrl = data.ctrl;

  const startCubeX = qpos[cQ];
  const startCubeY = qpos[cQ + 1];
  const lastAction = new Float32Array(6);
  const gripPosObsScale = meta.gripperIsSlide ? 25.0 : 1.0;
  const gripVelObsScale = meta.gripperIsSlide ? 2.5 : 0.10;

  const readPinchAndCube = () => {
    const sx = data.site_xpos;
    const qp = data.qpos;
    const fOff = meta.fixedTipSite * 3;
    const mOff = meta.movingTipSite * 3;
    const px = 0.5 * (sx[fOff] + sx[mOff]);
    const py = 0.5 * (sx[fOff + 1] + sx[mOff + 1]);
    const pz = 0.5 * (sx[fOff + 2] + sx[mOff + 2]);
    const minTipZ = Math.min(sx[fOff + 2], sx[mOff + 2]);
    return {
      px, py, pz, minTipZ,
      cx: qp[cQ], cy: qp[cQ + 1], cz: qp[cQ + 2],
    };
  };

  const initSt = readPinchAndCube();
  let minReachDist = Math.hypot(initSt.cx - initSt.px, initSt.cy - initSt.py, initSt.cz - initSt.pz);
  let maxLiftMm = 0.0;
  let maxLiftM = 0.0;
  let pinchSteps = 0;
  let totalReturn = 0.0;
  const rawTrajRows = [];

  const fillObs = (st) => {
    const obs = meta.obsBuf;
    const qp = data.qpos;
    const qv = data.qvel;
    for (let j = 0; j < 5; j++) {
      obs[j] = qp[j] - meta.homeQpos[j];
      obs[6 + j] = qv[j] * 0.10;
    }
    obs[5] = (qp[5] - meta.homeQpos[5]) * gripPosObsScale;
    obs[11] = qv[5] * gripVelObsScale;
    obs[12] = (st.px - 0.22) * 4.0;
    obs[13] = st.py * 4.0;
    obs[14] = (st.pz - 0.08) * 4.0;
    obs[15] = (st.cx - 0.22) * 4.0;
    obs[16] = st.cy * 4.0;
    obs[17] = (st.cz - 0.08) * 4.0;
    obs[18] = (st.cx - st.px) * 5.0;
    obs[19] = (st.cy - st.py) * 5.0;
    obs[20] = (st.cz - st.pz) * 5.0;
    obs[21] = (goalPos[0] - st.cx) * 4.0;
    obs[22] = (goalPos[1] - st.cy) * 4.0;
    obs[23] = (goalPos[2] - st.cz) * 4.0;
    obs[24] = qv[cV] * 0.5;
    obs[25] = qv[cV + 1] * 0.5;
    obs[26] = qv[cV + 2] * 0.5;
    for (let j = 0; j < 6; j++) obs[27 + j] = lastAction[j];
    return obs;
  };

  const bodyPoses = capturePoses ? [captureSceneBodyPoses(model, data)] : null;
  const vxTrace = capturePoses ? [0.0] : null;
  const stepTrace = capturePoses ? [[Math.round(minReachDist * 1000) / 10, 0.0, 0.0, 0.0, 0.0]] : null;
  const stepActs = includeActs ? [] : null;

  let steps = 0;
  let finalDistCm = minReachDist * 100.0;

  for (let s = 0; s < maxSteps; s++) {
    const stBefore = readPinchAndCube();
    const obs = fillObs(stBefore);
    const trajRow = new Array(15);
    trajRow[0] = stBefore.px;
    trajRow[1] = stBefore.py;
    trajRow[2] = stBefore.pz;
    trajRow[3] = stBefore.cx;
    trajRow[4] = stBefore.cy;
    trajRow[5] = stBefore.cz;
    trajRow[6] = stBefore.cx - stBefore.px;
    trajRow[7] = stBefore.cy - stBefore.py;
    trajRow[8] = stBefore.cz - stBefore.pz;
    for (let j = 0; j < 6; j++) trajRow[9 + j] = obs[j];
    rawTrajRows.push(trajRow);

    const { action, activations } = policyRunner.step(obs, includeActs);
    if (includeActs && (s % Math.max(1, poseStride) === 0)) stepActs.push(activations);

    ctrl = data.ctrl;
    let actRateAcc = 0.0;
    for (let j = 0; j < 6; j++) {
      const aRaw = Number.isFinite(action[j]) ? action[j] : 0.0;
      const aEff = Math.max(-1.0, Math.min(1.0, aRaw));
      const diff = aEff - lastAction[j];
      actRateAcc += diff * diff;
      const rawTarget = meta.homeCtrl[j] + actionScales[j] * aEff;
      ctrl[j] = Math.max(meta.ctrlMin[j], Math.min(meta.ctrlMax[j], rawTarget));
      lastAction[j] = aEff;
    }
    const actionRate = actRateAcc / 6.0;

    for (let sub = 0; sub < nSubsteps; sub++) mujoco.mj_step(model, data);
    qpos = data.qpos;
    qvel = data.qvel;
    steps++;

    if (!isMjStateFinite(model, qpos, qvel)) {
      totalReturn -= 50.0;
      break;
    }

    const dFix = mujoco.mj_geomDistance(model, data, meta.fixedJawGeom, meta.cubeGeom, 0.08, null);
    const dMov = mujoco.mj_geomDistance(model, data, meta.movingJawGeom, meta.cubeGeom, 0.08, null);
    qpos = data.qpos;
    const st = readPinchAndCube();

    const dist = Math.hypot(st.cx - st.px, st.cy - st.py, st.cz - st.pz);
    finalDistCm = dist * 100.0;
    const dReachProg = Math.max(0.0, minReachDist - dist);
    if (dist < minReachDist) minReachDist = dist;

    const bothTouch = dFix <= 0.0025 && dMov <= 0.0025;
    const anyTouch = dFix <= 0.003 || dMov <= 0.003;
    if (bothTouch) pinchSteps++;

    const liftM = Math.max(0.0, st.cz - cubeRestZ);
    const liftClamped = Math.min(liftM, 0.15);
    const dLiftNew = Math.max(0.0, liftClamped - maxLiftM);
    if (liftClamped > maxLiftM) maxLiftM = liftClamped;
    const curLiftMm = liftM * 1000.0;
    if (curLiftMm > maxLiftMm) maxLiftMm = curLiftMm;

    const rReach = reachW * (1.0 - Math.tanh(6.5 * dist)) * ctrlDt + reachProgW * dReachProg;
    const gripPos = qpos[5];
    let rGrip = 0.0;
    if (meta.gripperIsSlide) {
      if (dist > 0.042) {
        rGrip = 1.5 * Math.max(0.0, Math.min(1.0, (gripPos - 0.012) / 0.016)) * ctrlDt;
      } else {
        rGrip = preGraspW * (1.0 - Math.max(0.0, Math.min(1.0, (gripPos - 0.002) / 0.024))) * ctrlDt;
      }
    } else {
      if (dist > 0.042) {
        rGrip = 1.5 * Math.max(0.0, Math.min(1.0, (gripPos - 0.25) / 0.40)) * ctrlDt;
      } else {
        rGrip = preGraspW * (1.0 - Math.max(0.0, Math.min(1.0, (gripPos + 0.12) / 0.75))) * ctrlDt;
      }
    }
    if (bothTouch) rGrip += pinchGraspW * ctrlDt;
    else if (anyTouch && dist < 0.045) rGrip += singleTouchW * ctrlDt;

    const goalDist = Math.hypot(st.cx - goalPos[0], st.cy - goalPos[1], st.cz - goalPos[2]);
    const rLift =
      liftW * liftClamped * ctrlDt +
      liftHighWaterW * dLiftNew +
      (liftM > 0.015 ? goalHoldW * (1.0 - Math.tanh(5.0 * goalDist)) * ctrlDt : 0.0);

    const cubeWander = Math.hypot(st.cx - startCubeX, st.cy - startCubeY);
    const rPen =
      (cubeWander > 0.14 ? knockPenW * ctrlDt : 0.0) +
      (st.minTipZ < 0.002 ? floorScrapeW * ctrlDt : 0.0) +
      actionRateW * actionRate;

    const stepReward = rReach + rGrip + rLift - rPen;
    if (Number.isFinite(stepReward)) totalReturn += stepReward;

    const gripState = bothTouch ? 2 : (anyTouch ? 1 : 0);
    if (capturePoses && (s % Math.max(1, poseStride) === 0 || s === maxSteps - 1)) {
      bodyPoses.push(captureSceneBodyPoses(model, data));
      vxTrace.push(Math.round(curLiftMm * 10) / 10);
      stepTrace.push([
        Math.round(finalDistCm * 10) / 10,
        Math.round(totalReturn * 10) / 10,
        Math.round(curLiftMm * 10) / 10,
        0.0,
        gripState,
      ]);
    }
  }

  const trajSample = subsampleTrajectory(rawTrajRows, 16);

  return {
    seed,
    arm_model: armModel,
    scene_id: sceneId,
    steps,
    terminated: false,
    distance: maxLiftM,
    lift_mm: maxLiftMm,
    min_dist_cm: minReachDist * 100.0,
    final_dist_cm: finalDistCm,
    pinch_pct: (100.0 * pinchSteps) / Math.max(1, steps),
    return: totalReturn,
    pose_stride: Math.max(1, poseStride),
    body_poses: bodyPoses,
    vx_trace: vxTrace,
    stepTrace,
    stepActivations: stepActs,
    traj_sample: trajSample,
  };
}

async function evaluateSo101(genome, seeds, baseModel = null, physicsBuf = null, envParams = null) {
  const spec = taskSpecs.so101 || { max_steps: 250 };
  const maxSteps = spec.max_steps || 250;
  const { arm_model: armModel } = resolveSo101Variant(envParams);
  let sumRet = 0.0;
  let sumLiftMm = 0.0;
  let sumMinDistCm = 0.0;
  let sumPinchPct = 0.0;
  let minSteps = maxSteps;
  let primaryRollout = null;

  for (let i = 0; i < seeds.length; i++) {
    const ro = await so101Rollout(
      genome,
      seeds[i],
      maxSteps,
      null,
      false,
      i === 0,
      2,
      baseModel,
      physicsBuf,
      envParams
    );
    if (i === 0) primaryRollout = ro;
    sumRet += ro.return;
    sumLiftMm += ro.lift_mm;
    sumMinDistCm += ro.min_dist_cm;
    sumPinchPct += ro.pinch_pct;
    if (ro.steps < minSteps) minSteps = ro.steps;
  }

  const n = Math.max(1, seeds.length);
  const meanRet = Math.round((sumRet / n) * 10) / 10;
  const liftMm = Math.round((sumLiftMm / n) * 10) / 10;
  const minDistCm = Math.round((sumMinDistCm / n) * 10) / 10;
  const pinchPct = Math.round(sumPinchPct / n);

  const solved = liftMm >= 100.0 && meanRet >= 160.0;
  const tier = solved ? "solved" : (liftMm >= 12.0 || (pinchPct >= 5 && meanRet >= 45.0) ? "mid" : "");
  const dispTxt = `${meanRet >= 0 ? "+" : ""}${meanRet.toFixed(0)}`;
  const liftCmTxt = `${(liftMm / 10.0).toFixed(1)}cm`;

  return {
    summary: {
      task: "so101",
      arm_model: armModel,
      seeds,
      return: meanRet,
      lift_mm: liftMm,
      min_dist_cm: minDistCm,
      pinch_pct: pinchPct,
      min_steps: minSteps,
      solved,
      primary_value: meanRet,
      primary_display: `${dispTxt} (lift ${liftCmTxt})`,
      badge_text: `${dispTxt} \u00b7 \u2191${liftCmTxt}`,
      tier,
      secondary_text: `min dist ${minDistCm.toFixed(1)}cm \u00b7 grip ${pinchPct}% \u00b7 lift ${liftMm.toFixed(1)}mm`,
    },
    traj_sample: primaryRollout ? primaryRollout.traj_sample : null,
    primary_rollout: primaryRollout,
  };
}

// --- Faithful Gymnasium LunarLander-v3 (Box2D via planck.js + PopuLoRA WeightOnlyMLP) ---
const LL_SOLVED_RETURN = 200.0;
const LL_TIER_MID_RETURN = 80.0;
const LL_DEFAULT_IN_COORDS = [-0.60, -0.10, -0.35, 0.10, 0.35, 0.60, -0.85, 0.85];
const LL_DEFAULT_OUT_COORDS = [-0.45, 0.45];

let lunarPlanckLoaded = false;

function ensureLunarPlanck() {
  if (lunarPlanckLoaded && self.planck && self.DetLunarRenderer) return;
  const plUrl = new URL(`${basePath}/assets/vendor/planck/planck.min.js`, self.location.href).href;
  const lrUrl = new URL(`${basePath}/assets/lunar_renderer.js`, self.location.href).href;
  importScripts(plUrl, lrUrl);
  lunarPlanckLoaded = true;
}

async function lunarlanderRollout(
  genome,
  seed = 0,
  maxSteps = 1000,
  nudge = null,
  includeActs = false,
  captureFrames = true,
  suppliedBaseModel = null,
  envParams = null
) {
  ensureLunarPlanck();
  const stepActs = includeActs ? [] : null;
  let policyFn;

  if (genome && genome.kind === "populora") {
    const baseModel = await ensurePopuLoraBaseModel("lunarlander", genome, suppliedBaseModel);
    const compiled = compilePopuLoraIndividual(baseModel, genome);
    const runner = createPopuLoraRunner(compiled);
    runner.reset();
    const obsBuf = new Float32Array(8);
    policyFn = (obs) => {
      for (let i = 0; i < 8; i++) obsBuf[i] = obs[i];
      const out = runner.step(obsBuf, includeActs);
      if (includeActs && out.activations) stepActs.push(out.activations);
      return out.action;
    };
  } else {
    const spec = taskSpecs.lunarlander || {
      obs_dim: 8,
      hidden_dims: [12],
      action_dim: 2,
      layer_coords: [LL_DEFAULT_IN_COORDS, null, LL_DEFAULT_OUT_COORDS],
    };
    const dims = [spec.obs_dim || 8, ...(spec.hidden_dims || [12]), spec.action_dim || 2];
    const layerCoords =
      Array.isArray(spec.layer_coords) && spec.layer_coords.length
        ? spec.layer_coords
        : [LL_DEFAULT_IN_COORDS, null, LL_DEFAULT_OUT_COORDS];
    const decoded = decodeHyperNeat(genome, dims, layerCoords);
    policyFn = (obs) => {
      const { action, activations } = forwardPolicy(decoded, obs, includeActs);
      if (includeActs) stepActs.push(activations);
      return action;
    };
  }

  const ro = self.DetLunarRenderer.lunarlanderRolloutPlanck(policyFn, {
    seed,
    maxSteps: maxSteps || 1000,
    nudge,
    captureFrames,
    poseStride: 1,
    envParams,
    planck: self.planck,
  });
  if (includeActs) ro.stepActivations = stepActs;
  return ro;
}

async function evaluateLunarLander(genome, seeds, suppliedBaseModel = null, envParams = null) {
  const spec = taskSpecs.lunarlander || { max_steps: 1000 };
  const maxSteps = spec.max_steps || 1000;
  const effSeeds = Array.isArray(seeds) && seeds.length ? seeds : (spec.eval_seeds || [0, 1, 2]);
  const rets = [];
  let sumRet = 0.0;
  let minRet = Infinity;
  let maxRet = -Infinity;
  let sumSteps = 0;
  let landedCount = 0;
  let onPadCount = 0;
  let crashedCount = 0;
  let primaryRollout = null;

  for (let i = 0; i < effSeeds.length; i++) {
    const ro = await lunarlanderRollout(
      genome,
      effSeeds[i],
      maxSteps,
      null,
      false,
      i === 0,
      suppliedBaseModel,
      envParams
    );
    if (i === 0) primaryRollout = ro;
    rets.push(Math.round(ro.return * 10) / 10);
    sumRet += ro.return;
    if (ro.return < minRet) minRet = ro.return;
    if (ro.return > maxRet) maxRet = ro.return;
    sumSteps += ro.steps;
    if (ro.landed) landedCount += 1;
    if (ro.landed && ro.on_pad) onPadCount += 1;
    if (ro.crashed) crashedCount += 1;
  }

  const n = Math.max(1, effSeeds.length);
  const meanRet = Math.round((sumRet / n) * 10) / 10;
  const clampedMin = minRet === Infinity ? 0.0 : Math.round(minRet * 10) / 10;
  const clampedMax = maxRet === -Infinity ? 0.0 : Math.round(maxRet * 10) / 10;
  const meanSteps = Math.round((sumSteps / n) * 10) / 10;
  const solved = meanRet >= LL_SOLVED_RETURN && landedCount >= Math.max(1, Math.floor((n * 2) / 3));
  const tier = solved ? "solved" : (meanRet >= LL_TIER_MID_RETURN || landedCount >= 1 ? "mid" : "");
  const disp = `${meanRet >= 0 ? "+" : ""}${Math.round(meanRet)}`;

  return {
    summary: {
      task: "lunarlander",
      seeds: effSeeds,
      returns_per_seed: rets,
      return: meanRet,
      min_return: clampedMin,
      max_return: clampedMax,
      mean_steps: meanSteps,
      landed_count: landedCount,
      on_pad_count: onPadCount,
      crashed_count: crashedCount,
      solved,
      primary_value: meanRet,
      primary_display: `${disp} (${landedCount}/${n} landed)`,
      badge_text: `${disp} \u00b7 ${landedCount}/${n}`,
      tier,
      secondary_text: `mean ${meanRet >= 0 ? "+" : ""}${meanRet.toFixed(1)} (min ${clampedMin >= 0 ? "+" : ""}${clampedMin.toFixed(1)}) \u00b7 ${landedCount}/${n} landed (${onPadCount} on pad) \u00b7 ${Math.round(meanSteps)} steps`,
    },
    traj_sample: primaryRollout ? primaryRollout.traj_sample : null,
    primary_rollout: primaryRollout,
  };
}

// --- CartPole physics ---
const CP_GRAV = 9.8;
const CP_MC = 1.0;
const CP_MP = 0.1;
const CP_TOTAL = CP_MC + CP_MP;
const CP_HALF_LEN = 0.5;
const CP_PML = CP_MP * CP_HALF_LEN;
const CP_FORCE = 10.0;
const CP_TAU = 0.02;
const CP_THETA_TH = (12.0 * 2.0 * Math.PI) / 360.0;
const CP_X_TH = 2.4;

function cartpoleReset(seed) {
  const rng = mulberry32(seed);
  return [
    (rng() * 2.0 - 1.0) * 0.05,
    (rng() * 2.0 - 1.0) * 0.05,
    (rng() * 2.0 - 1.0) * 0.05,
    (rng() * 2.0 - 1.0) * 0.05,
  ];
}

function cartpoleStep(state, uRaw) {
  const [x, xDot, theta, thetaDot] = state;
  const u = Math.max(-1.0, Math.min(1.0, uRaw));
  const force = u * CP_FORCE;
  const ct = Math.cos(theta);
  const st = Math.sin(theta);
  const temp = (force + CP_PML * thetaDot * thetaDot * st) / CP_TOTAL;
  const thetaacc =
    (CP_GRAV * st - ct * temp) /
    (CP_HALF_LEN * (4.0 / 3.0 - (CP_MP * ct * ct) / CP_TOTAL));
  const xacc = temp - (CP_PML * thetaacc * ct) / CP_TOTAL;
  const nx = x + CP_TAU * xDot;
  const nxDot = xDot + CP_TAU * xacc;
  const ntheta = theta + CP_TAU * thetaDot;
  const nthetaDot = thetaDot + CP_TAU * thetaacc;
  const terminated = nx < -CP_X_TH || nx > CP_X_TH || ntheta < -CP_THETA_TH || ntheta > CP_THETA_TH;
  return { nextState: [nx, nxDot, ntheta, nthetaDot], force, terminated };
}

function cartpoleRollout(genome, seed = 0, maxSteps = 500, nudge = null, includeActs = false, captureFrames = true) {
  const spec = taskSpecs.cartpole || { obs_dim: 4, hidden_dims: [8], action_dim: 1, layer_coords: null };
  const dims = [spec.obs_dim || 4, ...(spec.hidden_dims || [8]), spec.action_dim || 1];
  const decoded = decodeHyperNeat(genome, dims, spec.layer_coords);
  let state = cartpoleReset(seed);
  if (nudge === "left" || nudge === "right") {
    const sign = nudge === "left" ? -1 : 1;
    state = [state[0], state[1] + sign * 0.35, state[2], state[3] + sign * 0.55];
  }
  const states = captureFrames ? [state] : null;
  const forces = captureFrames ? [] : null;
  const stepActs = includeActs ? [] : null;
  let absXSum = 0.0;
  let steps = 0;
  for (let s = 0; s < maxSteps; s++) {
    const obs = [state[0] / CP_X_TH, state[1] / 3.0, state[2] / CP_THETA_TH, state[3] / 3.0];
    const { action, activations } = forwardPolicy(decoded, obs, includeActs);
    if (includeActs) stepActs.push(activations);
    const res = cartpoleStep(state, action[0]);
    absXSum += Math.abs(state[0]);
    steps++;
    if (captureFrames) {
      forces.push(res.force);
      states.push(res.nextState);
    }
    state = res.nextState;
    if (res.terminated) break;
  }
  return {
    seed,
    steps,
    return: steps,
    mean_abs_x: absXSum / Math.max(1, steps),
    states,
    forces,
    stepActivations: stepActs,
  };
}

function evaluateCartpole(genome, seeds) {
  const spec = taskSpecs.cartpole || { max_steps: 500 };
  const maxSteps = spec.max_steps || 500;
  let sumSteps = 0;
  let minSteps = maxSteps;
  let sumAbsX = 0;
  let primaryRollout = null;
  const perSeed = [];
  for (let i = 0; i < seeds.length; i++) {
    const ro = cartpoleRollout(genome, seeds[i], maxSteps, null, false, i === 0);
    if (i === 0) primaryRollout = ro;
    perSeed.push(ro.steps);
    sumSteps += ro.steps;
    if (ro.steps < minSteps) minSteps = ro.steps;
    sumAbsX += ro.mean_abs_x;
  }
  const meanSteps = Math.round((sumSteps / Math.max(1, seeds.length)) * 10) / 10;
  const solved = minSteps >= 490;
  const valTxt = Number.isInteger(meanSteps) ? String(meanSteps) : meanSteps.toFixed(1).replace(/\.0$/, "");
  return {
    summary: {
      task: "cartpole",
      seeds,
      steps_per_seed: perSeed,
      mean_steps: meanSteps,
      min_steps: minSteps,
      mean_abs_x: Math.round((sumAbsX / Math.max(1, seeds.length)) * 1000) / 1000,
      solved,
      primary_value: meanSteps,
      primary_display: `${valTxt}/${maxSteps}`,
      badge_text: `${valTxt}/${maxSteps}`,
      tier: solved ? "solved" : (meanSteps >= 150 ? "mid" : ""),
      secondary_text: `min ${minSteps}/${maxSteps}, |x|=${(sumAbsX / Math.max(1, seeds.length)).toFixed(2)}m`,
    },
    primary_rollout: primaryRollout,
  };
}

// --- HalfCheetah (MuJoCo WASM) ---
const HC_FRAME_SKIP = 5;
const HC_DT = 0.01 * HC_FRAME_SKIP;
const HC_PHASE_FREQ = 2.0;
const HC_OBS_SCALE = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 2.0, 5.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0];

let cheetahGeomSpecs = null;
let cheetahNonPlaneIds = null;

async function ensureMujocoCheetah() {
  if (cheetahModel && cheetahData) {
    return {
      mujoco: await ensureMujocoWasm(),
      model: cheetahModel,
      data: cheetahData,
      geom_specs: cheetahGeomSpecs,
      nonPlaneIds: cheetahNonPlaneIds,
    };
  }
  if (!mujocoWasmUrl || !cheetahXmlUrl) {
    throw new Error("MuJoCo WASM URL or XML URL not configured.");
  }
  const absXmlUrl = new URL(cheetahXmlUrl, self.location.href).href;
  const xmlTextPromise = fetch(absXmlUrl, { cache: "force-cache" }).then((resp) => {
    if (!resp.ok) throw new Error(`Failed to load HalfCheetah XML (${resp.status})`);
    return resp.text();
  });
  const [mujoco, xmlText] = await Promise.all([ensureMujocoWasm(), xmlTextPromise]);
  if (!cheetahModel) {
    cheetahModel = mujoco.MjModel.from_xml_string(xmlText);
    cheetahData = new mujoco.MjData(cheetahModel);
    const specs = [];
    const ids = [];
    for (let gid = 0; gid < cheetahModel.ngeom; gid++) {
      const gtype = Number(cheetahModel.geom_type[gid]);
      if (gtype === 0) continue;
      ids.push(gid);
      specs.push({
        type: gtype,
        size: [
          Math.round(cheetahModel.geom_size[gid * 3] * 10000) / 10000,
          Math.round(cheetahModel.geom_size[gid * 3 + 1] * 10000) / 10000,
          Math.round(cheetahModel.geom_size[gid * 3 + 2] * 10000) / 10000,
        ],
        rgba: [
          Math.round(cheetahModel.geom_rgba[gid * 4] * 1000) / 1000,
          Math.round(cheetahModel.geom_rgba[gid * 4 + 1] * 1000) / 1000,
          Math.round(cheetahModel.geom_rgba[gid * 4 + 2] * 1000) / 1000,
          Math.round(cheetahModel.geom_rgba[gid * 4 + 3] * 1000) / 1000,
        ],
      });
    }
    cheetahGeomSpecs = specs;
    cheetahNonPlaneIds = ids;
  }
  return {
    mujoco,
    model: cheetahModel,
    data: cheetahData,
    geom_specs: cheetahGeomSpecs,
    nonPlaneIds: cheetahNonPlaneIds,
  };
}

const _hcObsBuf = new Float64Array(21);

function cheetahObs(qpos, qvel, stepIdx) {
  const obs = _hcObsBuf;
  obs[0] = (qpos[1] - 0.7) / HC_OBS_SCALE[0];
  for (let i = 1; i < 8; i++) obs[i] = qpos[i + 1] / HC_OBS_SCALE[i];
  for (let i = 0; i < 9; i++) obs[8 + i] = qvel[i] / HC_OBS_SCALE[8 + i];
  const ang = 2.0 * Math.PI * HC_PHASE_FREQ * stepIdx * HC_DT;
  obs[17] = Math.sin(ang);
  obs[18] = Math.cos(ang);
  obs[19] = Math.sin(2.0 * ang);
  obs[20] = Math.cos(2.0 * ang);
  return obs;
}

async function halfcheetahRollout(
  genome,
  seed = 0,
  maxSteps = 200,
  nudge = null,
  includeActs = false,
  includeGeoms = false,
  captureFrames = true
) {
  const { mujoco, model, data, geom_specs: geomSpecs, nonPlaneIds } = await ensureMujocoCheetah();
  const spec = taskSpecs.halfcheetah || {
    obs_dim: 21,
    hidden_dims: [12],
    action_dim: 6,
    layer_coords: null,
  };
  const dims = [spec.obs_dim || 21, ...(spec.hidden_dims || [12]), spec.action_dim || 6];
  const decoded = decodeHyperNeat(genome, dims, spec.layer_coords);

  mujoco.mj_resetData(model, data);
  let qpos = data.qpos;
  let qvel = data.qvel;
  let ctrl = data.ctrl;
  const rng = mulberry32(seed);
  for (let i = 0; i < model.nq; i++) qpos[i] += (rng() * 2.0 - 1.0) * 0.05;
  for (let i = 0; i < model.nv; i++) qvel[i] += (rng() * 2.0 - 1.0) * 0.05;
  mujoco.mj_forward(model, data);
  if (qpos.byteLength === 0) {
    qpos = data.qpos;
    qvel = data.qvel;
    ctrl = data.ctrl;
  }

  if (nudge === "back") qvel[0] -= 2.2;
  else if (nudge === "fwd") qvel[0] += 2.2;
  else if (nudge === "hop") qvel[1] += 2.6;

  const startX = qpos[0];
  const capFrame = (ctrlRms) => [
    Math.round(qpos[0] * 10000) / 10000,
    Math.round(qpos[1] * 10000) / 10000,
    Math.round(qpos[2] * 10000) / 10000,
    Math.round(qpos[3] * 10000) / 10000,
    Math.round(qpos[4] * 10000) / 10000,
    Math.round(qpos[5] * 10000) / 10000,
    Math.round(qpos[6] * 10000) / 10000,
    Math.round(qpos[7] * 10000) / 10000,
    Math.round(qpos[8] * 10000) / 10000,
    Math.round(ctrlRms * 1000) / 1000,
    0.0,
    0.0,
  ];

  const capGeoms = () => {
    const row = new Array(nonPlaneIds.length * 12);
    const gp = data.geom_xpos;
    const gm = data.geom_xmat;
    for (let k = 0; k < nonPlaneIds.length; k++) {
      const gid = nonPlaneIds[k];
      const baseP = gid * 3;
      const baseM = gid * 9;
      const outO = k * 12;
      row[outO] = Math.round(gp[baseP] * 10000) / 10000;
      row[outO + 1] = Math.round(gp[baseP + 1] * 10000) / 10000;
      row[outO + 2] = Math.round(gp[baseP + 2] * 10000) / 10000;
      for (let m = 0; m < 9; m++) {
        row[outO + 3 + m] = Math.round(gm[baseM + m] * 10000) / 10000;
      }
    }
    return row;
  };

  const frames = captureFrames ? [capFrame(0.0)] : null;
  const vxTrace = captureFrames ? [Math.round(qvel[0] * 1000) / 1000] : null;
  const stepActs = includeActs ? [] : null;
  const geomPoses = includeGeoms ? [capGeoms()] : null;
  let totalReturn = 0.0;
  let uprightSum = 0.0;
  let energySum = 0.0;

  for (let s = 0; s < maxSteps; s++) {
    const obs = cheetahObs(qpos, qvel, s);
    const { action, activations } = forwardPolicy(decoded, obs, includeActs);
    if (includeActs) stepActs.push(activations);
    const xBefore = qpos[0];
    ctrl[0] = action[0];
    ctrl[1] = action[1];
    ctrl[2] = action[2];
    ctrl[3] = action[3];
    ctrl[4] = action[4];
    ctrl[5] = action[5];
    for (let sub = 0; sub < HC_FRAME_SKIP; sub++) mujoco.mj_step(model, data);
    if (qpos.byteLength === 0) {
      qpos = data.qpos;
      qvel = data.qvel;
      ctrl = data.ctrl;
    }
    const xAfter = qpos[0];
    const vx = (xAfter - xBefore) / HC_DT;
    const actSqSum =
      action[0] * action[0] +
      action[1] * action[1] +
      action[2] * action[2] +
      action[3] * action[3] +
      action[4] * action[4] +
      action[5] * action[5];
    const ctrlSqMean = actSqSum / 6.0;
    totalReturn += vx - 0.1 * actSqSum;
    uprightSum += Math.cos(qpos[2]);
    energySum += ctrlSqMean;
    if (captureFrames) {
      vxTrace.push(Math.round(vx * 1000) / 1000);
      frames.push(capFrame(Math.sqrt(ctrlSqMean)));
    }
    if (includeGeoms) geomPoses.push(capGeoms());
  }

  const steps = captureFrames ? frames.length - 1 : maxSteps;
  const dx = qpos[0] - startX;
  const meanVx = dx / Math.max(HC_DT, steps * HC_DT);
  const meanUpright = uprightSum / Math.max(1, steps);
  const result = {
    seed,
    steps,
    distance: dx,
    mean_vx: meanVx,
    mean_upright: meanUpright,
    mean_energy: energySum / Math.max(1, steps),
    return: totalReturn,
    frames,
    vx_trace: vxTrace,
    stepActivations: stepActs,
  };
  if (includeGeoms) {
    result.geom_specs = geomSpecs;
    result.geom_poses = geomPoses;
  }
  return result;
}

async function evaluateHalfCheetah(genome, seeds) {
  const spec = taskSpecs.halfcheetah || { max_steps: 200 };
  const maxSteps = spec.max_steps || 200;
  const dists = [];
  let sumDx = 0.0;
  let minDx = Infinity;
  let sumVx = 0.0;
  let sumUp = 0.0;
  let sumRet = 0.0;
  let primaryRollout = null;

  for (let i = 0; i < seeds.length; i++) {
    const ro = await halfcheetahRollout(genome, seeds[i], maxSteps, null, false, false, i === 0);
    if (i === 0) primaryRollout = ro;
    const dx = Math.round(ro.distance * 100) / 100;
    dists.push(dx);
    sumDx += ro.distance;
    if (dx < minDx) minDx = dx;
    sumVx += ro.mean_vx;
    sumUp += ro.mean_upright;
    sumRet += ro.return;
  }

  const n = Math.max(1, seeds.length);
  const meanDx = Math.round((sumDx / n) * 100) / 100;
  const meanVx = Math.round((sumVx / n) * 100) / 100;
  const meanUp = Math.round((sumUp / n) * 100) / 100;
  const solved = meanDx >= 8.0 && meanUp >= 0.35;
  const tier = solved ? "solved" : (meanDx >= 2.5 && meanUp >= 0.2 ? "mid" : "");
  return {
    summary: {
      task: "halfcheetah",
      seeds,
      distances_per_seed: dists,
      mean_distance: meanDx,
      min_distance: Math.round((minDx === Infinity ? 0 : minDx) * 100) / 100,
      mean_vx: meanVx,
      mean_upright: meanUp,
      return: Math.round((sumRet / n) * 10) / 10,
      solved,
      primary_value: meanDx,
      primary_display: `${meanDx >= 0 ? "+" : ""}${meanDx.toFixed(1)}m`,
      badge_text: `${meanDx >= 0 ? "+" : ""}${meanDx.toFixed(1)}m`,
      tier,
      secondary_text: `vx=${meanVx >= 0 ? "+" : ""}${meanVx.toFixed(2)}m/s, up=${meanUp >= 0 ? "+" : ""}${meanUp.toFixed(2)}`,
    },
    primary_rollout: primaryRollout,
  };
}

const SIM_TASK_HANDLERS = {
  microduck: {
    wasm: true,
    evaluate: (g, seeds, bm, pb, ep) => evaluateMicroduck(g, seeds, bm, pb, ep),
    rollout: (msg) =>
      microduckRollout(
        msg.genome,
        msg.seed,
        msg.max_steps || 400,
        msg.nudge || null,
        Boolean(msg.includeActivations),
        true,
        1,
        msg.baseModel || null,
        msg.physicsBuffer || null,
        msg.env_params || null
      ),
  },
  unitree_go2: {
    wasm: true,
    evaluate: (g, seeds, bm, pb, ep) => evaluateUnitreeGo2(g, seeds, bm, pb, ep),
    rollout: (msg) =>
      unitreeGo2Rollout(
        msg.genome,
        msg.seed,
        msg.max_steps || 400,
        msg.nudge || null,
        Boolean(msg.includeActivations),
        true,
        1,
        msg.baseModel || null,
        msg.physicsBuffer || null,
        msg.env_params || null
      ),
  },
  so101: {
    wasm: true,
    evaluate: (g, seeds, bm, pb, ep) => evaluateSo101(g, seeds, bm, pb, ep),
    rollout: (msg) =>
      so101Rollout(
        msg.genome,
        msg.seed,
        msg.max_steps || 250,
        msg.nudge || null,
        Boolean(msg.includeActivations),
        true,
        1,
        msg.baseModel || null,
        msg.physicsBuffer || null,
        msg.env_params || null
      ),
  },
  halfcheetah: {
    wasm: true,
    evaluate: (g, seeds) => evaluateHalfCheetah(g, seeds),
    rollout: (msg) =>
      halfcheetahRollout(
        msg.genome,
        msg.seed,
        msg.max_steps || 200,
        msg.nudge || null,
        Boolean(msg.includeActivations),
        msg.include_geoms !== false
      ),
  },
  cartpole: {
    wasm: false,
    evaluate: (g, seeds) => evaluateCartpole(g, seeds),
    rollout: (msg) =>
      cartpoleRollout(msg.genome, msg.seed, msg.max_steps || 500, msg.nudge || null, Boolean(msg.includeActivations)),
  },
  lunarlander: {
    wasm: false,
    evaluate: (g, seeds, baseModel, _physBuf, envParams) =>
      evaluateLunarLander(g, seeds, baseModel || null, envParams || null),
    rollout: (msg) =>
      lunarlanderRollout(
        msg.genome,
        msg.seed,
        msg.max_steps || 1000,
        msg.nudge || null,
        Boolean(msg.includeActivations),
        true,
        msg.baseModel || null,
        msg.env_params || null
      ),
  },
};

function getSimHandler(taskId) {
  return SIM_TASK_HANDLERS[taskId] || SIM_TASK_HANDLERS.cartpole;
}

function isWasmTask(taskId) {
  if (SIM_TASK_HANDLERS[taskId]?.wasm) return true;
  const spec = taskSpecs[taskId];
  return Boolean(spec && (spec.scene_id || spec.mjcf_asset));
}

self.addEventListener("message", async (event) => {
  const msg = event.data || {};
  const { reqId, type } = msg;
  try {
    if (type === "init") {
      taskSpecs = msg.task_specs || {};
      basePath = msg.basePath || "";
      mujocoWasmUrl = msg.mujoco_wasm_url || "";
      cheetahXmlUrl = msg.cheetahXmlUrl || "";
      if (msg.warmMujoco && mujocoWasmUrl) {
        const warmSpec = msg.warmTaskId ? taskSpecs[msg.warmTaskId] : null;
        if (warmSpec && warmSpec.scene_manifest) {
          if (msg.physicsBuffer) {
            const vInfo = resolveSceneTaskConfig(msg.warmTaskId, msg.warmEnvParams || msg.env_params || null);
            await ensureMujocoScene(vInfo.scene_id, vInfo.manifest, msg.physicsBuffer);
          } else {
            await ensureMujocoWasm();
          }
        } else if ((!msg.warmTaskId || msg.warmTaskId === "halfcheetah") && cheetahXmlUrl) {
          await ensureMujocoCheetah();
        } else {
          await ensureMujocoWasm();
        }
      }
      self.postMessage({ reqId, ok: true, type: "init" });
      return;
    }

    if (type === "evalBatch") {
      const { taskId, genomes, seeds, baseModel, physicsBuffer, env_params: envParams } = msg;
      const handler = getSimHandler(taskId);
      const items = [];
      for (let i = 0; i < genomes.length; i++) {
        const g = genomes[i];
        const ev = await handler.evaluate(g, seeds, baseModel || null, physicsBuffer || null, envParams || null);
        items.push({
          index: i,
          fp: genomeFingerprint(g),
          summary: ev.summary,
          traj_sample: ev.traj_sample || null,
          primary_rollout: ev.primary_rollout,
        });
      }
      self.postMessage({
        reqId,
        ok: true,
        type: "evalBatch",
        engine: isWasmTask(taskId) ? "client-wasm" : "client-js",
        items,
      });
      return;
    }

    if (type === "rollout") {
      const ro = await getSimHandler(msg.taskId).rollout(msg);
      self.postMessage({
        reqId,
        ok: true,
        type: "rollout",
        engine: isWasmTask(msg.taskId) ? "client-wasm" : "client-js",
        rollout: ro,
      });
      return;
    }

    if (type === "evalSummary") {
      const { taskId, genome, seeds, baseModel, physicsBuffer, env_params: envParams } = msg;
      const ev = await getSimHandler(taskId).evaluate(genome, seeds, baseModel || null, physicsBuffer || null, envParams || null);
      self.postMessage({
        reqId,
        ok: true,
        type: "evalSummary",
        engine: isWasmTask(taskId) ? "client-wasm" : "client-js",
        summary: ev.summary,
        traj_sample: ev.traj_sample || null,
        primary_rollout: ev.primary_rollout,
      });
      return;
    }
  } catch (err) {
    self.postMessage({
      reqId,
      ok: false,
      type,
      error: err && err.message ? err.message : String(err),
      fallbackToServer: true,
    });
  }
});
