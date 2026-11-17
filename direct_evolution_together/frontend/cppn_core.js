// Shared CPPN evaluator, HyperNEAT weight decoder, and canonical genome hashing.
(function (global) {
  "use strict";

  const ACTIVATION_META = [
    { code: 0, name: "tanh", short: "tanh", color: "#38bdf8" },
    { code: 1, name: "sin(1.5z)", short: "sin", color: "#c084fc" },
    { code: 2, name: "exp(-z^2)", short: "gauss", color: "#34d399" },
    { code: 3, name: "|z|-0.5", short: "abs", color: "#f5c451" },
    { code: 4, name: "lrelu(0.1)", short: "lrelu", color: "#fb7185" },
  ];

  const ACT_BY_NAME = {
    tanh: ACTIVATION_META[0],
    sin: ACTIVATION_META[1],
    "sin(1.5z)": ACTIVATION_META[1],
    gauss: ACTIVATION_META[2],
    gaussian: ACTIVATION_META[2],
    "exp(-z^2)": ACTIVATION_META[2],
    abs: ACTIVATION_META[3],
    "|z|-0.5": ACTIVATION_META[3],
    lrelu: ACTIVATION_META[4],
    leaky_relu: ACTIVATION_META[4],
    relu: { code: 5, name: "relu", short: "relu", color: "#fb923c" },
    sigmoid: { code: 6, name: "sigmoid", short: "sigm", color: "#a78bfa" },
    linear: { code: 7, name: "linear", short: "lin", color: "#94a3b8" },
    identity: { code: 7, name: "linear", short: "lin", color: "#94a3b8" },
  };

  function getActMeta(codeOrName) {
    if (typeof codeOrName === "number") {
      return ACTIVATION_META[codeOrName] || ACTIVATION_META[4];
    }
    const key = String(codeOrName || "tanh").toLowerCase().trim();
    return ACT_BY_NAME[key] || { code: -1, name: key, short: key.slice(0, 5), color: "#94a3b8" };
  }

  function applyActivation(z, code) {
    switch (code) {
      case 0:
        return Math.tanh(z);
      case 1:
        return Math.sin(z * 1.5);
      case 2:
        return Math.exp(-z * z);
      case 3:
        return Math.abs(z) - 0.5;
      default:
        return z > 0 ? z : 0.1 * z;
    }
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

  const genomeFingerprints = new WeakMap();

  function genomeFingerprint(genome) {
    if (!genome) return "empty";
    const memo = genomeFingerprints.get(genome);
    if (memo) return memo;
    let fp;
    if (Array.isArray(genome.nodes)) {
      fp = "neat:" + JSON.stringify([genome.nodes.length, (genome.edges || genome.connections || []).length]);
    } else {
      const w = Array.isArray(genome.weights) ? genome.weights : [];
      const b = Array.isArray(genome.biases) ? genome.biases : [];
      const a = Array.isArray(genome.activations) ? genome.activations : [];
      const text = w.join(",") + "|" + b.join(",") + "|" + a.join(",");
      fp =
        hashText(text, 0x811c9dc5).toString(16).padStart(8, "0") +
        hashText(text, 0x9e3779b9).toString(16).padStart(8, "0");
    }
    genomeFingerprints.set(genome, fp);
    return fp;
  }

  const genomeShaCache = new WeakMap();

  /** Compute 20-char SHA-256 genome hash matching Python `genome_content_hash`. */
  async function genomeSha256Hex(genome) {
    if (!genome) return "";
    if (genome.hash) return String(genome.hash);
    if (genomeShaCache.has(genome)) return genomeShaCache.get(genome);
    const w = (genome.weights || []).map((v) => Number(v).toFixed(4)).join(",");
    const b = (genome.biases || []).map((v) => Number(v).toFixed(4)).join(",");
    const a = (genome.activations || []).map((v) => String(Math.trunc(Number(v) || 0))).join(",");
    if (global.crypto && global.crypto.subtle && typeof TextEncoder === "function") {
      try {
        const bytes = new TextEncoder().encode(`${w}|${b}|${a}`);
        const digest = await global.crypto.subtle.digest("SHA-256", bytes);
        const hex = Array.from(new Uint8Array(digest))
          .map((x) => x.toString(16).padStart(2, "0"))
          .join("")
          .slice(0, 20);
        genomeShaCache.set(genome, hex);
        return hex;
      } catch {}
    }
    return genomeFingerprint(genome);
  }

  function evalCppnPoint(genome, x0, x1, x2, x3, h1, h2, h3, outRgb) {
    const w = genome.weights || [];
    const b = genome.biases || [];
    const acts = genome.activations || [];
    for (let j = 0; j < 12; j++) {
      const wo = j * 4;
      const z =
        (w[wo] || 0) * x0 +
        (w[wo + 1] || 0) * x1 +
        (w[wo + 2] || 0) * x2 +
        (w[wo + 3] || 0) * x3 +
        (b[j] || 0);
      h1[j] = applyActivation(z, acts[j]);
    }
    for (let j = 0; j < 12; j++) {
      const wo = 48 + j * 12;
      let z = b[12 + j] || 0;
      for (let i = 0; i < 12; i++) z += (w[wo + i] || 0) * h1[i];
      h2[j] = applyActivation(z, acts[12 + j]);
    }
    for (let j = 0; j < 12; j++) {
      const wo = 192 + j * 12;
      let z = b[24 + j] || 0;
      for (let i = 0; i < 12; i++) z += (w[wo + i] || 0) * h2[i];
      h3[j] = applyActivation(z, acts[24 + j]);
    }
    for (let c = 0; c < 3; c++) {
      const wo = 336 + c * 12;
      let z = b[36 + c] || 0;
      for (let i = 0; i < 12; i++) z += (w[wo + i] || 0) * h3[i];
      outRgb[c] = 0.5 + 0.5 * Math.tanh(z);
    }
  }

  const decodedCache = new Map();

  function decodeHyperNeatWeights(genome, dims, layerCoords) {
    const safeDims = Array.isArray(dims) && dims.length >= 2 ? dims : [4, 6, 1];
    const fp = genomeFingerprint(genome);
    const cacheKey = `${safeDims.join(",")}|${JSON.stringify(layerCoords || [])}|${fp}`;
    if (decodedCache.has(cacheKey)) return decodedCache.get(cacheKey);

    const numLayers = safeDims.length - 1;
    const zs = linspace(-1.0, 1.0, numLayers + 1);
    const layerXs = safeDims.map((d, idx) =>
      Array.isArray(layerCoords) && Array.isArray(layerCoords[idx]) && layerCoords[idx].length === d
        ? layerCoords[idx]
        : linspace(-1.0, 1.0, d)
    );

    const h1 = new Float32Array(12);
    const h2 = new Float32Array(12);
    const h3 = new Float32Array(12);
    const outRgb = new Float32Array(3);

    const layers = [];
    for (let l = 0; l < numLayers; l++) {
      const dIn = safeDims[l];
      const dOut = safeDims[l + 1];
      const zMid = 0.5 * (zs[l] + zs[l + 1]);
      const w = new Float64Array(dIn * dOut);
      let sum = 0.0;
      for (let i = 0; i < dIn; i++) {
        for (let j = 0; j < dOut; j++) {
          evalCppnPoint(genome, layerXs[l][i], layerXs[l + 1][j], zMid, 1.0, h1, h2, h3, outRgb);
          const raw = outRgb[0] * 2.0 - 1.0;
          w[i * dOut + j] = raw;
          sum += raw;
        }
      }
      const n = w.length || 1;
      const mean = sum / n;
      let varSum = 0.0;
      for (let k = 0; k < n; k++) {
        const diff = w[k] - mean;
        varSum += diff * diff;
      }
      const std = Math.sqrt(varSum / n);
      const scale = Math.sqrt(1.0 / dIn) / (std + 1e-6);
      for (let k = 0; k < n; k++) {
        w[k] = (w[k] - mean) * scale;
      }
      const b = new Float64Array(dOut);
      for (let j = 0; j < dOut; j++) {
        evalCppnPoint(genome, 0.0, layerXs[l + 1][j], zs[l + 1], -1.0, h1, h2, h3, outRgb);
        b[j] = (outRgb[0] * 2.0 - 1.0) * 0.1;
      }
      layers.push({ dIn, dOut, w, b });
    }

    const result = { dims: safeDims, layerCoords: layerXs, layers };
    if (decodedCache.size > 400) decodedCache.clear();
    decodedCache.set(cacheKey, result);
    return result;
  }

  function forwardPolicyWithTrace(decoded, obs) {
    let h = obs;
    const activations = [Array.from(obs)];
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
      activations.push(Array.from(next));
    }
    return { action: h, activations };
  }

  const api = {
    ACTIVATION_META,
    getActMeta,
    applyActivation,
    linspace,
    hashText,
    genomeFingerprint,
    genomeSha256Hex,
    evalCppnPoint,
    decodeHyperNeatWeights,
    forwardPolicyWithTrace,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  global.DetCppnCore = api;
})(typeof window !== "undefined" ? window : globalThis);
