// Dedicated simulation & evaluation Web Worker for direct-evolution-together.
// Runs CartPole (pure JS) and HalfCheetah (official @mujoco/mujoco WASM) off the UI thread.

let taskSpecs = {};
let mujocoWasmUrl = "";
let cheetahXmlUrl = "";

let mujocoPromise = null;
let cheetahModel = null;
let cheetahData = null;

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

function forwardPolicy(decoded, obs, recordActs = false) {
  let h = obs;
  const acts = recordActs ? [Array.from(obs, (v) => Math.round(v * 1000) / 1000)] : null;
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
    if (recordActs) {
      acts.push(Array.from(h, (v) => Math.round(v * 1000) / 1000));
    }
  }
  return { action: h, activations: acts };
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

function cartpoleRollout(genome, seed = 0, maxSteps = 500, nudge = null, includeActs = false) {
  const spec = taskSpecs.cartpole || { obsDim: 4, hiddenDims: [8], actionDim: 1, layerCoords: null };
  const dims = [spec.obsDim || 4, ...(spec.hiddenDims || [8]), spec.actionDim || 1];
  const decoded = decodeHyperNeat(genome, dims, spec.layerCoords);
  let state = cartpoleReset(seed);
  if (nudge === "left" || nudge === "right") {
    const sign = nudge === "left" ? -1 : 1;
    state = [state[0], state[1] + sign * 0.35, state[2], state[3] + sign * 0.55];
  }
  const states = [state];
  const forces = [];
  const stepActs = includeActs ? [] : null;
  let absXSum = 0.0;
  for (let s = 0; s < maxSteps; s++) {
    const obs = [state[0] / CP_X_TH, state[1] / 3.0, state[2] / CP_THETA_TH, state[3] / 3.0];
    const { action, activations } = forwardPolicy(decoded, obs, includeActs);
    if (includeActs) stepActs.push(activations);
    const res = cartpoleStep(state, action[0]);
    absXSum += Math.abs(state[0]);
    forces.push(res.force);
    states.push(res.nextState);
    state = res.nextState;
    if (res.terminated) break;
  }
  const steps = forces.length;
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
  const spec = taskSpecs.cartpole || { maxSteps: 500 };
  const maxSteps = spec.maxSteps || 500;
  let sumSteps = 0;
  let minSteps = maxSteps;
  let sumAbsX = 0;
  let primaryRollout = null;
  const perSeed = [];
  for (let i = 0; i < seeds.length; i++) {
    const ro = cartpoleRollout(genome, seeds[i], maxSteps);
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
      stepsPerSeed: perSeed,
      meanSteps,
      minSteps,
      meanAbsX: Math.round((sumAbsX / Math.max(1, seeds.length)) * 1000) / 1000,
      solved,
      primaryValue: meanSteps,
      primaryDisplay: `${valTxt}/${maxSteps}`,
      badgeText: `${valTxt}/${maxSteps}`,
      tier: solved ? "solved" : (meanSteps >= 150 ? "mid" : ""),
      secondaryText: `min ${minSteps}/${maxSteps}, |x|=${(sumAbsX / Math.max(1, seeds.length)).toFixed(2)}m`,
    },
    primaryRollout,
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
      mujoco: await mujocoPromise,
      model: cheetahModel,
      data: cheetahData,
      geomSpecs: cheetahGeomSpecs,
      nonPlaneIds: cheetahNonPlaneIds,
    };
  }
  if (!mujocoWasmUrl || !cheetahXmlUrl) {
    throw new Error("MuJoCo WASM URL or XML URL not configured.");
  }
  const absWasmUrl = new URL(mujocoWasmUrl, self.location.href).href;
  const absXmlUrl = new URL(cheetahXmlUrl, self.location.href).href;
  if (!mujocoPromise) {
    mujocoPromise = (async () => {
      const mod = await import(absWasmUrl);
      const loadFn = mod.default || mod;
      return await loadFn();
    })();
  }
  const mujoco = await mujocoPromise;
  if (!cheetahModel) {
    const resp = await fetch(absXmlUrl, { cache: "force-cache" });
    if (!resp.ok) throw new Error(`Failed to load HalfCheetah XML (${resp.status})`);
    const xmlText = await resp.text();
    cheetahModel = mujoco.MjModel.from_xml_string(xmlText);
    cheetahData = new mujoco.MjData(cheetahModel);
    const specs = [];
    const ids = [];
    for (let gid = 0; gid < cheetahModel.ngeom; gid++) {
      const gtype = Number(cheetahModel.geom_type[gid]);
      if (gtype === 0) continue; // skip mjGEOM_PLANE
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
    geomSpecs: cheetahGeomSpecs,
    nonPlaneIds: cheetahNonPlaneIds,
  };
}

function cheetahObs(data, stepIdx) {
  const obs = new Float64Array(21);
  obs[0] = (data.qpos[1] - 0.7) / HC_OBS_SCALE[0];
  for (let i = 1; i < 8; i++) obs[i] = data.qpos[i + 1] / HC_OBS_SCALE[i];
  for (let i = 0; i < 9; i++) obs[8 + i] = data.qvel[i] / HC_OBS_SCALE[8 + i];
  const t = stepIdx * HC_DT;
  const ang = 2.0 * Math.PI * HC_PHASE_FREQ * t;
  obs[17] = Math.sin(ang);
  obs[18] = Math.cos(ang);
  obs[19] = Math.sin(2.0 * ang);
  obs[20] = Math.cos(2.0 * ang);
  return obs;
}

async function halfcheetahRollout(genome, seed = 0, maxSteps = 200, nudge = null, includeActs = false, includeGeoms = false) {
  const { mujoco, model, data, geomSpecs, nonPlaneIds } = await ensureMujocoCheetah();
  const spec = taskSpecs.halfcheetah || {
    obsDim: 21,
    hiddenDims: [12],
    actionDim: 6,
    layerCoords: null,
  };
  const dims = [spec.obsDim || 21, ...(spec.hiddenDims || [12]), spec.actionDim || 6];
  const decoded = decodeHyperNeat(genome, dims, spec.layerCoords);

  mujoco.mj_resetData(model, data);
  const rng = mulberry32(seed);
  for (let i = 0; i < model.nq; i++) data.qpos[i] += (rng() * 2.0 - 1.0) * 0.05;
  for (let i = 0; i < model.nv; i++) data.qvel[i] += (rng() * 2.0 - 1.0) * 0.05;
  mujoco.mj_forward(model, data);

  if (nudge === "back") data.qvel[0] -= 2.2;
  else if (nudge === "fwd") data.qvel[0] += 2.2;
  else if (nudge === "hop") data.qvel[1] += 2.6;

  const startX = data.qpos[0];
  const capFrame = (ctrlRms) => {
    const q = data.qpos;
    return [
      Math.round(q[0] * 10000) / 10000,
      Math.round(q[1] * 10000) / 10000,
      Math.round(q[2] * 10000) / 10000,
      Math.round(q[3] * 10000) / 10000,
      Math.round(q[4] * 10000) / 10000,
      Math.round(q[5] * 10000) / 10000,
      Math.round(q[6] * 10000) / 10000,
      Math.round(q[7] * 10000) / 10000,
      Math.round(q[8] * 10000) / 10000,
      Math.round(ctrlRms * 1000) / 1000,
      0.0,
      0.0,
    ];
  };

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

  const frames = [capFrame(0.0)];
  const vxTrace = [Math.round(data.qvel[0] * 1000) / 1000];
  const stepActs = includeActs ? [] : null;
  const geomPoses = includeGeoms ? [capGeoms()] : null;
  let totalReturn = 0.0;
  let uprightSum = 0.0;
  let energySum = 0.0;

  for (let s = 0; s < maxSteps; s++) {
    const obs = cheetahObs(data, s);
    const { action, activations } = forwardPolicy(decoded, obs, includeActs);
    if (includeActs) stepActs.push(activations);
    const xBefore = data.qpos[0];
    for (let a = 0; a < 6; a++) data.ctrl[a] = action[a];
    for (let sub = 0; sub < HC_FRAME_SKIP; sub++) mujoco.mj_step(model, data);
    const xAfter = data.qpos[0];
    const vx = (xAfter - xBefore) / HC_DT;
    let actSqSum = 0.0;
    for (let a = 0; a < 6; a++) actSqSum += action[a] * action[a];
    const ctrlSqMean = actSqSum / 6.0;
    totalReturn += vx - 0.1 * actSqSum;
    uprightSum += Math.cos(data.qpos[2]);
    energySum += ctrlSqMean;
    vxTrace.push(Math.round(vx * 1000) / 1000);
    frames.push(capFrame(Math.sqrt(ctrlSqMean)));
    if (includeGeoms) geomPoses.push(capGeoms());
  }

  const steps = frames.length - 1;
  const dx = data.qpos[0] - startX;
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
    result.geomSpecs = geomSpecs;
    result.geomPoses = geomPoses;
  }
  return result;
}

async function evaluateHalfCheetah(genome, seeds) {
  const spec = taskSpecs.halfcheetah || { maxSteps: 200 };
  const maxSteps = spec.maxSteps || 200;
  const dists = [];
  let sumDx = 0.0;
  let minDx = Infinity;
  let sumVx = 0.0;
  let sumUp = 0.0;
  let sumRet = 0.0;
  let primaryRollout = null;

  for (let i = 0; i < seeds.length; i++) {
    const ro = await halfcheetahRollout(genome, seeds[i], maxSteps);
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
      distancesPerSeed: dists,
      meanDistance: meanDx,
      minDistance: Math.round((minDx === Infinity ? 0 : minDx) * 100) / 100,
      meanVx,
      meanUpright: meanUp,
      return: Math.round((sumRet / n) * 10) / 10,
      solved,
      primaryValue: meanDx,
      primaryDisplay: `${meanDx >= 0 ? "+" : ""}${meanDx.toFixed(1)}m`,
      badgeText: `${meanDx >= 0 ? "+" : ""}${meanDx.toFixed(1)}m`,
      tier,
      secondaryText: `vx=${meanVx >= 0 ? "+" : ""}${meanVx.toFixed(2)}m/s, up=${meanUp >= 0 ? "+" : ""}${meanUp.toFixed(2)}`,
    },
    primaryRollout,
  };
}

self.addEventListener("message", async (event) => {
  const msg = event.data || {};
  const { reqId, type } = msg;
  try {
    if (type === "init") {
      taskSpecs = msg.taskSpecs || {};
      mujocoWasmUrl = msg.mujocoWasmUrl || "";
      cheetahXmlUrl = msg.cheetahXmlUrl || "";
      self.postMessage({ reqId, ok: true, type: "init" });
      return;
    }

    if (type === "evalBatch") {
      const { taskId, genomes, seeds } = msg;
      const items = [];
      for (let i = 0; i < genomes.length; i++) {
        const g = genomes[i];
        const fp = genomeFingerprint(g);
        const ev =
          taskId === "halfcheetah"
            ? await evaluateHalfCheetah(g, seeds)
            : evaluateCartpole(g, seeds);
        items.push({ index: i, fp, summary: ev.summary, primaryRollout: ev.primaryRollout });
      }
      self.postMessage({
        reqId,
        ok: true,
        type: "evalBatch",
        engine: taskId === "halfcheetah" ? "client-wasm" : "client-js",
        items,
      });
      return;
    }

    if (type === "rollout") {
      const { taskId, genome, seed, maxSteps, nudge, includeActivations, includeGeoms } = msg;
      const ro =
        taskId === "halfcheetah"
          ? await halfcheetahRollout(genome, seed, maxSteps || 200, nudge || null, Boolean(includeActivations), includeGeoms !== false)
          : cartpoleRollout(genome, seed, maxSteps || 500, nudge || null, Boolean(includeActivations));
      self.postMessage({
        reqId,
        ok: true,
        type: "rollout",
        engine: taskId === "halfcheetah" ? "client-wasm" : "client-js",
        rollout: ro,
      });
      return;
    }

    if (type === "evalSummary") {
      const { taskId, genome, seeds } = msg;
      const ev =
        taskId === "halfcheetah"
          ? await evaluateHalfCheetah(genome, seeds)
          : evaluateCartpole(genome, seeds);
      self.postMessage({
        reqId,
        ok: true,
        type: "evalSummary",
        engine: taskId === "halfcheetah" ? "client-wasm" : "client-js",
        summary: ev.summary,
        primaryRollout: ev.primaryRollout,
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
