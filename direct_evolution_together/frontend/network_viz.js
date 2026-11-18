// Publication-grade interactive Network Structure Visualizer for HyperNEAT & NEAT.
// Supports decoded HyperNEAT substrates, 4->12->12->12->3 CPPN graphs, arbitrary NEAT DAGs,
// diverging weight matrix heatmaps, interactive pruning/path-isolation, and live sim streaming.
(function (global) {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";

  const core =
    global.DetCppnCore ||
    (typeof require === "function" ? require("./cppn_core.js") : null);
  if (!core) {
    throw new Error("DetCppnCore must be loaded before DetNetworkViz");
  }
  const {
    ACTIVATION_META,
    getActMeta,
    applyActivation,
    linspace,
    hashText,
    genomeFingerprint,
    evalCppnPoint,
    decodeHyperNeatWeights,
  } = core;

  function readSpecField(taskSpec, camel, snake) {
    if (!taskSpec) return undefined;
    return taskSpec[camel] !== undefined ? taskSpec[camel] : taskSpec[snake];
  }

  function isControlTaskSpec(taskSpec) {
    if (!taskSpec) return false;
    if (taskSpec.kind) return taskSpec.kind === "control";
    return Boolean(readSpecField(taskSpec, "isControl", "is_control"));
  }

  function readSpecDims(taskSpec) {
    const explicit = readSpecField(taskSpec, "substrateDims", "substrate_dims");
    if (Array.isArray(explicit) && explicit.length >= 2) return explicit;
    const obsDim = Number(readSpecField(taskSpec, "obsDim", "obs_dim") || 0);
    const actDim = Number(readSpecField(taskSpec, "actionDim", "action_dim") || 0);
    const hidden = readSpecField(taskSpec, "hiddenDims", "hidden_dims");
    if (obsDim > 0 && actDim > 0) {
      return [obsDim, ...(Array.isArray(hidden) ? hidden : []), actDim];
    }
    return null;
  }

  function assignCoordinates(layersNodes, dimsInfo) {
    const numLayers = layersNodes.length;
    const maxNodes = Math.max(1, ...layersNodes.map((arr) => arr.length));
    const width = Math.max(780, 176 + (numLayers - 1) * 190);
    const padXLeft = 164;
    const padXRight = 152;
    const innerW = Math.max(240, width - padXLeft - padXRight);
    const minRowPitch = maxNodes > 18 ? 25 : maxNodes > 12 ? 32 : 42;
    const padTop = 58;
    const padBottom = 46;
    const innerH = Math.max(290, (maxNodes - 1) * minRowPitch);
    const height = padTop + innerH + padBottom;

    for (let l = 0; l < numLayers; l++) {
      const arr = layersNodes[l];
      const x = numLayers === 1 ? width * 0.5 : padXLeft + (innerW * l) / (numLayers - 1);
      const count = arr.length;
      const span = Math.min(innerH, Math.max(0, (count - 1) * Math.max(minRowPitch, Math.min(54, innerH / Math.max(1, count - 1)))));
      const startY = padTop + (innerH - span) * 0.5;
      for (let idx = 0; idx < count; idx++) {
        const node = arr[idx];
        node.x = x;
        node.y = count === 1 ? padTop + innerH * 0.5 : startY + (span * idx) / (count - 1);
      }
    }
    return { width, height, layerNames: dimsInfo };
  }

  function buildSubstrateGraph(genome, taskSpec) {
    const dims = readSpecDims(taskSpec) || [4, 8, 1];
    const layerCoords = readSpecField(taskSpec, "layerCoords", "layer_coords") || null;
    const obsLabels = readSpecField(taskSpec, "obsLabels", "obs_labels") || [];
    const actionLabels = readSpecField(taskSpec, "actionLabels", "action_labels") || [];

    const decoded = decodeHyperNeatWeights(genome, dims, layerCoords);
    const numLayers = decoded.dims.length;
    const nodes = [];
    const layersNodes = [];
    const nodeMap = new Map();

    for (let l = 0; l < numLayers; l++) {
      const count = decoded.dims[l];
      const kind = l === 0 ? "input" : l === numLayers - 1 ? "output" : "hidden";
      const arr = [];
      for (let i = 0; i < count; i++) {
        let label = `${kind === "hidden" ? "h" + l + "_" : ""}${i}`;
        if (kind === "input") label = obsLabels[i] || `obs_${i}`;
        else if (kind === "output") label = actionLabels[i] || `act_${i}`;
        else label = numLayers === 3 ? `h${i}` : `h${l}.${i}`;

        const bias = l > 0 ? decoded.layers[l - 1].b[i] : 0.0;
        const coord = decoded.layerCoords[l] ? decoded.layerCoords[l][i] : 0.0;
        const node = {
          id: `L${l}_${i}`,
          layer: l,
          index: i,
          kind,
          label,
          activation: kind === "input" ? -1 : 0,
          activationName: kind === "input" ? "sensor" : "tanh",
          bias,
          coord,
        };
        nodes.push(node);
        arr.push(node);
        nodeMap.set(node.id, node);
      }
      layersNodes.push(arr);
    }

    const edges = [];
    const matrices = [];
    let maxAbsWeight = 0.0;

    for (let l = 0; l < decoded.layers.length; l++) {
      const layer = decoded.layers[l];
      const rowLabels = layersNodes[l].map((n) => n.label);
      const colLabels = layersNodes[l + 1].map((n) => n.label);
      matrices.push({
        title: `W${l + 1} (${rowLabels.length}x${colLabels.length}) + b${l + 1}`,
        dIn: layer.dIn,
        dOut: layer.dOut,
        w: layer.w,
        b: layer.b,
        rowLabels,
        colLabels,
      });
      for (let i = 0; i < layer.dIn; i++) {
        for (let j = 0; j < layer.dOut; j++) {
          const weight = layer.w[i * layer.dOut + j];
          const absW = Math.abs(weight);
          if (absW > maxAbsWeight) maxAbsWeight = absW;
          edges.push({
            id: `e_${l}_${i}_${j}`,
            from: `L${l}_${i}`,
            to: `L${l + 1}_${j}`,
            fromLayer: l,
            toLayer: l + 1,
            weight,
            enabled: true,
            recurrent: false,
          });
        }
      }
    }

    const layerTitles = layersNodes.map((arr, idx) =>
      idx === 0
        ? `Sensors (${arr.length})`
        : idx === layersNodes.length - 1
          ? `Actuators (${arr.length})`
          : `Hidden ${layersNodes.length > 3 ? idx : ""} (${arr.length})`.trim()
    );
    const geom = assignCoordinates(layersNodes, layerTitles);

    return {
      kind: "substrate",
      title: `${taskSpec?.label || "Control"} Substrate`,
      dims: decoded.dims,
      layersNodes,
      layerTitles,
      nodes,
      nodeMap,
      edges,
      matrices,
      maxAbsWeight: maxAbsWeight || 1.0,
      paramCount: edges.length + nodes.filter((n) => n.layer > 0).length,
      width: geom.width,
      height: geom.height,
    };
  }

  function buildCppnGraph(genome, taskSpec) {
    const w = genome?.weights || [];
    const b = genome?.biases || [];
    const acts = genome?.activations || [];
    const isControl = isControlTaskSpec(taskSpec);
    const dims = [4, 12, 12, 12, 3];
    const inLabels = isControl ? ["x_src", "y_dst", "z_layer", "const"] : ["x", "y", "r (radial)", "1 (bias)"];
    const outLabels = isControl ? ["W / b (ch 0)", "G (unused)", "B (unused)"] : ["R", "G", "B"];

    const nodes = [];
    const layersNodes = [];
    const nodeMap = new Map();
    const actCounts = [0, 0, 0, 0, 0];

    let biasIdx = 0;
    let actIdx = 0;
    for (let l = 0; l < dims.length; l++) {
      const count = dims[l];
      const kind = l === 0 ? "input" : l === dims.length - 1 ? "output" : "hidden";
      const arr = [];
      for (let i = 0; i < count; i++) {
        let label = "";
        let actCode = 0;
        let bias = 0.0;
        if (kind === "input") {
          label = inLabels[i];
          actCode = -1;
        } else if (kind === "output") {
          label = outLabels[i];
          actCode = 0;
          bias = b[biasIdx++] || 0.0;
        } else {
          label = `c${l}.${i}`;
          actCode = typeof acts[actIdx] === "number" ? acts[actIdx] : 0;
          if (actCode >= 0 && actCode < actCounts.length) actCounts[actCode]++;
          actIdx++;
          bias = b[biasIdx++] || 0.0;
        }
        const meta = kind === "input" ? { name: "input", short: "in", color: "#38bdf8" } : getActMeta(actCode);
        const node = {
          id: `C${l}_${i}`,
          layer: l,
          index: i,
          kind,
          label,
          activation: actCode,
          activationName: kind === "output" ? "0.5+0.5tanh" : meta.short,
          activationColor: meta.color,
          bias,
        };
        nodes.push(node);
        arr.push(node);
        nodeMap.set(node.id, node);
      }
      layersNodes.push(arr);
    }

    const edges = [];
    const matrices = [];
    let wOffset = 0;
    let bOffset = 0;
    let maxAbsWeight = 0.0;
    for (let l = 0; l < dims.length - 1; l++) {
      const dIn = dims[l];
      const dOut = dims[l + 1];
      const wMat = new Float64Array(dIn * dOut);
      const bVec = new Float64Array(dOut);
      for (let j = 0; j < dOut; j++) {
        bVec[j] = b[bOffset + j] || 0.0;
        for (let i = 0; i < dIn; i++) {
          const weight = w[wOffset + j * dIn + i] || 0.0;
          wMat[i * dOut + j] = weight;
          const absW = Math.abs(weight);
          if (absW > maxAbsWeight) maxAbsWeight = absW;
          edges.push({
            id: `ce_${l}_${i}_${j}`,
            from: `C${l}_${i}`,
            to: `C${l + 1}_${j}`,
            fromLayer: l,
            toLayer: l + 1,
            weight,
            enabled: true,
            recurrent: false,
          });
        }
      }
      matrices.push({
        title: `CPPN W${l + 1} (${dIn}x${dOut}) + b${l + 1}`,
        dIn,
        dOut,
        w: wMat,
        b: bVec,
        rowLabels: layersNodes[l].map((n) => n.label),
        colLabels: layersNodes[l + 1].map((n) => n.label),
      });
      wOffset += dIn * dOut;
      bOffset += dOut;
    }

    const layerTitles = ["Coords (4)", "CPPN H1 (12)", "CPPN H2 (12)", "CPPN H3 (12)", "Out (3)"];
    const geom = assignCoordinates(layersNodes, layerTitles);

    return {
      kind: "cppn",
      title: isControl ? "HyperNEAT CPPN (4->12->12->12->3)" : "Picbreeder CPPN (4->12->12->12->3)",
      dims,
      layersNodes,
      layerTitles,
      nodes,
      nodeMap,
      edges,
      matrices,
      actCounts,
      isControl,
      genome,
      maxAbsWeight: maxAbsWeight || 1.0,
      paramCount: edges.length + 39,
      width: geom.width,
      height: geom.height,
    };
  }

  function buildNeatDagGraph(neatGenome, taskSpec) {
    const rawNodes = Array.isArray(neatGenome?.nodes) ? neatGenome.nodes : [];
    const rawEdges = Array.isArray(neatGenome?.edges)
      ? neatGenome.edges
      : Array.isArray(neatGenome?.connections)
        ? neatGenome.connections
        : [];
    const obsLabels = readSpecField(taskSpec, "obsLabels", "obs_labels") || [];
    const actionLabels = readSpecField(taskSpec, "actionLabels", "action_labels") || [];

    const nodes = [];
    const nodeMap = new Map();
    let inIdx = 0;
    let outIdx = 0;

    for (let i = 0; i < rawNodes.length; i++) {
      const rn = rawNodes[i];
      const id = String(rn.id !== undefined ? rn.id : i);
      const rawKind = String(rn.kind || rn.type || "hidden").toLowerCase();
      const kind =
        rawKind === "sensor" || rawKind === "input" || rawKind === "bias"
          ? "input"
          : rawKind === "motor" || rawKind === "output"
            ? "output"
            : "hidden";
      let label = rn.label;
      if (!label) {
        if (kind === "input") label = rawKind === "bias" ? "bias" : obsLabels[inIdx++] || `in_${id}`;
        else if (kind === "output") label = actionLabels[outIdx++] || `out_${id}`;
        else label = `n${id}`;
      }
      const meta = kind === "input" ? { name: rawKind, short: rawKind === "bias" ? "1" : "in", color: "#38bdf8" } : getActMeta(rn.activation || "tanh");
      const node = {
        id,
        layer: 0,
        index: i,
        kind,
        rawKind,
        label,
        activation: rn.activation || "tanh",
        activationName: meta.short,
        activationColor: meta.color,
        bias: Number(rn.bias || 0.0),
      };
      nodes.push(node);
      nodeMap.set(id, node);
    }

    const adj = new Map();
    const indeg = new Map();
    for (const n of nodes) {
      adj.set(n.id, []);
      indeg.set(n.id, 0);
    }

    const parsedEdges = [];
    for (let i = 0; i < rawEdges.length; i++) {
      const re = rawEdges[i];
      const from = String(re.from !== undefined ? re.from : re.in);
      const to = String(re.to !== undefined ? re.to : re.out);
      if (!nodeMap.has(from) || !nodeMap.has(to)) continue;
      const enabled = re.enabled !== false;
      const weight = Number(re.weight || 0.0);
      parsedEdges.push({
        id: `ne_${i}_${from}_${to}`,
        from,
        to,
        weight,
        enabled,
        innovation: re.innovation,
        recurrent: false,
      });
      if (enabled && from !== to) {
        adj.get(from).push(to);
        indeg.set(to, (indeg.get(to) || 0) + 1);
      }
    }

    // Kahn topological sort with cycle breaking for recurrent NEAT nets
    const queue = [];
    for (const n of nodes) {
      if ((indeg.get(n.id) || 0) === 0) queue.push(n.id);
    }
    const topoOrder = [];
    const seen = new Set();
    while (queue.length > 0 || seen.size < nodes.length) {
      if (queue.length === 0) {
        for (const n of nodes) {
          if (!seen.has(n.id)) {
            queue.push(n.id);
            break;
          }
        }
      }
      const u = queue.shift();
      if (seen.has(u)) continue;
      seen.add(u);
      topoOrder.push(u);
      for (const v of adj.get(u) || []) {
        indeg.set(v, (indeg.get(v) || 0) - 1);
        if ((indeg.get(v) || 0) <= 0 && !seen.has(v)) queue.push(v);
      }
    }

    const posInTopo = new Map();
    topoOrder.forEach((id, idx) => posInTopo.set(id, idx));

    const layerMap = new Map();
    for (const n of nodes) layerMap.set(n.id, 0);

    // Longest-path DAG relaxation over forward edges
    for (const u of topoOrder) {
      const uNode = nodeMap.get(u);
      const lu = uNode.kind === "input" ? 0 : layerMap.get(u);
      layerMap.set(u, lu);
      for (const e of parsedEdges) {
        if (!e.enabled || e.from !== u) continue;
        if (posInTopo.get(e.from) >= posInTopo.get(e.to)) {
          e.recurrent = true;
          continue;
        }
        const candidate = lu + 1;
        if (candidate > (layerMap.get(e.to) || 0)) {
          layerMap.set(e.to, candidate);
        }
      }
    }

    let maxHiddenLayer = 0;
    for (const n of nodes) {
      if (n.kind === "input") layerMap.set(n.id, 0);
      else if (n.kind !== "output") {
        const lv = Math.max(1, layerMap.get(n.id) || 1);
        layerMap.set(n.id, lv);
        if (lv > maxHiddenLayer) maxHiddenLayer = lv;
      }
    }
    let outputLayer = Math.max(1, maxHiddenLayer + 1);
    for (const n of nodes) {
      if (n.kind === "output") {
        outputLayer = Math.max(outputLayer, layerMap.get(n.id) || 1);
      }
    }
    for (const n of nodes) {
      n.layer = n.kind === "output" ? outputLayer : layerMap.get(n.id) || 0;
    }

    const layersNodes = Array.from({ length: outputLayer + 1 }, () => []);
    for (const n of nodes) {
      layersNodes[n.layer].push(n);
    }

    // 2-pass barycenter sweep on hidden layers to reduce crossings
    for (let pass = 0; pass < 2; pass++) {
      for (let l = 1; l < outputLayer; l++) {
        layersNodes[l].sort((a, b) => {
          const score = (node) => {
            const rels = parsedEdges.filter((e) => e.enabled && !e.recurrent && e.to === node.id);
            if (!rels.length) return node.index;
            let s = 0;
            for (const e of rels) {
              const src = nodeMap.get(e.from);
              s += src ? layersNodes[src.layer].indexOf(src) : 0;
            }
            return s / rels.length;
          };
          return score(a) - score(b);
        });
      }
      for (let l = outputLayer - 1; l >= 1; l--) {
        layersNodes[l].sort((a, b) => {
          const score = (node) => {
            const rels = parsedEdges.filter((e) => e.enabled && !e.recurrent && e.from === node.id);
            if (!rels.length) return node.index;
            let s = 0;
            for (const e of rels) {
              const dst = nodeMap.get(e.to);
              s += dst ? layersNodes[dst.layer].indexOf(dst) : 0;
            }
            return s / rels.length;
          };
          return score(a) - score(b);
        });
      }
    }

    // Drop empty internal layers if any (keep 0 and outputLayer compacted)
    const compactLayers = [];
    const oldToNew = new Map();
    for (let l = 0; l < layersNodes.length; l++) {
      if (layersNodes[l].length > 0) {
        oldToNew.set(l, compactLayers.length);
        compactLayers.push(layersNodes[l]);
      }
    }
    for (let l = 0; l < compactLayers.length; l++) {
      compactLayers[l].forEach((n, idx) => {
        n.layer = l;
        n.index = idx;
      });
    }

    let maxAbsWeight = 0.0;
    for (const e of parsedEdges) {
      const src = nodeMap.get(e.from);
      const dst = nodeMap.get(e.to);
      e.fromLayer = src ? src.layer : 0;
      e.toLayer = dst ? dst.layer : 0;
      if (e.fromLayer >= e.toLayer && e.enabled) e.recurrent = true;
      if (e.enabled && Math.abs(e.weight) > maxAbsWeight) maxAbsWeight = Math.abs(e.weight);
    }

    // Build N x N adjacency matrix sorted by topological layer
    const orderedNodes = compactLayers.flat();
    const nTotal = orderedNodes.length;
    const adjW = new Float64Array(nTotal * nTotal);
    const bVec = new Float64Array(nTotal);
    const ordIdx = new Map();
    orderedNodes.forEach((n, idx) => {
      ordIdx.set(n.id, idx);
      bVec[idx] = n.bias || 0.0;
    });
    for (const e of parsedEdges) {
      if (!e.enabled) continue;
      const r = ordIdx.get(e.from);
      const c = ordIdx.get(e.to);
      if (r !== undefined && c !== undefined) adjW[r * nTotal + c] = e.weight;
    }

    const labels = orderedNodes.map((n) => n.label);
    const matrices = nTotal > 0 ? [{
      title: `NEAT Adjacency (${nTotal}x${nTotal}) + bias`,
      dIn: nTotal,
      dOut: nTotal,
      w: adjW,
      b: bVec,
      rowLabels: labels,
      colLabels: labels,
    }] : [];

    const layerTitles = compactLayers.map((arr, idx) =>
      idx === 0
        ? `Sensors (${arr.length})`
        : idx === compactLayers.length - 1
          ? `Outputs (${arr.length})`
          : `Hidden L${idx} (${arr.length})`
    );
    const geom = assignCoordinates(compactLayers, layerTitles);

    return {
      kind: "neat",
      title: `${taskSpec?.label || "NEAT"} Topology`,
      dims: compactLayers.map((a) => a.length),
      layersNodes: compactLayers,
      layerTitles,
      nodes,
      nodeMap,
      edges: parsedEdges,
      matrices,
      maxAbsWeight: maxAbsWeight || 1.0,
      paramCount: parsedEdges.filter((e) => e.enabled).length + nodes.filter((n) => n.kind !== "input").length,
      width: geom.width,
      height: geom.height,
    };
  }

  function decodeF16Sample(b64) {
    if (!b64 || typeof atob !== "function") return new Float32Array(0);
    const bin = atob(b64);
    const n = bin.length >> 1;
    const out = new Float32Array(n);
    const f32 = new Float32Array(1);
    const u32 = new Uint32Array(f32.buffer);
    for (let i = 0; i < n; i++) {
      const h = bin.charCodeAt(i * 2) | (bin.charCodeAt(i * 2 + 1) << 8);
      const s = (h & 0x8000) << 16;
      const e = (h >> 10) & 0x1f;
      const m = h & 0x03ff;
      if (e === 0) u32[0] = s;
      else if (e === 31) u32[0] = s | 0x7f800000 | (m << 13);
      else u32[0] = s | ((e - 15 + 127) << 23) | (m << 13);
      out[i] = f32[0];
    }
    return out;
  }

  function buildPopuloraGraph(genome, taskSpec) {
    const r = Number(genome.lowRank || taskSpec?.policySpec?.lowRank || 8);
    const adapters = genome.adapters || {};
    const actLabels = Array.isArray(taskSpec?.actionLabels) ? taskSpec.actionLabels : [];
    const obsLabels = Array.isArray(taskSpec?.obsLabels) ? taskSpec.obsLabels : [];

    // Sampled slice matching createPopuLoraRunner stepActivations (16 obs -> 16 enc -> 16 gru -> 16 dec -> actionDim)
    const nSample = 16;
    const nAct = Math.max(1, Number(taskSpec?.actionDim || actLabels.length || 14));
    const stageSpecs = [
      { title: `Obs [0..${nSample - 1}]`, count: nSample, kind: "input", act: "linear", labels: obsLabels },
      { title: `Enc LoRA (r=${r})`, count: nSample, kind: "hidden", act: "tanh", key: "net_encoder_2" },
      { title: `GRU Cell (r=${r})`, count: nSample, kind: "hidden", act: "tanh", key: "net_cell_linear_hh" },
      { title: `Dec LoRA (r=${r})`, count: nSample, kind: "hidden", act: "tanh", key: "net_decoder_0" },
      { title: `Actions (${nAct})`, count: nAct, kind: "output", act: "tanh", key: "net_decoder_2", labels: actLabels },
    ];

    const layersNodes = [];
    const nodes = [];
    const nodeMap = new Map();
    stageSpecs.forEach((st, lIdx) => {
      const col = [];
      for (let i = 0; i < st.count; i++) {
        const id = `pl_${lIdx}_${i}`;
        const node = {
          id,
          layer: lIdx,
          index: i,
          kind: st.kind,
          label: (st.labels && st.labels[i]) ? String(st.labels[i]) : `${lIdx === 0 ? "obs" : lIdx === 4 ? "act" : "h"}${i}`,
          bias: 0.0,
          actMeta: getActMeta(st.act),
        };
        col.push(node);
        nodes.push(node);
        nodeMap.set(id, node);
      }
      layersNodes.push(col);
    });

    const edges = [];
    const matrices = [];
    let maxAbsWeight = 0.01;
    let totalParams = 0;

    for (const [k, ad] of Object.entries(adapters)) {
      totalParams += (Number(ad.inDim || 0) + Number(ad.outDim || 0)) * r;
    }

    const stageKeys = ["net_encoder_0", "net_cell_linear_ih", "net_decoder_0", "net_decoder_2"];
    for (let s = 0; s < stageKeys.length; s++) {
      const srcCol = layersNodes[s];
      const dstCol = layersNodes[s + 1];
      const key = stageKeys[s];
      const ad = adapters[key];
      if (!ad || !ad.down || !ad.up) continue;
      const down = decodeF16Sample(ad.down); // [inDim, r]
      const up = decodeF16Sample(ad.up);     // [outDim, r]
      const rows = srcCol.length;
      const cols = dstCol.length;
      const subW = new Float64Array(rows * cols);

      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          let w = 0.0;
          for (let rk = 0; rk < r; rk++) {
            w += (up[j * r + rk] || 0) * (down[i * r + rk] || 0);
          }
          subW[i * cols + j] = w;
          const aw = Math.abs(w);
          if (aw > maxAbsWeight) maxAbsWeight = aw;
          edges.push({
            id: `pe_${s}_${i}_${j}`,
            from: srcCol[i].id,
            to: dstCol[j].id,
            layer: s,
            weight: w,
            enabled: true,
          });
        }
      }
      matrices.push({
        title: `LoRA \u0394W ${key} (slice ${rows}\u00d7${cols}, \u2016\u0394W\u2016=${(ad.norm ?? 0).toFixed(2)})`,
        dIn: rows,
        dOut: cols,
        w: subW,
        b: new Float64Array(cols),
        rowLabels: srcCol.map((n) => n.label),
        colLabels: dstCol.map((n) => n.label),
      });
    }

    const layerTitles = stageSpecs.map((s) => s.title);
    const geom = assignCoordinates(layersNodes, layerTitles);
    return {
      kind: "populora",
      title: `PopuLoRA Adapter (rank ${r})`,
      dims: stageSpecs.map((s) => s.count),
      layersNodes,
      layerTitles,
      nodes,
      nodeMap,
      edges,
      matrices,
      maxAbsWeight,
      paramCount: totalParams,
      width: geom.width,
      height: geom.height,
    };
  }

  function buildGraphModel(genome, taskSpec, viewMode) {
    if (genome && genome.kind === "populora") {
      return buildPopuloraGraph(genome, taskSpec);
    }
    if (genome && Array.isArray(genome.nodes)) {
      return buildNeatDagGraph(genome, taskSpec);
    }
    const hasSubstrate = Boolean(readSpecDims(taskSpec));
    if (viewMode === "cppn" || !hasSubstrate) {
      return buildCppnGraph(genome, taskSpec);
    }
    return buildSubstrateGraph(genome, taskSpec);
  }

  function formatWeight(val) {
    if (!Number.isFinite(val)) return "0.000";
    return (val >= 0 ? "+" : "") + val.toFixed(3);
  }

  function weightToColor(val, maxAbs) {
    const t = Math.max(-1, Math.min(1, val / Math.max(1e-6, maxAbs)));
    if (t >= 0) {
      // Dark muted -> emerald #34d399 (52, 211, 153)
      const a = 0.14 + 0.86 * Math.pow(t, 0.72);
      return `rgba(52, 211, 153, ${a.toFixed(3)})`;
    }
    // Coral #fb7185 (251, 113, 133)
    const a = 0.14 + 0.86 * Math.pow(-t, 0.72);
    return `rgba(251, 113, 133, ${a.toFixed(3)})`;
  }

  function elSvg(tag, attrs) {
    const node = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      for (const k of Object.keys(attrs)) {
        node.setAttribute(k, String(attrs[k]));
      }
    }
    return node;
  }

  function elHtml(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  class NetworkInspector {
    constructor(container, opts) {
      this.container = container;
      this.opts = opts || {};
      this.genome = null;
      this.taskSpec = null;
      this.viewMode = "substrate"; // 'substrate' | 'cppn' | 'matrices'
      this.threshold = 0.05;
      this.hideIsolated = false;
      this.model = null;
      this.hoveredNodeId = null;
      this.hoveredEdgeId = null;
      this.liveActivations = null;
      this._rafPending = false;
      this._nodeDom = new Map();
      this._edgeDom = new Map();
      this._outputBarDom = new Map();
      this._matrixHitBoxes = [];
      this._buildShell();
    }

    _buildShell() {
      this.container.innerHTML = "";
      this.root = elHtml("div", "det-nv-root");

      // Top toolbar
      this.toolbar = elHtml("div", "det-nv-toolbar");
      this.tabsEl = elHtml("div", "det-nv-tabs");
      this.tabBtns = {};
      [
        ["substrate", "Substrate"],
        ["cppn", "CPPN Genome"],
        ["matrices", "Matrices"],
      ].forEach(([mode, label]) => {
        const btn = elHtml("button", "det-nv-tab", label);
        btn.type = "button";
        btn.dataset.mode = mode;
        btn.addEventListener("click", () => this.setMode(mode));
        this.tabsEl.appendChild(btn);
        this.tabBtns[mode] = btn;
      });
      this.toolbar.appendChild(this.tabsEl);

      // Controls group: pruning threshold + hide isolated + export SVG
      this.controlsEl = elHtml("div", "det-nv-controls");

      this.threshWrap = elHtml("label", "det-nv-slider-wrap");
      this.threshTitle = elHtml("span", "det-nv-ctrl-label", "|w| >= ");
      this.threshSlider = document.createElement("input");
      this.threshSlider.type = "range";
      this.threshSlider.min = "0";
      this.threshSlider.max = "100";
      this.threshSlider.value = "8";
      this.threshSlider.className = "det-nv-slider";
      this.threshValueEl = elHtml("span", "det-nv-ctrl-val", "0.05");
      this.threshSlider.addEventListener("input", () => {
        const maxW = this.model ? this.model.maxAbsWeight : 1.0;
        const frac = Number(this.threshSlider.value) / 100.0;
        this.threshold = Math.round(frac * maxW * 0.9 * 1000) / 1000;
        this.threshValueEl.textContent = this.threshold.toFixed(2);
        this._applyFilterState();
      });
      this.threshWrap.appendChild(this.threshTitle);
      this.threshWrap.appendChild(this.threshSlider);
      this.threshWrap.appendChild(this.threshValueEl);
      this.controlsEl.appendChild(this.threshWrap);

      this.dimBtn = elHtml("button", "det-nv-chip-btn", "Dim Isolated");
      this.dimBtn.type = "button";
      this.dimBtn.addEventListener("click", () => {
        this.hideIsolated = !this.hideIsolated;
        this.dimBtn.classList.toggle("active", this.hideIsolated);
        this._applyFilterState();
      });
      this.controlsEl.appendChild(this.dimBtn);

      this.exportBtn = elHtml("button", "det-nv-chip-btn", "Export SVG");
      this.exportBtn.type = "button";
      this.exportBtn.addEventListener("click", () => this.exportSvg());
      this.controlsEl.appendChild(this.exportBtn);

      this.toolbar.appendChild(this.controlsEl);
      this.root.appendChild(this.toolbar);

      // Meta stats strip
      this.metaBar = elHtml("div", "det-nv-meta");
      this.root.appendChild(this.metaBar);

      // Stage container
      this.stage = elHtml("div", "det-nv-stage");
      this.root.appendChild(this.stage);

      // Fixed-height inspector detail bar (no layout jitter on hover)
      this.detailBar = elHtml("div", "det-nv-detail", "Hover any neuron or synapse to inspect weights, activations, and receptive paths.");
      this.root.appendChild(this.detailBar);

      this.container.appendChild(this.root);
    }

    setData(genome, taskSpec, preferredMode) {
      this.genome = genome;
      this.taskSpec = taskSpec || {};
      const isNeat = Boolean(genome && Array.isArray(genome.nodes));
      const isPopulora = Boolean(genome && genome.kind === "populora");
      const hasSubstrate = Boolean(readSpecDims(this.taskSpec)) || isPopulora;

      if (preferredMode) {
        this.viewMode = preferredMode;
      } else if (!hasSubstrate && !isNeat && this.viewMode === "substrate") {
        this.viewMode = "cppn";
      } else if (hasSubstrate && (isPopulora ? this.viewMode === "cppn" : (this.viewMode !== "cppn" && this.viewMode !== "matrices"))) {
        this.viewMode = "substrate";
      }

      // Toggle visible mode tabs appropriately
      this.tabBtns.substrate.hidden = !hasSubstrate && !isNeat;
      this.tabBtns.cppn.hidden = isNeat || isPopulora;
      this.tabBtns.substrate.textContent = isPopulora ? "LoRA Graph" : (isNeat ? "NEAT Topology" : "Substrate");

      this.hoveredNodeId = null;
      this.hoveredEdgeId = null;
      this.liveActivations = null;
      this.render();
    }

    setMode(mode) {
      if (this.viewMode === mode) return;
      this.viewMode = mode;
      this.hoveredNodeId = null;
      this.hoveredEdgeId = null;
      this.render();
    }

    updateActivations(layerActs) {
      this.liveActivations = layerActs && (Array.isArray(layerActs) || typeof layerActs === "object")
        ? layerActs
        : null;
      if (this.viewMode !== "substrate" || !this.model) return;
      if (this._rafPending) return;
      if (typeof requestAnimationFrame === "function") {
        this._rafPending = true;
        requestAnimationFrame(() => {
          this._rafPending = false;
          this._paintLiveActivations();
        });
      } else {
        this._paintLiveActivations();
      }
    }

    render() {
      if (!this.genome) {
        this.stage.innerHTML = '<div class="det-nv-empty">No genome selected.</div>';
        return;
      }
      for (const [m, btn] of Object.entries(this.tabBtns)) {
        btn.classList.toggle("active", m === this.viewMode);
      }

      const graphMode = this.viewMode === "matrices"
        ? (Boolean(readSpecDims(this.taskSpec)) || Array.isArray(this.genome.nodes) ? "substrate" : "cppn")
        : this.viewMode;
      this.model = buildGraphModel(this.genome, this.taskSpec, graphMode);

      const sliderFrac = Number(this.threshSlider.value) / 100.0;
      this.threshold = Math.round(sliderFrac * this.model.maxAbsWeight * 0.9 * 1000) / 1000;
      this.threshValueEl.textContent = this.threshold.toFixed(2);
      this.threshWrap.hidden = this.viewMode === "matrices";
      this.dimBtn.hidden = this.viewMode === "matrices";
      this.exportBtn.hidden = this.viewMode === "matrices";

      if (this.viewMode === "matrices") {
        this._renderMatrices();
      } else {
        this._renderSvgGraph();
      }
      this._updateMetaBar();
    }

    _updateMetaBar() {
      if (!this.model) return;
      const m = this.model;
      const totalEdges = m.edges.filter((e) => e.enabled !== false).length;
      const activeEdges = m.edges.filter((e) => e.enabled !== false && Math.abs(e.weight) >= this.threshold).length;
      const sparsityPct = totalEdges > 0 ? Math.round((1 - activeEdges / totalEdges) * 100) : 0;

      this.metaBar.innerHTML = "";
      const pills = [
        `${m.dims.join(" -> ")}`,
        `${activeEdges}/${totalEdges} links (${sparsityPct}% pruned)`,
        `${m.paramCount} params`,
        `max |w| ${m.maxAbsWeight.toFixed(2)}`,
      ];
      for (const text of pills) {
        this.metaBar.appendChild(elHtml("span", "det-nv-pill", text));
      }
      if (m.kind === "cppn" && Array.isArray(m.actCounts)) {
        m.actCounts.forEach((cnt, idx) => {
          if (!cnt) return;
          const meta = ACTIVATION_META[idx];
          const chip = elHtml("span", "det-nv-act-chip", `${cnt}x ${meta.short}`);
          chip.style.borderColor = meta.color;
          chip.style.color = meta.color;
          this.metaBar.appendChild(chip);
        });
      }
    }

    _renderSvgGraph() {
      const m = this.model;
      this.stage.innerHTML = "";
      this._nodeDom.clear();
      this._edgeDom.clear();
      this._outputBarDom.clear();

      const wrap = elHtml("div", "det-nv-svg-wrap");
      const svg = elSvg("svg", {
        viewBox: `0 0 ${m.width} ${m.height}`,
        class: "det-nv-svg",
        preserveAspectRatio: "xMidYMid meet",
      });
      this.svgEl = svg;

      // Clean dark canvas backdrop for standalone SVG export too
      svg.appendChild(elSvg("rect", {
        x: 0,
        y: 0,
        width: m.width,
        height: m.height,
        rx: 12,
        fill: "#0b0b0f",
      }));

      // Layer column guide lines + headers
      const headerGroup = elSvg("g", { class: "det-nv-layer-headers" });
      m.layersNodes.forEach((arr, idx) => {
        if (!arr.length) return;
        const x = arr[0].x;
        headerGroup.appendChild(elSvg("line", {
          x1: x,
          y1: 36,
          x2: x,
          y2: m.height - 20,
          stroke: "rgba(255,255,255,0.04)",
          "stroke-dasharray": "3 4",
        }));
        const title = elSvg("text", {
          x,
          y: 22,
          "text-anchor": "middle",
          fill: "#71717a",
          "font-size": "11",
          "font-weight": "600",
          "font-family": "ui-monospace, SFMono-Regular, Menlo, monospace",
        });
        title.textContent = m.layerTitles[idx] || `L${idx}`;
        headerGroup.appendChild(title);
      });
      svg.appendChild(headerGroup);

      // Edges group (drawn below nodes)
      const edgeGroup = elSvg("g", { class: "det-nv-edges" });
      for (const e of m.edges) {
        const src = m.nodeMap.get(e.from);
        const dst = m.nodeMap.get(e.to);
        if (!src || !dst) continue;
        const path = elSvg("path", {
          d: this._edgePath(src, dst, e),
          fill: "none",
          "stroke-linecap": "round",
          class: "det-nv-edge",
        });
        path.addEventListener("mouseenter", () => this._onHoverEdge(e));
        path.addEventListener("mouseleave", () => this._clearHover());
        edgeGroup.appendChild(path);
        this._edgeDom.set(e.id, { path, edge: e });
      }
      svg.appendChild(edgeGroup);

      // Nodes group
      const nodeGroup = elSvg("g", { class: "det-nv-nodes" });
      const dense = m.layersNodes.some((arr) => arr.length >= 16);
      const radiusBase = dense ? 8 : 11;

      for (const n of m.nodes) {
        const g = elSvg("g", {
          transform: `translate(${n.x.toFixed(1)}, ${n.y.toFixed(1)})`,
          class: "det-nv-node",
          style: "cursor:pointer",
        });

        // Bias outer ring when |bias| > 0.01
        const biasMag = Math.abs(n.bias || 0);
        if (n.kind !== "input" && biasMag > 0.005) {
          const biasRing = elSvg("circle", {
            r: radiusBase + 3.5,
            fill: "none",
            stroke: n.bias >= 0 ? "rgba(52,211,153,0.55)" : "rgba(251,113,133,0.55)",
            "stroke-width": Math.min(2.8, 0.8 + biasMag * 1.4).toFixed(2),
            "stroke-dasharray": "3 2",
          });
          g.appendChild(biasRing);
        }

        const fillCol =
          n.kind === "input"
            ? "#13293d"
            : n.kind === "output"
              ? "#1f2419"
              : "#18181f";
        const strokeCol =
          n.kind === "input"
            ? "#38bdf8"
            : n.kind === "output"
              ? "#f5c451"
              : n.activationColor || "#a1a1aa";

        const circle = elSvg("circle", {
          r: radiusBase,
          fill: fillCol,
          stroke: strokeCol,
          "stroke-width": n.kind === "hidden" ? "1.6" : "1.8",
        });
        g.appendChild(circle);

        // Activation badge inside/next to hidden node in CPPN / NEAT modes
        if ((m.kind === "cppn" || m.kind === "neat") && n.kind === "hidden") {
          const badge = elSvg("text", {
            x: 0,
            y: 3,
            "text-anchor": "middle",
            fill: n.activationColor || "#e4e4e7",
            "font-size": radiusBase >= 10 ? "7.5" : "6.5",
            "font-weight": "700",
            "font-family": "ui-monospace, SFMono-Regular, Menlo, monospace",
            "pointer-events": "none",
          });
          badge.textContent = (n.activationName || "").slice(0, 4);
          g.appendChild(badge);
        }

        // Pin label on left (inputs) or right (outputs)
        if (n.kind === "input") {
          const lbl = elSvg("text", {
            x: -(radiusBase + 7),
            y: 3.5,
            "text-anchor": "end",
            fill: "#bae6fd",
            "font-size": dense ? "10" : "11",
            "font-family": "ui-monospace, SFMono-Regular, Menlo, monospace",
          });
          lbl.textContent = n.label;
          g.appendChild(lbl);
        } else if (n.kind === "output") {
          const lbl = elSvg("text", {
            x: radiusBase + 8,
            y: 3.5,
            "text-anchor": "start",
            fill: "#fde68a",
            "font-size": "11",
            "font-weight": "600",
            "font-family": "ui-monospace, SFMono-Regular, Menlo, monospace",
          });
          lbl.textContent = n.label;
          g.appendChild(lbl);

          // Output live torque / signal bar slot (only for control substrates / NEAT policies)
          if (m.kind === "substrate" || m.kind === "neat") {
            const barBg = elSvg("rect", {
              x: radiusBase + 62,
              y: -4,
              width: 56,
              height: 8,
              rx: 4,
              fill: "rgba(255,255,255,0.06)",
              stroke: "rgba(255,255,255,0.08)",
              "stroke-width": 0.6,
            });
            const barCenter = elSvg("line", {
              x1: radiusBase + 90,
              y1: -5,
              x2: radiusBase + 90,
              y2: 5,
              stroke: "rgba(255,255,255,0.22)",
              "stroke-width": 0.8,
            });
            const barFill = elSvg("rect", {
              x: radiusBase + 90,
              y: -3.5,
              width: 0,
              height: 7,
              rx: 3,
              fill: "#34d399",
            });
            g.appendChild(barBg);
            g.appendChild(barCenter);
            g.appendChild(barFill);
            this._outputBarDom.set(n.id, {
              barFill,
              cx: radiusBase + 90,
              halfW: 27,
            });
          }
        }

        // Live activation value pill (positioned to prevent neighbor overlap in dense layers)
        const liveValAttrs =
          n.kind === "input"
            ? { x: -(radiusBase + 72), y: 3.5, "text-anchor": "end", fill: "#94a3b8" }
            : n.kind === "output"
              ? { x: radiusBase + 90, y: -8, "text-anchor": "middle", fill: "#e4e4e7" }
              : { x: 0, y: -(radiusBase + 5), "text-anchor": "middle", fill: "#e4e4e7" };
        const liveValText = elSvg("text", {
          ...liveValAttrs,
          "font-size": "8.5",
          "font-family": "ui-monospace, SFMono-Regular, Menlo, monospace",
          opacity: "0",
          "pointer-events": "none",
        });
        g.appendChild(liveValText);

        g.addEventListener("mouseenter", () => this._onHoverNode(n));
        g.addEventListener("mouseleave", () => this._clearHover());
        nodeGroup.appendChild(g);
        this._nodeDom.set(n.id, { group: g, circle, liveValText, node: n, strokeCol, fillCol });
      }
      svg.appendChild(nodeGroup);
      wrap.appendChild(svg);

      // In CPPN mode, overlay a compact 72x72 2D CPPN spatial slice preview in top-right corner
      if (m.kind === "cppn" && m.genome) {
        const previewCard = elHtml("div", "det-nv-cppn-preview");
        const pCanvas = document.createElement("canvas");
        pCanvas.width = 68;
        pCanvas.height = 68;
        this._drawCppnSlice(m.genome, m.isControl, pCanvas);
        const pCap = elHtml("span", "det-nv-cppn-cap", m.isControl ? "CPPN ch0 Field" : "CPPN RGB (68px)");
        previewCard.appendChild(pCanvas);
        previewCard.appendChild(pCap);
        wrap.appendChild(previewCard);
      }

      this.stage.appendChild(wrap);
      this._applyFilterState();
      if (this.liveActivations) this._paintLiveActivations();
    }

    _edgePath(src, dst, edge) {
      if (src.id === dst.id) {
        const r = 18;
        return `M ${src.x} ${src.y - 8} C ${src.x - r} ${src.y - 28}, ${src.x + r} ${src.y - 28}, ${src.x} ${src.y - 8}`;
      }
      if (edge.recurrent || src.layer >= dst.layer) {
        const lift = 28 + Math.abs(src.x - dst.x) * 0.14;
        const my = Math.min(src.y, dst.y) - lift;
        return `M ${src.x} ${src.y} Q ${(src.x + dst.x) * 0.5} ${my} ${dst.x} ${dst.y}`;
      }
      const skipLayers = dst.layer - src.layer;
      if (skipLayers > 1) {
        const bend = ((src.index + dst.index) % 2 === 0 ? -1 : 1) * (18 + 10 * skipLayers);
        const cx = (src.x + dst.x) * 0.5;
        const cy = (src.y + dst.y) * 0.5 + bend;
        return `M ${src.x} ${src.y} Q ${cx} ${cy} ${dst.x} ${dst.y}`;
      }
      const dx = (dst.x - src.x) * 0.48;
      return `M ${src.x} ${src.y} C ${src.x + dx} ${src.y}, ${dst.x - dx} ${dst.y}, ${dst.x} ${dst.y}`;
    }

    _applyFilterState() {
      if (!this.model || this.viewMode === "matrices") return;
      const m = this.model;
      const tau = this.threshold;
      const activeByNode = new Map();
      for (const n of m.nodes) activeByNode.set(n.id, 0);

      const focusNode = this.hoveredNodeId;
      const focusEdge = this.hoveredEdgeId;
      const connectedNodes = new Set();
      if (focusNode) connectedNodes.add(focusNode);

      // First pass: check active edges & collect hover neighbors
      for (const { edge } of this._edgeDom.values()) {
        const isEnabled = edge.enabled !== false;
        const passes = isEnabled && Math.abs(edge.weight) >= tau;
        if (passes) {
          activeByNode.set(edge.from, (activeByNode.get(edge.from) || 0) + 1);
          activeByNode.set(edge.to, (activeByNode.get(edge.to) || 0) + 1);
        }
        if (focusNode && passes && (edge.from === focusNode || edge.to === focusNode)) {
          connectedNodes.add(edge.from);
          connectedNodes.add(edge.to);
        }
        if (focusEdge && edge.id === focusEdge) {
          connectedNodes.add(edge.from);
          connectedNodes.add(edge.to);
        }
      }

      const hasFocus = Boolean(focusNode || focusEdge);

      for (const { path, edge } of this._edgeDom.values()) {
        const isEnabled = edge.enabled !== false;
        const absW = Math.abs(edge.weight);
        const passes = isEnabled && absW >= tau;
        if (!passes) {
          path.style.display = "none";
          continue;
        }
        path.style.display = "";
        const norm = Math.min(1.0, absW / Math.max(1e-6, m.maxAbsWeight));
        const isPos = edge.weight >= 0;
        const stroke = edge.recurrent ? "#f5c451" : isPos ? "#34d399" : "#fb7185";
        let opacity = 0.14 + 0.72 * Math.pow(norm, 0.78);
        let width = 0.65 + 2.5 * Math.pow(norm, 0.85);

        if (hasFocus) {
          const hit = focusEdge
            ? edge.id === focusEdge
            : edge.from === focusNode || edge.to === focusNode;
          opacity = hit ? 0.96 : 0.05;
          if (hit) width = Math.max(1.8, width * 1.35);
        }
        path.setAttribute("stroke", stroke);
        path.setAttribute("stroke-opacity", opacity.toFixed(3));
        path.setAttribute("stroke-width", width.toFixed(2));
        if (edge.recurrent) {
          path.setAttribute("stroke-dasharray", "4 3");
        } else {
          path.removeAttribute("stroke-dasharray");
        }
      }

      for (const { group, circle, node, strokeCol } of this._nodeDom.values()) {
        const deg = activeByNode.get(node.id) || 0;
        let op = 1.0;
        if (hasFocus) {
          op = connectedNodes.has(node.id) ? 1.0 : 0.22;
        } else if (this.hideIsolated && deg === 0) {
          op = 0.18;
        }
        group.setAttribute("opacity", op.toFixed(2));
        circle.setAttribute("stroke", connectedNodes.has(node.id) && hasFocus ? "#ffffff" : strokeCol);
      }

      this._updateMetaBar();
    }

    _onHoverNode(node) {
      this.hoveredNodeId = node.id;
      this.hoveredEdgeId = null;
      this._applyFilterState();

      const incoming = [];
      const outgoing = [];
      for (const e of this.model.edges) {
        if (e.enabled === false || Math.abs(e.weight) < this.threshold) continue;
        if (e.to === node.id) incoming.push(e);
        if (e.from === node.id) outgoing.push(e);
      }
      incoming.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
      outgoing.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));

      const parts = [
        `${node.kind.toUpperCase()} "${node.label}"`,
        node.kind === "input" ? "" : `act=${node.activationName}`,
        node.kind === "input" ? "" : `bias=${formatWeight(node.bias || 0)}`,
        `in=${incoming.length}`,
        `out=${outgoing.length}`,
      ].filter(Boolean);

      if (incoming.length) {
        const topIn = incoming.slice(0, 3).map((e) => {
          const src = this.model.nodeMap.get(e.from);
          return `${src ? src.label : e.from}:${formatWeight(e.weight)}`;
        });
        parts.push(`top-in=[${topIn.join(", ")}]`);
      }
      if (outgoing.length) {
        const topOut = outgoing.slice(0, 3).map((e) => {
          const dst = this.model.nodeMap.get(e.to);
          return `${dst ? dst.label : e.to}:${formatWeight(e.weight)}`;
        });
        parts.push(`top-out=[${topOut.join(", ")}]`);
      }
      this.detailBar.textContent = parts.join("  ·  ");
    }

    _onHoverEdge(edge) {
      this.hoveredEdgeId = edge.id;
      this.hoveredNodeId = null;
      this._applyFilterState();
      const src = this.model.nodeMap.get(edge.from);
      const dst = this.model.nodeMap.get(edge.to);
      const tag = edge.recurrent ? " [recurrent]" : "";
      this.detailBar.textContent = `${src ? src.label : edge.from}  --->  ${dst ? dst.label : edge.to}${tag}   |   w = ${formatWeight(edge.weight)}`;
    }

    _clearHover() {
      this.hoveredNodeId = null;
      this.hoveredEdgeId = null;
      this._applyFilterState();
      this.detailBar.textContent = "Hover any neuron or synapse to inspect weights, activations, and receptive paths.";
    }

    _paintLiveActivations() {
      const m = this.model;
      const acts = this.liveActivations;
      if (!m || this.viewMode !== "substrate") return;

      const applyNodeVal = (node, v, scaleDiv) => {
        const dom = this._nodeDom.get(node.id);
        if (!dom) return;
        if (v === null || v === undefined || !Number.isFinite(v)) {
          dom.circle.setAttribute("fill", dom.fillCol);
          dom.liveValText.setAttribute("opacity", "0");
          const bar = this._outputBarDom.get(node.id);
          if (bar) bar.barFill.setAttribute("width", "0");
          return;
        }
        const mag = Math.min(1.0, Math.abs(v) / scaleDiv);
        dom.circle.setAttribute(
          "fill",
          v >= 0
            ? `rgba(52, 211, 153, ${(0.14 + 0.72 * mag).toFixed(2)})`
            : `rgba(251, 113, 133, ${(0.14 + 0.72 * mag).toFixed(2)})`
        );
        if (node.kind === "output" || node.kind === "input") {
          dom.liveValText.textContent = v.toFixed(2);
          dom.liveValText.setAttribute("opacity", "0.85");
        }
        const bar = this._outputBarDom.get(node.id);
        if (bar) {
          const clamped = Math.max(-1.0, Math.min(1.0, v));
          const wPx = Math.abs(clamped) * bar.halfW;
          bar.barFill.setAttribute("width", wPx.toFixed(1));
          bar.barFill.setAttribute("x", (clamped >= 0 ? bar.cx : bar.cx - wPx).toFixed(1));
          bar.barFill.setAttribute("fill", clamped >= 0 ? "#34d399" : "#fb7185");
        }
      };

      if (!acts) {
        for (const n of m.nodes) applyNodeVal(n, null, 1.0);
        return;
      }

      if (!Array.isArray(acts)) {
        for (const n of m.nodes) {
          const raw = acts[n.id];
          applyNodeVal(n, raw !== undefined ? Number(raw) : null, n.layer === 0 ? 2.5 : 1.0);
        }
        return;
      }

      for (let l = 0; l < m.layersNodes.length; l++) {
        const layerVals = acts[l];
        const nodes = m.layersNodes[l];
        if (!Array.isArray(layerVals) || !nodes) continue;
        for (let i = 0; i < nodes.length && i < layerVals.length; i++) {
          applyNodeVal(nodes[i], Number(layerVals[i] || 0), l === 0 ? 2.5 : 1.0);
        }
      }
    }

    _drawCppnSlice(genome, isControl, canvas) {
      const ctx = canvas.getContext("2d");
      const sz = canvas.width;
      const img = ctx.createImageData(sz, sz);
      const data = img.data;
      const h1 = new Float32Array(12);
      const h2 = new Float32Array(12);
      const h3 = new Float32Array(12);
      const outRgb = new Float32Array(3);
      for (let py = 0; py < sz; py++) {
        const y = 1.0 - (2.0 * py) / (sz - 1);
        for (let px = 0; px < sz; px++) {
          const x = (2.0 * px) / (sz - 1) - 1.0;
          const x2 = isControl ? 0.0 : Math.min(Math.SQRT2, Math.hypot(x, y));
          evalCppnPoint(genome, x, y, x2, 1.0, h1, h2, h3, outRgb);
          const off = (py * sz + px) * 4;
          if (isControl) {
            const wVal = outRgb[0] * 2.0 - 1.0;
            if (wVal >= 0) {
              data[off] = 16;
              data[off + 1] = Math.round(40 + 185 * wVal);
              data[off + 2] = Math.round(32 + 125 * wVal);
            } else {
              const neg = -wVal;
              data[off] = Math.round(40 + 210 * neg);
              data[off + 1] = Math.round(20 + 60 * neg);
              data[off + 2] = Math.round(30 + 95 * neg);
            }
          } else {
            data[off] = Math.round(outRgb[0] * 255);
            data[off + 1] = Math.round(outRgb[1] * 255);
            data[off + 2] = Math.round(outRgb[2] * 255);
          }
          data[off + 3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
    }

    _renderMatrices() {
      const m = this.model;
      this.stage.innerHTML = "";
      this._matrixHitBoxes = [];
      const wrap = elHtml("div", "det-nv-matrices");

      for (const mat of m.matrices) {
        const card = elHtml("div", "det-nv-matrix-card");
        card.appendChild(elHtml("div", "det-nv-matrix-title", mat.title));

        const rows = mat.dIn;
        const cols = mat.dOut;
        const cellW = cols > 18 ? 14 : cols > 10 ? 18 : 22;
        const cellH = rows > 18 ? 13 : rows > 10 ? 17 : 21;
        const padLeft = 74;
        const padTop = 58;
        const biasGap = 8;
        const width = padLeft + cols * cellW + 18;
        const height = padTop + rows * cellH + biasGap + cellH + 14;

        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        canvas.className = "det-nv-matrix-canvas";
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#0b0b0f";
        ctx.fillRect(0, 0, width, height);

        let maxW = 1e-6;
        for (let k = 0; k < mat.w.length; k++) {
          const a = Math.abs(mat.w[k]);
          if (a > maxW) maxW = a;
        }

        // Column labels (rotated -45 deg for compact readability)
        ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = "#fde68a";
        for (let j = 0; j < cols; j++) {
          ctx.save();
          ctx.translate(padLeft + j * cellW + cellW * 0.55, padTop - 6);
          ctx.rotate(-Math.PI / 4);
          ctx.textAlign = "left";
          ctx.fillText(String(mat.colLabels[j] || j).slice(0, 9), 0, 0);
          ctx.restore();
        }

        // Row labels & weight cells
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        for (let i = 0; i < rows; i++) {
          const y = padTop + i * cellH;
          ctx.fillStyle = "#bae6fd";
          ctx.fillText(String(mat.rowLabels[i] || i).slice(0, 10), padLeft - 6, y + cellH * 0.5);
          for (let j = 0; j < cols; j++) {
            const x = padLeft + j * cellW;
            const val = mat.w[i * cols + j];
            ctx.fillStyle = "#141418";
            ctx.fillRect(x + 0.5, y + 0.5, cellW - 1, cellH - 1);
            ctx.fillStyle = weightToColor(val, maxW);
            ctx.fillRect(x + 0.5, y + 0.5, cellW - 1, cellH - 1);
          }
        }

        // Destination bias row aligned beneath each output column j = 0..cols-1
        let maxB = 1e-6;
        for (let j = 0; j < mat.b.length; j++) {
          const a = Math.abs(mat.b[j]);
          if (a > maxB) maxB = a;
        }
        const by = padTop + rows * cellH + biasGap;
        ctx.fillStyle = "#a1a1aa";
        ctx.fillText("bias b", padLeft - 6, by + cellH * 0.5);
        for (let j = 0; j < cols; j++) {
          const x = padLeft + j * cellW;
          ctx.fillStyle = "#141418";
          ctx.fillRect(x + 0.5, by + 0.5, cellW - 1, cellH - 1);
          ctx.fillStyle = weightToColor(mat.b[j] || 0, maxB);
          ctx.fillRect(x + 0.5, by + 0.5, cellW - 1, cellH - 1);
        }

        canvas.addEventListener("mousemove", (ev) => {
          const rect = canvas.getBoundingClientRect();
          const sx = ((ev.clientX - rect.left) * canvas.width) / Math.max(1, rect.width);
          const sy = ((ev.clientY - rect.top) * canvas.height) / Math.max(1, rect.height);
          const col = Math.floor((sx - padLeft) / cellW);
          if (col < 0 || col >= cols) return;
          if (sy >= by && sy <= by + cellH) {
            this.detailBar.textContent = `bias[${mat.colLabels[col]}]  =  ${formatWeight(mat.b[col] || 0)}`;
            return;
          }
          const row = Math.floor((sy - padTop) / cellH);
          if (row >= 0 && row < rows) {
            const val = mat.w[row * cols + col];
            const bVal = mat.b[col] || 0;
            this.detailBar.textContent = `${mat.rowLabels[row]}  ->  ${mat.colLabels[col]}   |   w = ${formatWeight(val)}   ·   dst bias = ${formatWeight(bVal)}`;
          }
        });
        canvas.addEventListener("mouseleave", () => {
          this.detailBar.textContent = "Hover any matrix cell to inspect connection weight and destination bias.";
        });

        card.appendChild(canvas);
        wrap.appendChild(card);
      }

      this.stage.appendChild(wrap);
    }

    exportSvg() {
      if (!this.svgEl || typeof XMLSerializer === "undefined" || typeof Blob === "undefined") return;
      const serializer = new XMLSerializer();
      const raw = serializer.serializeToString(this.svgEl);
      const blob = new Blob([raw], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const slug = String(this.taskSpec?.task || this.taskSpec?.id || "network").replace(/[^a-z0-9_-]/gi, "_");
      a.download = `network_${slug}_${this.viewMode}.svg`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    destroy() {
      this.container.innerHTML = "";
      this._nodeDom.clear();
      this._edgeDom.clear();
      this._outputBarDom.clear();
    }
  }

  const api = {
    ACTIVATION_META,
    getActMeta,
    applyActivation,
    decodeHyperNeatWeights,
    buildSubstrateGraph,
    buildCppnGraph,
    buildNeatDagGraph,
    buildGraphModel,
    NetworkInspector,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  global.DetNetworkViz = api;
})(typeof window !== "undefined" ? window : globalThis);
