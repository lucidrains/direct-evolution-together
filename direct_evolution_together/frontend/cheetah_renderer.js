// Crisp 2D sagittal HalfCheetah canvas renderer matching `half_cheetah.xml` exact kinematics.
(function (global) {
  "use strict";

  const HEAD_SIN = Math.sin(0.87);
  const HEAD_COS = Math.cos(0.87);
  const BACK_LEG_SPECS = [
    ["bthigh", -0.5, 0.0, 3, 0.1, -0.13, Math.sin(-3.8), Math.cos(-3.8), 0.145],
    ["bshin", 0.16, -0.25, 4, -0.14, -0.07, Math.sin(-2.03), Math.cos(-2.03), 0.15],
    ["bfoot", -0.28, -0.14, 5, 0.03, -0.097, Math.sin(-0.27), Math.cos(-0.27), 0.094],
  ];
  const FRONT_LEG_SPECS = [
    ["fthigh", 0.5, 0.0, 6, -0.07, -0.12, Math.sin(0.52), Math.cos(0.52), 0.133],
    ["fshin", -0.14, -0.24, 7, 0.065, -0.09, Math.sin(-0.6), Math.cos(-0.6), 0.106],
    ["ffoot", 0.13, -0.18, 8, 0.045, -0.07, Math.sin(-0.6), Math.cos(-0.6), 0.07],
  ];

  function rotXZ(x, z, a) {
    const c = Math.cos(a);
    const s = Math.sin(a);
    return [x * c + z * s, -x * s + z * c];
  }

  function appendLegSegments(segs, frame, px, pz, pa, specs) {
    let x = px;
    let z = pz;
    let a = pa;
    for (let i = 0; i < specs.length; i++) {
      const [name, ox, oz, qIdx, gx, gz, sga, cga, h] = specs[i];
      const ca = Math.cos(a);
      const sa = Math.sin(a);
      x += ox * ca + oz * sa;
      z += -ox * sa + oz * ca;
      a += frame[qIdx];
      const ca2 = Math.cos(a);
      const sa2 = Math.sin(a);
      const cx = gx * ca2 + gz * sa2;
      const cz = -gx * sa2 + gz * ca2;
      const ax0 = sga * ca2 + cga * sa2;
      const ax1 = -sga * sa2 + cga * ca2;
      segs.push([
        name,
        [x + cx - h * ax0, z + cz - h * ax1],
        [x + cx + h * ax0, z + cz + h * ax1],
      ]);
    }
  }

  function cheetahFkSegments(frame) {
    if (!frame || frame.length < 9) return [];
    const rx = frame[0];
    const rz = frame[1];
    const ry = frame[2];
    const rzw = rz + 0.7;

    const t1 = rotXZ(-0.5, 0.0, ry);
    const t2 = rotXZ(0.5, 0.0, ry);
    const hc = rotXZ(0.6, 0.1, ry);
    const ha = rotXZ(HEAD_SIN, HEAD_COS, ry);

    const segs = [
      ["torso", [rx + t1[0], rzw + t1[1]], [rx + t2[0], rzw + t2[1]]],
      ["head", [rx + hc[0] - 0.15 * ha[0], rzw + hc[1] - 0.15 * ha[1]], [rx + hc[0] + 0.15 * ha[0], rzw + hc[1] + 0.15 * ha[1]]],
    ];

    appendLegSegments(segs, frame, rx, rzw, ry, BACK_LEG_SPECS);
    appendLegSegments(segs, frame, rx, rzw, ry, FRONT_LEG_SPECS);
    return segs;
  }

  const SEG_COLORS = {
    bthigh: "#d99b3a",
    bshin: "#f5c451",
    bfoot: "#fbd379",
    torso: "#e4e4e7",
    head: "#fafafa",
    fthigh: "#38bdf8",
    fshin: "#67d4fc",
    ffoot: "#a5f3fc",
  };

  function drawCheetahPose(ctx, frame, camX, width, height, groundY, pxPerM, alpha, colorOverride) {
    const segs = cheetahFkSegments(frame);
    if (!segs.length) return;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.lineCap = "round";
    ctx.lineWidth = Math.max(2, Math.round(pxPerM * 0.09));
    for (const [name, p1, p2] of segs) {
      const x1 = width * 0.44 + (p1[0] - camX) * pxPerM;
      const y1 = groundY - p1[1] * pxPerM;
      const x2 = width * 0.44 + (p2[0] - camX) * pxPerM;
      const y2 = groundY - p2[1] * pxPerM;
      ctx.strokeStyle = colorOverride || SEG_COLORS[name] || "#e4e4e7";
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();

      if (!colorOverride && (name === "bfoot" || name === "ffoot")) {
        const minZ = Math.min(p1[1], p2[1]);
        if (minZ <= 0.056) {
          const tipX = p1[1] < p2[1] ? x1 : x2;
          ctx.fillStyle = name === "bfoot" ? "rgba(245, 196, 81, 0.75)" : "rgba(56, 189, 248, 0.75)";
          ctx.beginPath();
          ctx.arc(tipX, groundY, Math.max(2.5, pxPerM * 0.045), 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
    ctx.restore();
  }

  function renderCheetahStaticCanvas(summary, rollout, canvas) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.fillStyle = "#09090b";
    ctx.fillRect(0, 0, w, h);

    const frames = (rollout && rollout.frames) || [];
    if (!frames.length) {
      const restPose = [0, 0, 0, 0, 0, 0, 0, 0, 0];
      const groundY = w <= 40 ? h * 0.72 : Math.round(h * 0.60);
      const pxPerM = w <= 40 ? h * 0.34 : h * 0.32;
      ctx.strokeStyle = "#27272a";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(2, groundY);
      ctx.lineTo(w - 2, groundY);
      ctx.stroke();
      drawCheetahPose(ctx, restPose, 0, w, h, groundY, pxPerM, 0.28, "#71717a");
      return;
    }

    const meanDx = summary && typeof summary.mean_distance === "number" ? summary.mean_distance : 0;
    const solved = Boolean(summary && summary.solved);
    const accent = solved ? "#34d399" : (meanDx >= 2.5 ? "#f5c451" : "#fb7185");

    if (w <= 40) {
      const groundY = h * 0.72;
      ctx.strokeStyle = "#3f3f46";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(2, groundY);
      ctx.lineTo(w - 2, groundY);
      ctx.stroke();
      const mid = frames[Math.floor((frames.length - 1) * 0.65)] || frames[0];
      drawCheetahPose(ctx, mid, mid[0], w, h, groundY, h * 0.34, 0.95, accent);
      const pct = Math.max(0.05, Math.min(1.0, Math.max(0, meanDx) / 10.0));
      ctx.fillStyle = "#27272a";
      ctx.fillRect(3, h - 5, w - 6, 3);
      ctx.fillStyle = accent;
      ctx.fillRect(3, h - 5, Math.max(2, Math.round((w - 6) * pct)), 3);
      return;
    }

    const margin = Math.round(w * 0.08);
    const usableW = w - 2 * margin;
    const groundY = Math.round(h * 0.60);
    const pxPerM = h * 0.32;
    const finalFrame = frames[frames.length - 1];
    const camX = finalFrame[0];

    const mStart = Math.floor(camX - 2);
    const mEnd = Math.ceil(camX + 2);
    ctx.strokeStyle = "#18181f";
    ctx.lineWidth = 1;
    for (let m = mStart; m <= mEnd; m++) {
      const gx = w * 0.44 + (m - camX) * pxPerM;
      if (gx >= margin && gx <= w - margin) {
        ctx.beginPath();
        ctx.moveTo(gx, 12);
        ctx.lineTo(gx, groundY + 6);
        ctx.stroke();
      }
    }

    ctx.strokeStyle = "#3f3f46";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(margin, groundY);
    ctx.lineTo(w - margin, groundY);
    ctx.stroke();

    const snaps = [0.2, 0.5, 0.78];
    for (const frac of snaps) {
      const fi = Math.min(frames.length - 1, Math.floor((frames.length - 1) * frac));
      drawCheetahPose(ctx, frames[fi], camX, w, h, groundY, pxPerM, 0.2, "#71717a");
    }
    drawCheetahPose(ctx, finalFrame, camX, w, h, groundY, pxPerM, 0.96, null);

    const waveTop = Math.round(h * 0.71);
    const waveH = Math.round(h * 0.23);
    const waveMidY = waveTop + waveH * 0.5;
    ctx.fillStyle = "#111116";
    ctx.fillRect(margin, waveTop, usableW, waveH);
    ctx.strokeStyle = "#27272a";
    ctx.lineWidth = 1;
    ctx.strokeRect(margin, waveTop, usableW, waveH);
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.beginPath();
    ctx.moveTo(margin, waveMidY);
    ctx.lineTo(w - margin, waveMidY);
    ctx.stroke();

    const vxs = (rollout && rollout.vx_trace) || [];
    if (vxs.length >= 2) {
      ctx.strokeStyle = accent;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      const stride = Math.max(1, Math.floor(vxs.length / 60));
      for (let i = 0; i < vxs.length; i += stride) {
        const px = margin + (i / (vxs.length - 1)) * usableW;
        const normVx = Math.max(-1, Math.min(1, vxs[i] / 3.5));
        const py = waveMidY - normVx * (waveH * 0.42);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
  }

  function renderCheetahLiveFrame(canvas, frames, stepIdx, maxSteps) {
    if (!frames || !frames.length) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const margin = Math.round(w * 0.06);
    const usableW = w - 2 * margin;
    const groundY = Math.round(h * 0.73);
    const pxPerM = h * 0.36;

    const idx = Math.min(stepIdx, frames.length - 1);
    const cur = frames[idx];
    const camX = cur[0];
    const startX = frames[0][0];
    const dx = camX - startX;

    ctx.fillStyle = "#09090b";
    ctx.fillRect(0, 0, w, h);

    const mStart = Math.floor(camX - 3.5);
    const mEnd = Math.ceil(camX + 3.5);
    ctx.font = `10px ui-monospace, SFMono-Regular, Menlo, monospace`;
    for (let m = mStart; m <= mEnd; m++) {
      const gx = w * 0.44 + (m - camX) * pxPerM;
      if (gx < 0 || gx > w) continue;
      ctx.strokeStyle = "#18181f";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(gx, 24);
      ctx.lineTo(gx, groundY);
      ctx.stroke();

      ctx.strokeStyle = m % 2 === 0 ? "#71717a" : "#3f3f46";
      ctx.beginPath();
      ctx.moveTo(gx, groundY);
      ctx.lineTo(gx, groundY + (m % 2 === 0 ? 7 : 4));
      ctx.stroke();

      if (m % 2 === 0 && w >= 240) {
        ctx.fillStyle = "#71717a";
        ctx.fillText(`${m}m`, gx + 3, groundY + 15);
      }
    }

    ctx.strokeStyle = "#52525b";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, groundY);
    ctx.lineTo(w, groundY);
    ctx.stroke();

    const trailStart = Math.max(0, idx - 32);
    if (idx - trailStart >= 2) {
      ctx.strokeStyle = "rgba(56, 189, 248, 0.48)";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let i = trailStart; i <= idx; i++) {
        const f = frames[i];
        const tx = w * 0.44 + (f[0] - camX) * pxPerM;
        const ty = groundY - (f[1] + 0.7) * pxPerM;
        if (i === trailStart) ctx.moveTo(tx, ty);
        else ctx.lineTo(tx, ty);
      }
      ctx.stroke();
    }

    ctx.fillStyle = "#1f1f26";
    ctx.fillRect(margin, 12, usableW, 5);
    ctx.fillStyle = dx >= 8.0 ? "#34d399" : (dx >= 2.5 ? "#f5c451" : "#fb7185");
    ctx.fillRect(margin, 12, Math.round(usableW * Math.min(1, idx / Math.max(1, maxSteps))), 5);

    drawCheetahPose(ctx, cur, camX, w, h, groundY, pxPerM, 1.0, null);
  }

  global.DetCheetahRenderer = {
    cheetahFkSegments,
    renderCheetahStaticCanvas,
    renderCheetahLiveFrame,
  };
})(window);
