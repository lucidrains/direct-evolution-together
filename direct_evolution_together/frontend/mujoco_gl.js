// Zero-dependency WebGL2 3D MuJoCo renderer for Direct Evolution Together.
// Supports both:
//   1) General articulated STL/OBJ body meshpacks (`DETVIS1` visual.bin) driven by `(xpos, xquat)`
//   2) Native primitive MuJoCo geoms (capsule, sphere, ellipsoid, cylinder, box) driven by `(geom_xpos, geom_xmat)`
(function (global) {
  "use strict";

  const VERT_GEOM_SRC = `#version 300 es
  precision highp float;
  layout(location = 0) in vec3 a_pos;
  layout(location = 1) in vec3 a_norm;
  layout(location = 2) in float a_cap;

  uniform mat4 u_viewProj;
  uniform vec3 u_xpos;
  uniform mat3 u_xmat; // MuJoCo row-major 3x3 rotation
  uniform vec3 u_size;
  uniform int u_mode; // 0 = capsule, 1 = ellipsoid/sphere, 2 = box/cylinder
  uniform float u_shadow; // 0.0 = solid geom, 1.0 = ground contact shadow

  out vec3 v_normGl;
  out vec3 v_worldGl;

  vec3 toGl(vec3 v) {
    return vec3(v.x, v.z, -v.y);
  }

  void main() {
    vec3 localPos;
    vec3 localNorm = a_norm;
    if (u_mode == 0) {
      float r = max(0.002, u_size.x);
      float hl = max(0.0, u_size.y);
      localPos = vec3(a_pos.x * r, a_pos.y * r, a_pos.z * r + a_cap * hl);
    } else if (u_mode == 1) {
      vec3 sc = vec3(max(0.002, u_size.x), max(0.002, u_size.y > 0.0 ? u_size.y : u_size.x), max(0.002, u_size.z > 0.0 ? u_size.z : u_size.x));
      localPos = a_pos * sc;
      localNorm = normalize(a_norm / sc);
    } else {
      vec3 sc = vec3(max(0.002, u_size.x), max(0.002, u_size.y > 0.0 ? u_size.y : u_size.x), max(0.002, u_size.z > 0.0 ? u_size.z : u_size.y));
      localPos = a_pos * sc;
    }

    vec3 mjWorld = vec3(
      dot(u_xmat[0], localPos),
      dot(u_xmat[1], localPos),
      dot(u_xmat[2], localPos)
    ) + u_xpos;

    if (u_shadow > 0.5) {
      float h = max(0.0, mjWorld.z);
      mjWorld.x -= h * 0.28;
      mjWorld.y += h * 0.32;
      mjWorld.z = 0.0015;
      v_normGl = vec3(0.0, 1.0, 0.0);
    } else {
      vec3 mjNorm = vec3(
        dot(u_xmat[0], localNorm),
        dot(u_xmat[1], localNorm),
        dot(u_xmat[2], localNorm)
      );
      v_normGl = normalize(toGl(mjNorm));
    }

    vec3 glPos = toGl(mjWorld);
    v_worldGl = glPos;
    gl_Position = u_viewProj * vec4(glPos, 1.0);
  }`;

  const FRAG_GEOM_SRC = `#version 300 es
  precision highp float;
  in vec3 v_normGl;
  in vec3 v_worldGl;

  uniform vec4 u_rgba;
  uniform vec3 u_camGl;
  uniform float u_shadow;

  out vec4 outColor;

  void main() {
    if (u_shadow > 0.5) {
      outColor = vec4(0.02, 0.02, 0.03, 0.42);
      return;
    }
    vec3 N = normalize(v_normGl);
    vec3 V = normalize(u_camGl - v_worldGl);
    vec3 L1 = normalize(vec3(0.45, 0.85, 0.42));
    vec3 L2 = normalize(vec3(-0.55, 0.35, -0.6));

    float diff1 = max(dot(N, L1), 0.0);
    float diff2 = max(dot(N, L2), 0.0);
    float hemi = 0.5 + 0.5 * N.y;
    vec3 amb = mix(vec3(0.11, 0.12, 0.15), vec3(0.24, 0.30, 0.38), hemi);

    vec3 H = normalize(L1 + V);
    float spec = pow(max(dot(N, H), 0.0), 36.0) * 0.32;
    float rim = pow(1.0 - max(dot(N, V), 0.0), 2.8) * 0.14;

    vec3 col = u_rgba.rgb * (amb + diff1 * 0.82 + diff2 * 0.25) + vec3(spec) + vec3(0.22, 0.58, 0.85) * rim;
    outColor = vec4(col, u_rgba.a);
  }`;

  const VERT_BODY_SRC = `#version 300 es
  precision highp float;
  layout(location = 0) in vec3 a_pos;
  layout(location = 1) in vec3 a_norm;
  layout(location = 2) in vec4 a_rgba;

  uniform mat4 u_viewProj;
  uniform vec3 u_xpos;
  uniform mat3 u_xmat;
  uniform float u_shadow;

  out vec3 v_normGl;
  out vec3 v_worldGl;
  out vec4 v_rgba;

  vec3 toGl(vec3 v) {
    return vec3(v.x, v.z, -v.y);
  }

  void main() {
    vec3 mjWorld = vec3(
      dot(u_xmat[0], a_pos),
      dot(u_xmat[1], a_pos),
      dot(u_xmat[2], a_pos)
    ) + u_xpos;

    if (u_shadow > 0.5) {
      float h = max(0.0, mjWorld.z);
      mjWorld.x -= h * 0.26;
      mjWorld.y += h * 0.30;
      mjWorld.z = 0.0012;
      v_normGl = vec3(0.0, 1.0, 0.0);
    } else {
      vec3 mjNorm = vec3(
        dot(u_xmat[0], a_norm),
        dot(u_xmat[1], a_norm),
        dot(u_xmat[2], a_norm)
      );
      v_normGl = normalize(toGl(mjNorm));
    }

    v_rgba = a_rgba;
    vec3 glPos = toGl(mjWorld);
    v_worldGl = glPos;
    gl_Position = u_viewProj * vec4(glPos, 1.0);
  }`;

  const FRAG_BODY_SRC = `#version 300 es
  precision highp float;
  in vec3 v_normGl;
  in vec3 v_worldGl;
  in vec4 v_rgba;

  uniform vec3 u_camGl;
  uniform float u_shadow;

  out vec4 outColor;

  void main() {
    if (u_shadow > 0.5) {
      outColor = vec4(0.02, 0.02, 0.03, 0.40);
      return;
    }
    vec3 N = normalize(gl_FrontFacing ? v_normGl : -v_normGl);
    vec3 V = normalize(u_camGl - v_worldGl);
    vec3 L1 = normalize(vec3(0.48, 0.86, 0.40));
    vec3 L2 = normalize(vec3(-0.58, 0.38, -0.56));

    float diff1 = max(dot(N, L1), 0.0);
    float diff2 = max(dot(N, L2), 0.0);
    float hemi = 0.5 + 0.5 * N.y;
    vec3 amb = mix(vec3(0.14, 0.15, 0.18), vec3(0.28, 0.33, 0.40), hemi);

    vec3 H = normalize(L1 + V);
    float spec = pow(max(dot(N, H), 0.0), 32.0) * 0.26;
    float rim = pow(1.0 - max(dot(N, V), 0.0), 2.8) * 0.12;

    vec3 baseRgb = max(v_rgba.rgb, vec3(0.14, 0.15, 0.18));
    vec3 col = baseRgb * (amb + diff1 * 0.80 + diff2 * 0.28) + vec3(spec) + vec3(0.22, 0.58, 0.85) * rim;
    outColor = vec4(col, v_rgba.a);
  }`;

  const VERT_FLOOR_SRC = `#version 300 es
  precision highp float;
  layout(location = 0) in vec2 a_xz;
  uniform mat4 u_viewProj;
  uniform vec2 u_focusXZ;
  out vec2 v_worldXZ;
  void main() {
    vec3 p = vec3(a_xz.x + u_focusXZ.x, 0.0, a_xz.y + u_focusXZ.y);
    v_worldXZ = p.xz;
    gl_Position = u_viewProj * vec4(p, 1.0);
  }`;

  const FRAG_FLOOR_SRC = `#version 300 es
  precision highp float;
  in vec2 v_worldXZ;
  uniform vec2 u_focusXZ;
  uniform float u_gridScale;
  out vec4 outColor;
  void main() {
    float gs = max(0.05, u_gridScale);
    vec2 scaled = v_worldXZ / gs;
    vec2 cell = floor(scaled * 2.0);
    float checker = mod(cell.x + cell.y, 2.0);
    vec3 cA = vec3(0.062, 0.068, 0.082);
    vec3 cB = vec3(0.088, 0.096, 0.116);
    vec3 col = mix(cA, cB, checker);

    vec2 g = abs(fract(scaled - 0.5) - 0.5) / max(vec2(1e-4), fwidth(scaled));
    float line = 1.0 - min(min(g.x, g.y), 1.0);
    col = mix(col, vec3(0.18, 0.22, 0.28), line * 0.45);

    float axisDist = abs(v_worldXZ.y) / max(1e-4, fwidth(v_worldXZ.y));
    float axisLine = 1.0 - min(axisDist * 0.7, 1.0);
    col = mix(col, vec3(0.20, 0.65, 0.50), axisLine * 0.38);

    float d = length(v_worldXZ - u_focusXZ);
    float fog = smoothstep(4.5 * gs, 16.0 * gs, d);
    col = mix(col, vec3(0.035, 0.035, 0.043), fog);
    outColor = vec4(col, 1.0);
  }`;

  function compileShader(gl, type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(sh);
      gl.deleteShader(sh);
      throw new Error("Shader error: " + log);
    }
    return sh;
  }

  function linkProgram(gl, vSrc, fSrc) {
    const vs = compileShader(gl, gl.VERTEX_SHADER, vSrc);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, fSrc);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog);
      gl.deleteProgram(prog);
      throw new Error("Program link error: " + log);
    }
    return prog;
  }

  function buildCapsuleGeometry(slices = 20, hemiRings = 8) {
    const pos = [];
    const norm = [];
    const cap = [];
    const idx = [];
    const rings = [];
    for (let r = 0; r <= hemiRings; r++) {
      rings.push({ phi: (r / hemiRings) * (Math.PI * 0.5), c: 1.0 });
    }
    for (let r = 0; r <= hemiRings; r++) {
      rings.push({ phi: Math.PI * 0.5 + (r / hemiRings) * (Math.PI * 0.5), c: -1.0 });
    }
    const stride = slices + 1;
    for (let r = 0; r < rings.length; r++) {
      const { phi, c } = rings[r];
      const sp = Math.sin(phi);
      const cp = Math.cos(phi);
      for (let s = 0; s <= slices; s++) {
        const th = (s / slices) * Math.PI * 2.0;
        const x = sp * Math.cos(th);
        const y = sp * Math.sin(th);
        const z = cp;
        pos.push(x, y, z);
        norm.push(x, y, z);
        cap.push(c);
      }
    }
    for (let r = 0; r < rings.length - 1; r++) {
      for (let s = 0; s < slices; s++) {
        const a = r * stride + s;
        const b = (r + 1) * stride + s;
        const d = (r + 1) * stride + s + 1;
        const e = r * stride + s + 1;
        idx.push(a, b, d, a, d, e);
      }
    }
    return {
      pos: new Float32Array(pos),
      norm: new Float32Array(norm),
      cap: new Float32Array(cap),
      idx: new Uint16Array(idx),
    };
  }

  function buildBoxGeometry() {
    const faces = [
      [[0, 0, 1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
      [[0, 0, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1], [1, 1, -1]],
      [[1, 0, 0], [1, -1, 1], [1, -1, -1], [1, 1, -1], [1, 1, 1]],
      [[-1, 0, 0], [-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]],
      [[0, 1, 0], [-1, 1, 1], [1, 1, 1], [1, 1, -1], [-1, 1, -1]],
      [[0, -1, 0], [-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]],
    ];
    const pos = [];
    const norm = [];
    const cap = [];
    const idx = [];
    let v = 0;
    for (const [n, a, b, c, d] of faces) {
      [a, b, c, d].forEach((p) => {
        pos.push(p[0], p[1], p[2]);
        norm.push(n[0], n[1], n[2]);
        cap.push(0);
      });
      idx.push(v, v + 1, v + 2, v, v + 2, v + 3);
      v += 4;
    }
    return {
      pos: new Float32Array(pos),
      norm: new Float32Array(norm),
      cap: new Float32Array(cap),
      idx: new Uint16Array(idx),
    };
  }

  function quatToRowMat3(qw, qx, qy, qz, out) {
    const n = Math.hypot(qw, qx, qy, qz) || 1.0;
    const w = qw / n, x = qx / n, y = qy / n, z = qz / n;
    const xx = x * x, yy = y * y, zz = z * z;
    const xy = x * y, xz = x * z, yz = y * z;
    const wx = w * x, wy = w * y, wz = w * z;
    out[0] = 1.0 - 2.0 * (yy + zz);
    out[1] = 2.0 * (xy - wz);
    out[2] = 2.0 * (xz + wy);
    out[3] = 2.0 * (xy + wz);
    out[4] = 1.0 - 2.0 * (xx + zz);
    out[5] = 2.0 * (yz - wx);
    out[6] = 2.0 * (xz - wy);
    out[7] = 2.0 * (yz + wx);
    out[8] = 1.0 - 2.0 * (xx + yy);
    return out;
  }

  function mat4Perspective(fovy, aspect, near, far) {
    const f = 1.0 / Math.tan(fovy * 0.5);
    const nf = 1.0 / (near - far);
    return new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far + near) * nf, -1,
      0, 0, 2 * far * near * nf, 0,
    ]);
  }

  function mat4LookAt(eye, target, up) {
    let zx = eye[0] - target[0], zy = eye[1] - target[1], zz = eye[2] - target[2];
    let len = Math.hypot(zx, zy, zz) || 1;
    zx /= len; zy /= len; zz /= len;
    let xx = up[1] * zz - up[2] * zy;
    let xy = up[2] * zx - up[0] * zz;
    let xz = up[0] * zy - up[1] * zx;
    len = Math.hypot(xx, xy, xz) || 1;
    xx /= len; xy /= len; xz /= len;
    const yx = zy * xz - zz * xy;
    const yy = zz * xx - zx * xz;
    const yz = zx * xy - zy * xx;
    return new Float32Array([
      xx, yx, zx, 0,
      xy, yy, zy, 0,
      xz, yz, zz, 0,
      -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
      -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
      -(zx * eye[0] + zy * eye[1] + zz * eye[2]),
      1,
    ]);
  }

  function mat4Mul(a, b) {
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        out[c * 4 + r] =
          a[0 * 4 + r] * b[c * 4 + 0] +
          a[1 * 4 + r] * b[c * 4 + 1] +
          a[2 * 4 + r] * b[c * 4 + 2] +
          a[3 * 4 + r] * b[c * 4 + 3];
      }
    }
    return out;
  }

  class DetMujocoViewer {
    constructor(canvas, interactive = true) {
      this.canvas = canvas;
      this.interactive = interactive;
      this.yaw = -0.26;
      this.pitch = 0.22;
      this.distance = 2.65;
      this._defaultCam = { yaw: -0.26, pitch: 0.22, distance: 2.65, target_height: 0.56, grid_scale: 1.0 };
      this._activeSceneKey = "";
      this._sceneVaos = new Map(); // `${sceneId}:${sha}` -> Map<bodyId, { vao, count, buffers }>
      this.dragging = false;
      this.lastPointer = null;
      this.gl = null;
      this.ready = false;
      this._tmpMat3 = new Float32Array(9);
      this._initGl();
      if (interactive) this._bindInput();
    }

    _initGl() {
      if (!this.canvas || typeof this.canvas.getContext !== "function") return;
      const gl = this.canvas.getContext("webgl2", {
        antialias: true,
        preserveDrawingBuffer: false,
        powerPreference: "high-performance",
      });
      if (!gl) return;
      this.gl = gl;
      this.geomProg = linkProgram(gl, VERT_GEOM_SRC, FRAG_GEOM_SRC);
      this.bodyProg = linkProgram(gl, VERT_BODY_SRC, FRAG_BODY_SRC);
      this.floorProg = linkProgram(gl, VERT_FLOOR_SRC, FRAG_FLOOR_SRC);

      this.capsuleMesh = this._uploadPrimMesh(buildCapsuleGeometry(22, 8));
      this.boxMesh = this._uploadPrimMesh(buildBoxGeometry());

      const fVerts = new Float32Array([-22, -22, 22, -22, 22, 22, -22, 22]);
      const fIdx = new Uint16Array([0, 1, 2, 0, 2, 3]);
      this.floorVao = gl.createVertexArray();
      gl.bindVertexArray(this.floorVao);
      const vb = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vb);
      gl.bufferData(gl.ARRAY_BUFFER, fVerts, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      const ib = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, fIdx, gl.STATIC_DRAW);
      gl.bindVertexArray(null);
      this._floorBuffers = [vb, ib];

      this.uGeom = {
        viewProj: gl.getUniformLocation(this.geomProg, "u_viewProj"),
        xpos: gl.getUniformLocation(this.geomProg, "u_xpos"),
        xmat: gl.getUniformLocation(this.geomProg, "u_xmat"),
        size: gl.getUniformLocation(this.geomProg, "u_size"),
        mode: gl.getUniformLocation(this.geomProg, "u_mode"),
        rgba: gl.getUniformLocation(this.geomProg, "u_rgba"),
        camGl: gl.getUniformLocation(this.geomProg, "u_camGl"),
        shadow: gl.getUniformLocation(this.geomProg, "u_shadow"),
      };
      this.uBody = {
        viewProj: gl.getUniformLocation(this.bodyProg, "u_viewProj"),
        xpos: gl.getUniformLocation(this.bodyProg, "u_xpos"),
        xmat: gl.getUniformLocation(this.bodyProg, "u_xmat"),
        camGl: gl.getUniformLocation(this.bodyProg, "u_camGl"),
        shadow: gl.getUniformLocation(this.bodyProg, "u_shadow"),
      };
      this.uFloor = {
        viewProj: gl.getUniformLocation(this.floorProg, "u_viewProj"),
        focusXZ: gl.getUniformLocation(this.floorProg, "u_focusXZ"),
        grid_scale: gl.getUniformLocation(this.floorProg, "u_gridScale"),
      };
      this._sceneVaos.clear();
      this.ready = true;
    }

    _uploadPrimMesh(mesh) {
      const gl = this.gl;
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      const bindAttr = (loc, data, size) => {
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
        return buf;
      };
      const b0 = bindAttr(0, mesh.pos, 3);
      const b1 = bindAttr(1, mesh.norm, 3);
      const b2 = bindAttr(2, mesh.cap, 1);
      const ib = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.idx, gl.STATIC_DRAW);
      gl.bindVertexArray(null);
      return { vao, count: mesh.idx.length, buffers: [b0, b1, b2, ib] };
    }

    _ensureSceneVaos(visualPack) {
      const key = `${visualPack.scene_id}:${visualPack.sha256 || ""}`;
      if (this._sceneVaos.has(key)) return this._sceneVaos.get(key);
      const gl = this.gl;
      const byBody = new Map();
      for (const bm of visualPack.bodyMeshes || []) {
        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);

        const bPos = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, bPos);
        gl.bufferData(gl.ARRAY_BUFFER, bm.pos, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

        const bNorm = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, bNorm);
        gl.bufferData(gl.ARRAY_BUFFER, bm.norm, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 3, gl.BYTE, true, 0, 0);

        const bRgba = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, bRgba);
        gl.bufferData(gl.ARRAY_BUFFER, bm.rgba, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(2);
        gl.vertexAttribPointer(2, 4, gl.UNSIGNED_BYTE, true, 0, 0);

        const bIdx = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, bIdx);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, bm.idx, gl.STATIC_DRAW);

        gl.bindVertexArray(null);
        byBody.set(bm.bodyId, { vao, count: bm.idx.length, buffers: [bPos, bNorm, bRgba, bIdx] });
      }
      this._sceneVaos.set(key, byBody);
      return byBody;
    }

    _applySceneCameraIfChanged(sceneKey, camDefaults) {
      if (this._activeSceneKey === sceneKey) return;
      this._activeSceneKey = sceneKey;
      const cd = camDefaults || {};
      this._defaultCam = {
        yaw: typeof cd.yaw === "number" ? cd.yaw : -0.26,
        pitch: typeof cd.pitch === "number" ? cd.pitch : 0.22,
        distance: typeof cd.distance === "number" ? cd.distance : 2.65,
        target_height: typeof cd.target_height === "number" ? cd.target_height : 0.56,
        grid_scale: typeof cd.grid_scale === "number" ? cd.grid_scale : 1.0,
      };
      this.yaw = this._defaultCam.yaw;
      this.pitch = this._defaultCam.pitch;
      this.distance = this._defaultCam.distance;
    }

    _redrawLast() {
      if (!this._lastRender) return;
      if (this._lastRender.kind === "bodies") {
        this.renderSceneBodies(this._lastRender.visualPack, this._lastRender.bodyPoseRow);
      } else {
        this.renderFrame(this._lastRender.geom_specs, this._lastRender.poseRow, this._lastRender.focusX);
      }
    }

    _bindInput() {
      if (!this.canvas || typeof this.canvas.addEventListener !== "function") return;
      this._onDown = (e) => {
        this.dragging = true;
        this.lastPointer = { x: e.clientX, y: e.clientY };
      };
      this._onMove = (e) => {
        if (!this.dragging || !this.lastPointer) return;
        const dx = e.clientX - this.lastPointer.x;
        const dy = e.clientY - this.lastPointer.y;
        this.lastPointer = { x: e.clientX, y: e.clientY };
        this.yaw -= dx * 0.008;
        this.pitch = Math.max(0.06, Math.min(1.35, this.pitch + dy * 0.008));
        this._redrawLast();
      };
      this._onUp = () => {
        this.dragging = false;
        this.lastPointer = null;
      };
      this._onWheel = (e) => {
        if (!this._lastRender) return;
        e.preventDefault();
        this.distance = Math.max(0.22, Math.min(6.0, this.distance * (1.0 + e.deltaY * 0.0012)));
        this._redrawLast();
      };
      this._onDblClick = () => {
        this.yaw = this._defaultCam.yaw;
        this.pitch = this._defaultCam.pitch;
        this.distance = this._defaultCam.distance;
        this._redrawLast();
      };
      this._onContextLost = (e) => {
        e.preventDefault();
        this.ready = false;
        this.gl = null;
      };
      this._onContextRestored = () => {
        this._initGl();
        if (this.ready) this._redrawLast();
      };
      this.canvas.addEventListener("pointerdown", this._onDown);
      window.addEventListener("pointermove", this._onMove);
      window.addEventListener("pointerup", this._onUp);
      this.canvas.addEventListener("wheel", this._onWheel, { passive: false });
      this.canvas.addEventListener("dblclick", this._onDblClick);
      this.canvas.addEventListener("webglcontextlost", this._onContextLost);
      this.canvas.addEventListener("webglcontextrestored", this._onContextRestored);
    }

    renderSceneBodies(visualPack, bodyPoseRow, cameraOverride = null) {
      if (!this.ready || !this.gl || !visualPack || !Array.isArray(bodyPoseRow) && !(bodyPoseRow instanceof Float32Array)) {
        return false;
      }
      this._applySceneCameraIfChanged(`pack:${visualPack.scene_id}`, visualPack.camera_defaults);
      this._lastRender = { kind: "bodies", visualPack, bodyPoseRow };

      const gl = this.gl;
      const w = this.canvas.width || 512;
      const h = this.canvas.height || 340;
      gl.viewport(0, 0, w, h);
      gl.clearColor(0.035, 0.035, 0.043, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);

      const trackId = Math.max(1, Number(visualPack.camera_defaults?.track_body_id || 1));
      const trackOff = (trackId - 1) * 7;
      const camOff = Array.isArray(visualPack.camera_defaults?.target_offset)
        ? visualPack.camera_defaults.target_offset
        : [0.0, 0.0];
      const focusX = (bodyPoseRow.length > trackOff ? Number(bodyPoseRow[trackOff] || 0) : 0.0) + Number(camOff[0] || 0);
      const focusY = (bodyPoseRow.length > trackOff + 1 ? Number(bodyPoseRow[trackOff + 1] || 0) : 0.0) + Number(camOff[1] || 0);
      const focusGlZ = -focusY;

      const yaw = cameraOverride && typeof cameraOverride.yaw === "number" ? cameraOverride.yaw : this.yaw;
      const pitch = cameraOverride && typeof cameraOverride.pitch === "number" ? cameraOverride.pitch : this.pitch;
      const dist = cameraOverride && typeof cameraOverride.distance === "number" ? cameraOverride.distance : this.distance;
      const targetH = this._defaultCam.target_height;

      const cp = Math.cos(pitch);
      const sp = Math.sin(pitch);
      const cy = Math.cos(yaw);
      const sy = Math.sin(yaw);
      const target = [focusX, targetH, focusGlZ];
      const eye = [
        target[0] + dist * cp * sy,
        target[1] + dist * sp,
        target[2] + dist * cp * cy,
      ];
      const proj = mat4Perspective(0.66, w / Math.max(1, h), 0.02, 40.0);
      const view = mat4LookAt(eye, target, [0, 1, 0]);
      const vp = mat4Mul(proj, view);

      gl.useProgram(this.floorProg);
      gl.uniformMatrix4fv(this.uFloor.viewProj, false, vp);
      gl.uniform2f(this.uFloor.focusXZ, focusX, focusGlZ);
      gl.uniform1f(this.uFloor.grid_scale, this._defaultCam.grid_scale);
      gl.bindVertexArray(this.floorVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);

      const byBody = this._ensureSceneVaos(visualPack);
      gl.useProgram(this.bodyProg);
      gl.uniformMatrix4fv(this.uBody.viewProj, false, vp);
      gl.uniform3f(this.uBody.camGl, eye[0], eye[1], eye[2]);

      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      this._drawBodiesPass(byBody, bodyPoseRow, 1.0);
      gl.depthMask(true);
      gl.disable(gl.BLEND);
      this._drawBodiesPass(byBody, bodyPoseRow, 0.0);
      return true;
    }

    _drawBodiesPass(byBody, bodyPoseRow, shadowFlag) {
      const gl = this.gl;
      const m3 = this._tmpMat3;
      gl.uniform1f(this.uBody.shadow, shadowFlag);
      for (const [bodyId, mesh] of byBody.entries()) {
        const off = (bodyId - 1) * 7;
        if (off + 6 >= bodyPoseRow.length) continue;
        gl.uniform3f(this.uBody.xpos, bodyPoseRow[off], bodyPoseRow[off + 1], bodyPoseRow[off + 2]);
        quatToRowMat3(
          bodyPoseRow[off + 3],
          bodyPoseRow[off + 4],
          bodyPoseRow[off + 5],
          bodyPoseRow[off + 6],
          m3
        );
        gl.uniformMatrix3fv(this.uBody.xmat, false, m3);
        gl.bindVertexArray(mesh.vao);
        gl.drawElements(gl.TRIANGLES, mesh.count, gl.UNSIGNED_SHORT, 0);
      }
    }

    renderFrame(geomSpecs, poseRow, focusX = 0.0) {
      if (!this.ready || !this.gl || !Array.isArray(geomSpecs) || !Array.isArray(poseRow)) return false;
      this._applySceneCameraIfChanged("prim:cheetah", {
        yaw: -0.26,
        pitch: 0.22,
        distance: 2.65,
        target_height: 0.56,
        grid_scale: 1.0,
      });
      this._lastRender = { kind: "geoms", geom_specs: geomSpecs, poseRow, focusX };

      const gl = this.gl;
      const w = this.canvas.width || 512;
      const h = this.canvas.height || 340;
      gl.viewport(0, 0, w, h);
      gl.clearColor(0.035, 0.035, 0.043, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);

      const cp = Math.cos(this.pitch);
      const sp = Math.sin(this.pitch);
      const cy = Math.cos(this.yaw);
      const sy = Math.sin(this.yaw);
      const target = [focusX, this._defaultCam.target_height, 0.0];
      const eye = [
        target[0] + this.distance * cp * sy,
        target[1] + this.distance * sp,
        target[2] + this.distance * cp * cy,
      ];
      const proj = mat4Perspective(0.68, w / Math.max(1, h), 0.05, 60.0);
      const view = mat4LookAt(eye, target, [0, 1, 0]);
      const vp = mat4Mul(proj, view);

      gl.useProgram(this.floorProg);
      gl.uniformMatrix4fv(this.uFloor.viewProj, false, vp);
      gl.uniform2f(this.uFloor.focusXZ, focusX, 0.0);
      gl.uniform1f(this.uFloor.grid_scale, this._defaultCam.grid_scale);
      gl.bindVertexArray(this.floorVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);

      gl.useProgram(this.geomProg);
      gl.uniformMatrix4fv(this.uGeom.viewProj, false, vp);
      gl.uniform3f(this.uGeom.camGl, eye[0], eye[1], eye[2]);

      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      this._drawGeomsPass(geomSpecs, poseRow, 1.0);
      gl.depthMask(true);
      gl.disable(gl.BLEND);
      this._drawGeomsPass(geomSpecs, poseRow, 0.0);
      return true;
    }

    _drawGeomsPass(geomSpecs, poseRow, shadowFlag) {
      const gl = this.gl;
      gl.uniform1f(this.uGeom.shadow, shadowFlag);
      for (let i = 0; i < geomSpecs.length; i++) {
        const base = i * 12;
        if (base + 11 >= poseRow.length) break;
        const spec = geomSpecs[i];
        const sz = spec.size || [0.05, 0.1, 0.05];
        const rgba = spec.rgba || [0.8, 0.65, 0.45, 1.0];
        const isBox = spec.type === 6;
        const mode = isBox ? 2 : spec.type === 3 ? 0 : 1;
        const mesh = isBox ? this.boxMesh : this.capsuleMesh;

        gl.uniform1i(this.uGeom.mode, mode);
        gl.uniform3f(this.uGeom.size, sz[0] || 0.05, sz[1] || 0.1, sz[2] || 0.05);
        gl.uniform4f(this.uGeom.rgba, rgba[0], rgba[1], rgba[2], rgba[3]);
        gl.uniform3f(this.uGeom.xpos, poseRow[base], poseRow[base + 1], poseRow[base + 2]);
        const m3 = this._tmpMat3;
        for (let k = 0; k < 9; k++) m3[k] = poseRow[base + 3 + k];
        gl.uniformMatrix3fv(this.uGeom.xmat, false, m3);
        gl.bindVertexArray(mesh.vao);
        gl.drawElements(gl.TRIANGLES, mesh.count, gl.UNSIGNED_SHORT, 0);
      }
    }

    destroy() {
      if (this.interactive && this.canvas) {
        this.canvas.removeEventListener("pointerdown", this._onDown);
        this.canvas.removeEventListener("wheel", this._onWheel);
        this.canvas.removeEventListener("dblclick", this._onDblClick);
        this.canvas.removeEventListener("webglcontextlost", this._onContextLost);
        this.canvas.removeEventListener("webglcontextrestored", this._onContextRestored);
      }
      if (this.interactive) {
        window.removeEventListener("pointermove", this._onMove);
        window.removeEventListener("pointerup", this._onUp);
      }
      if (this.gl) {
        const gl = this.gl;
        for (const byBody of this._sceneVaos.values()) {
          for (const m of byBody.values()) {
            m.buffers.forEach((b) => gl.deleteBuffer(b));
            gl.deleteVertexArray(m.vao);
          }
        }
        this._sceneVaos.clear();
        [...(this.capsuleMesh?.buffers || []), ...(this.boxMesh?.buffers || []), ...(this._floorBuffers || [])].forEach((b) => gl.deleteBuffer(b));
        if (this.capsuleMesh?.vao) gl.deleteVertexArray(this.capsuleMesh.vao);
        if (this.boxMesh?.vao) gl.deleteVertexArray(this.boxMesh.vao);
        if (this.floorVao) gl.deleteVertexArray(this.floorVao);
        if (this.geomProg) gl.deleteProgram(this.geomProg);
        if (this.bodyProg) gl.deleteProgram(this.bodyProg);
        if (this.floorProg) gl.deleteProgram(this.floorProg);
      }
      this.ready = false;
      this.gl = null;
    }
  }

  DetMujocoViewer._supported = null;
  DetMujocoViewer.isSupported = function () {
    if (DetMujocoViewer._supported !== null) return DetMujocoViewer._supported;
    try {
      const c = document.createElement("canvas");
      const gl = c.getContext("webgl2");
      DetMujocoViewer._supported = Boolean(gl);
      if (gl) gl.getExtension("WEBGL_lose_context")?.loseContext();
    } catch {
      DetMujocoViewer._supported = false;
    }
    return DetMujocoViewer._supported;
  };

  // Shared single-context offscreen renderer for crisp 3D grid cards & hover animations
  let sharedOffscreenCanvas = null;
  let sharedOffscreenViewer = null;

  function ensureSharedCardViewer(width, height) {
    if (!DetMujocoViewer.isSupported()) return null;
    if (!sharedOffscreenCanvas) {
      sharedOffscreenCanvas = document.createElement("canvas");
      sharedOffscreenCanvas.width = width;
      sharedOffscreenCanvas.height = height;
      sharedOffscreenViewer = new DetMujocoViewer(sharedOffscreenCanvas, false);
    }
    if (sharedOffscreenCanvas.width !== width || sharedOffscreenCanvas.height !== height) {
      sharedOffscreenCanvas.width = width;
      sharedOffscreenCanvas.height = height;
    }
    return sharedOffscreenViewer && sharedOffscreenViewer.ready ? sharedOffscreenViewer : null;
  }

  function render2dBodyPoseFallback(ctx, w, h, bodyPoseRow, accent) {
    ctx.fillStyle = "#09090b";
    ctx.fillRect(0, 0, w, h);
    const groundY = Math.round(h * 0.72);
    ctx.strokeStyle = "#3f3f46";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(8, groundY);
    ctx.lineTo(w - 8, groundY);
    ctx.stroke();
    if (!bodyPoseRow || bodyPoseRow.length < 7) return;
    const tx = bodyPoseRow[0];
    const pxPerM = h * 1.9;
    ctx.fillStyle = accent || "#f5c451";
    const nBodies = Math.floor(bodyPoseRow.length / 7);
    for (let b = 0; b < nBodies; b++) {
      const bx = bodyPoseRow[b * 7];
      const bz = bodyPoseRow[b * 7 + 2];
      const sx = w * 0.48 + (bx - tx) * pxPerM;
      const sy = groundY - bz * pxPerM;
      ctx.beginPath();
      ctx.arc(sx, sy, b === 0 ? Math.max(4, w * 0.04) : Math.max(2.5, w * 0.022), 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function renderSceneCardStatic(canvas, visualPack, summary, rollout) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const poses = (rollout && (rollout.body_poses || rollout.bodyPoses)) || [];
    const poseRow = poses.length ? poses[Math.min(poses.length - 1, Math.floor(poses.length * 0.75))] : null;
    const solved = Boolean(summary && summary.solved);
    const ret = summary && typeof summary.return === "number" ? summary.return : (summary?.primary_value ?? summary?.primaryValue ?? -1);
    const accent = solved ? "#34d399" : (ret >= 20 ? "#f5c451" : "#fb7185");

    let drewGl = false;
    if (visualPack && poseRow) {
      const viewer = ensureSharedCardViewer(w, h);
      if (viewer && viewer.renderSceneBodies(visualPack, poseRow)) {
        ctx.drawImage(sharedOffscreenCanvas, 0, 0, w, h);
        drewGl = true;
      }
    }
    if (!drewGl) {
      render2dBodyPoseFallback(ctx, w, h, poseRow, poses.length ? accent : "#52525b");
      if (!poseRow && w > 48) {
        ctx.fillStyle = "#71717a";
        ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.textAlign = "center";
        ctx.fillText("simulating...", w * 0.5, h * 0.48);
        ctx.textAlign = "start";
      }
    }

    // Compact velocity sparkline strip at bottom of grid cards
    const vxs = (rollout && (rollout.vx_trace || rollout.vxTrace)) || [];
    if (w > 60 && vxs.length >= 2) {
      const margin = Math.round(w * 0.07);
      const usableW = w - 2 * margin;
      const waveH = Math.round(h * 0.17);
      const waveTop = h - waveH - Math.round(h * 0.04);
      const midY = waveTop + waveH * 0.5;
      ctx.fillStyle = "rgba(12, 12, 16, 0.78)";
      ctx.fillRect(margin, waveTop, usableW, waveH);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
      ctx.lineWidth = 1;
      ctx.strokeRect(margin, waveTop, usableW, waveH);
      ctx.strokeStyle = accent;
      ctx.lineWidth = 1.35;
      ctx.beginPath();
      const stride = Math.max(1, Math.floor(vxs.length / 50));
      for (let i = 0; i < vxs.length; i += stride) {
        const px = margin + (i / (vxs.length - 1)) * usableW;
        const normV = Math.max(-1, Math.min(1, vxs[i] / 0.35));
        const py = midY - normV * (waveH * 0.42);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
  }

  function renderSceneCardLiveFrame(canvas, visualPack, bodyPoses, stepIdx, maxSteps) {
    const poses = Array.isArray(bodyPoses) ? bodyPoses : (bodyPoses && (bodyPoses.body_poses || bodyPoses.bodyPoses)) || [];
    if (!poses.length) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const idx = Math.min(stepIdx, poses.length - 1);
    const poseRow = poses[idx];
    const viewer = visualPack ? ensureSharedCardViewer(w, h) : null;
    if (viewer && viewer.renderSceneBodies(visualPack, poseRow)) {
      ctx.drawImage(sharedOffscreenCanvas, 0, 0, w, h);
    } else {
      render2dBodyPoseFallback(ctx, w, h, poseRow, "#38bdf8");
    }
    const margin = Math.round(w * 0.06);
    const usableW = w - 2 * margin;
    ctx.fillStyle = "rgba(24, 24, 32, 0.85)";
    ctx.fillRect(margin, 10, usableW, 4);
    ctx.fillStyle = "#38bdf8";
    ctx.fillRect(margin, 10, Math.round(usableW * Math.min(1, idx / Math.max(1, poses.length - 1))), 4);
  }

  global.DetMujocoGL = {
    DetMujocoViewer,
    renderSceneCardStatic,
    renderSceneCardLiveFrame,
  };
})(typeof window !== "undefined" ? window : globalThis);
