// Zero-dependency WebGL2 3D MuJoCo geometry renderer for Direct Evolution Together.
// Renders native MuJoCo MjModel geoms (plane, sphere, capsule, ellipsoid, cylinder, box)
// driven directly by MuJoCo MjData `geom_xpos` and `geom_xmat` poses.
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

    // Apply MuJoCo row-major 3x3 rotation + translation
    vec3 mjWorld = vec3(
      dot(u_xmat[0], localPos),
      dot(u_xmat[1], localPos),
      dot(u_xmat[2], localPos)
    ) + u_xpos;

    if (u_shadow > 0.5) {
      float h = max(0.0, mjWorld.z);
      mjWorld.x -= h * 0.28;
      mjWorld.y += h * 0.32;
      mjWorld.z = 0.0025;
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

  const VERT_FLOOR_SRC = `#version 300 es
  precision highp float;
  layout(location = 0) in vec2 a_xz;
  uniform mat4 u_viewProj;
  uniform float u_focusX;
  out vec2 v_worldXZ;
  void main() {
    vec3 p = vec3(a_xz.x + u_focusX, 0.0, a_xz.y);
    v_worldXZ = p.xz;
    gl_Position = u_viewProj * vec4(p, 1.0);
  }`;

  const FRAG_FLOOR_SRC = `#version 300 es
  precision highp float;
  in vec2 v_worldXZ;
  uniform float u_focusX;
  out vec4 outColor;
  void main() {
    vec2 cell = floor(v_worldXZ * 2.0);
    float checker = mod(cell.x + cell.y, 2.0);
    vec3 cA = vec3(0.062, 0.068, 0.082);
    vec3 cB = vec3(0.088, 0.096, 0.116);
    vec3 col = mix(cA, cB, checker);

    // 1.0m major grid hairline
    vec2 g = abs(fract(v_worldXZ - 0.5) - 0.5) / max(vec2(1e-4), fwidth(v_worldXZ));
    float line = 1.0 - min(min(g.x, g.y), 1.0);
    col = mix(col, vec3(0.18, 0.22, 0.28), line * 0.45);

    // X-axis travel strip along z_gl = 0 (y_mj = 0)
    float axisDist = abs(v_worldXZ.y) / max(1e-4, fwidth(v_worldXZ.y));
    float axisLine = 1.0 - min(axisDist * 0.7, 1.0);
    col = mix(col, vec3(0.20, 0.65, 0.50), axisLine * 0.35);

    float d = length(vec2(v_worldXZ.x - u_focusX, v_worldXZ.y));
    float fog = smoothstep(4.5, 16.0, d);
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
    // Builds unit sphere split by equator into +cap (cap=+1) and -cap (cap=-1) plus connecting rings.
    const pos = [];
    const norm = [];
    const cap = [];
    const idx = [];
    const rings = [];
    // Top cap: theta 0 .. pi/2 with cap = +1
    for (let r = 0; r <= hemiRings; r++) {
      rings.push({ phi: (r / hemiRings) * (Math.PI * 0.5), c: 1.0 });
    }
    // Bottom cap: theta pi/2 .. pi with cap = -1
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
    constructor(canvas) {
      this.canvas = canvas;
      this.yaw = -0.26;
      this.pitch = 0.22;
      this.distance = 2.65;
      this.dragging = false;
      this.lastPointer = null;
      this.gl = null;
      this.ready = false;
      this._tmpMat3 = new Float32Array(9);
      this._onContextChange = null;
      this._initGl();
      this._bindInput();
    }

    _initGl() {
      if (!this.canvas || typeof this.canvas.getContext !== "function") return;
      const gl = this.canvas.getContext("webgl2", { antialias: true, preserveDrawingBuffer: true });
      if (!gl) return;
      this.gl = gl;
      this.geomProg = linkProgram(gl, VERT_GEOM_SRC, FRAG_GEOM_SRC);
      this.floorProg = linkProgram(gl, VERT_FLOOR_SRC, FRAG_FLOOR_SRC);

      this.capsuleMesh = this._uploadMesh(buildCapsuleGeometry(22, 8));
      this.boxMesh = this._uploadMesh(buildBoxGeometry());

      // Floor quad centered around 0
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
      this.uFloor = {
        viewProj: gl.getUniformLocation(this.floorProg, "u_viewProj"),
        focusX: gl.getUniformLocation(this.floorProg, "u_focusX"),
      };
      this.ready = true;
    }

    _uploadMesh(mesh) {
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
        if (this._lastRender) {
          this.renderFrame(this._lastRender.geomSpecs, this._lastRender.poseRow, this._lastRender.focusX);
        }
      };
      this._onUp = () => {
        this.dragging = false;
        this.lastPointer = null;
      };
      this._onWheel = (e) => {
        if (!this._lastRender) return;
        e.preventDefault();
        this.distance = Math.max(1.2, Math.min(6.0, this.distance * (1.0 + e.deltaY * 0.0012)));
        this.renderFrame(this._lastRender.geomSpecs, this._lastRender.poseRow, this._lastRender.focusX);
      };
      this._onDblClick = () => {
        this.yaw = -0.26;
        this.pitch = 0.22;
        this.distance = 2.65;
        if (this._lastRender) {
          this.renderFrame(this._lastRender.geomSpecs, this._lastRender.poseRow, this._lastRender.focusX);
        }
      };
      this.canvas.addEventListener("pointerdown", this._onDown);
      window.addEventListener("pointermove", this._onMove);
      window.addEventListener("pointerup", this._onUp);
      this.canvas.addEventListener("wheel", this._onWheel, { passive: false });
      this.canvas.addEventListener("dblclick", this._onDblClick);
    }

    renderFrame(geomSpecs, poseRow, focusX = 0.0) {
      if (!this.ready || !this.gl || !Array.isArray(geomSpecs) || !Array.isArray(poseRow)) return false;
      this._lastRender = { geomSpecs, poseRow, focusX };

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
      const target = [focusX, 0.56, 0.0];
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
      gl.uniform1f(this.uFloor.focusX, focusX);
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
      if (this.canvas) {
        this.canvas.removeEventListener("pointerdown", this._onDown);
        this.canvas.removeEventListener("wheel", this._onWheel);
        this.canvas.removeEventListener("dblclick", this._onDblClick);
      }
      window.removeEventListener("pointermove", this._onMove);
      window.removeEventListener("pointerup", this._onUp);
      if (this.gl) {
        const gl = this.gl;
        [...(this.capsuleMesh?.buffers || []), ...(this.boxMesh?.buffers || []), ...(this._floorBuffers || [])].forEach((b) => gl.deleteBuffer(b));
        if (this.capsuleMesh?.vao) gl.deleteVertexArray(this.capsuleMesh.vao);
        if (this.boxMesh?.vao) gl.deleteVertexArray(this.boxMesh.vao);
        if (this.floorVao) gl.deleteVertexArray(this.floorVao);
        if (this.geomProg) gl.deleteProgram(this.geomProg);
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

  global.DetMujocoGL = { DetMujocoViewer };
})(typeof window !== "undefined" ? window : globalThis);
