// Faithful Gymnasium LunarLander-v3 (Box2D / planck.js) mirror & crisp 2D canvas renderer.
(function (global) {
  "use strict";

  const FPS = 50;
  const DT = 1.0 / FPS;
  const SCALE = 30.0;
  const VIEWPORT_W = 600;
  const VIEWPORT_H = 400;
  const WORLD_W = VIEWPORT_W / SCALE; // 20.0
  const HALF_W = WORLD_W * 0.5; // 10.0
  const WORLD_H = VIEWPORT_H / SCALE; // 13.3333...

  const GRAVITY = -10.0;
  const MAIN_ENGINE_POWER = 13.0;
  const SIDE_ENGINE_POWER = 0.6;
  const INITIAL_RANDOM = 1000.0;

  const LANDER_POLY = [
    [-14, +17],
    [-17, 0],
    [-17, -10],
    [+17, -10],
    [+17, 0],
    [+14, +17],
  ];
  const LEG_AWAY = 20;
  const LEG_DOWN = 18;
  const LEG_W = 2;
  const LEG_H = 8;
  const LEG_SPRING_TORQUE = 40;

  const SIDE_ENGINE_HEIGHT = 14;
  const SIDE_ENGINE_AWAY = 12;
  const MAIN_ENGINE_Y_LOCATION = 4;

  const CHUNKS = 11;
  const HELIPAD_Y = WORLD_H / 4.0;
  const HELIPAD_X1 = (WORLD_W / (CHUNKS - 1)) * (Math.floor(CHUNKS / 2) - 1);
  const HELIPAD_X2 = (WORLD_W / (CHUNKS - 1)) * (Math.floor(CHUNKS / 2) + 1);
  const REST_COM_Y = HELIPAD_Y + LEG_DOWN / SCALE;

  const MAX_STEPS = 1000;
  const SOLVER_VEL_ITERS = 6 * 30;
  const SOLVER_POS_ITERS = 2 * 30;
  const SOLVED_RETURN = 200.0;
  const TIER_MID_RETURN = 80.0;

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296.0;
    };
  }

  function lunarlanderObsFromBody(posX, posY, velX, velY, angle, angVel, leg0, leg1, helipadY = HELIPAD_Y) {
    return [
      (posX - HALF_W) / HALF_W,
      (posY - (helipadY + LEG_DOWN / SCALE)) / (WORLD_H * 0.5),
      (velX * HALF_W) / FPS,
      (velY * (WORLD_H * 0.5)) / FPS,
      angle,
      (20.0 * angVel) / FPS,
      leg0 ? 1.0 : 0.0,
      leg1 ? 1.0 : 0.0,
    ];
  }

  function lunarlanderHeuristic(s) {
    const angleTarg = Math.max(-0.4, Math.min(0.4, s[0] * 0.5 + s[2] * 1.0));
    const hoverTarg = 0.55 * Math.abs(s[0]);
    let angleTodo = (angleTarg - s[4]) * 0.5 - s[5] * 1.0;
    let hoverTodo = (hoverTarg - s[1]) * 0.5 - s[3] * 0.5;
    if (s[6] || s[7]) {
      angleTodo = 0.0;
      hoverTodo = -s[3] * 0.5;
    }
    return [
      Math.max(-1.0, Math.min(1.0, hoverTodo * 20.0 - 1.0)),
      Math.max(-1.0, Math.min(1.0, -angleTodo * 20.0)),
    ];
  }

  function resolveLunarEnvParams(envParams) {
    const p = envParams && typeof envParams === "object" ? envParams : {};
    const rawG = Number(p.gravity);
    const gravity = Number.isFinite(rawG) ? Math.max(-11.99, Math.min(-0.1, rawG)) : GRAVITY;
    const rawW = p.enable_wind;
    const enableWind =
      typeof rawW === "boolean"
        ? rawW
        : /^(1|true|yes|on)$/i.test(String(rawW || "").trim());
    const rawWP = Number(p.wind_power);
    const windPower = Number.isFinite(rawWP) ? Math.max(0.0, Math.min(20.0, rawWP)) : 15.0;
    const rawTP = Number(p.turbulence_power);
    const turbulencePower = Number.isFinite(rawTP) ? Math.max(0.0, Math.min(2.0, rawTP)) : 1.5;
    return { gravity, enableWind, windPower, turbulencePower };
  }

  function createLunarLanderV3Sim(plLib, envParams = null) {
    const pl = plLib || global.planck;
    if (!pl || !pl.World) {
      throw new Error("planck.js Box2D library is not loaded.");
    }
    const Vec2 = pl.Vec2;
    const cfg = resolveLunarEnvParams(envParams);

    const sim = {
      gravity: cfg.gravity,
      enableWind: cfg.enableWind,
      windPower: cfg.windPower,
      turbulencePower: cfg.turbulencePower,
      world: null,
      moon: null,
      lander: null,
      legs: [],
      gameOver: false,
      prevShaping: null,
      chunkX: [],
      smoothY: [],
      helipadX1: HELIPAD_X1,
      helipadX2: HELIPAD_X2,
      helipadY: HELIPAD_Y,
      windIdx: 0,
      torqueIdx: 0,
      rng: mulberry32(0),

      reset(seed = 0) {
        this.rng = mulberry32(seed >>> 0);
        this.world = pl.World({ gravity: Vec2(0.0, this.gravity) });
        this.gameOver = false;
        this.prevShaping = null;

        const height = new Array(CHUNKS + 1);
        for (let i = 0; i < CHUNKS + 1; i++) height[i] = this.rng() * (WORLD_H * 0.5);
        const chunkX = new Array(CHUNKS);
        for (let i = 0; i < CHUNKS; i++) chunkX[i] = (WORLD_W / (CHUNKS - 1)) * i;
        const mid = Math.floor(CHUNKS / 2);
        this.helipadX1 = chunkX[mid - 1];
        this.helipadX2 = chunkX[mid + 1];
        this.helipadY = WORLD_H / 4.0;
        for (let idx = mid - 2; idx <= mid + 2; idx++) height[idx] = this.helipadY;

        const smoothY = new Array(CHUNKS);
        for (let i = 0; i < CHUNKS; i++) {
          const prev = i === 0 ? height[CHUNKS] : height[i - 1];
          smoothY[i] = 0.33 * (prev + height[i] + height[i + 1]);
        }
        this.chunkX = chunkX;
        this.smoothY = smoothY;

        this.moon = this.world.createBody({ type: "static" });
        this.moon.createFixture(pl.Edge(Vec2(0.0, 0.0), Vec2(WORLD_W, 0.0)), { density: 0.0, friction: 0.1 });
        for (let i = 0; i < CHUNKS - 1; i++) {
          this.moon.createFixture(
            pl.Edge(Vec2(chunkX[i], smoothY[i]), Vec2(chunkX[i + 1], smoothY[i + 1])),
            { density: 0.0, friction: 0.1 }
          );
        }

        const initialX = HALF_W;
        const initialY = WORLD_H;
        this.lander = this.world.createDynamicBody({ position: Vec2(initialX, initialY), angle: 0.0 });
        this.lander.createFixture(
          pl.Polygon(LANDER_POLY.map(([px, py]) => Vec2(px / SCALE, py / SCALE))),
          {
            density: 5.0,
            friction: 0.1,
            filterCategoryBits: 0x0010,
            filterMaskBits: 0x001,
            restitution: 0.0,
          }
        );
        this.lander.applyForceToCenter(
          Vec2((this.rng() * 2.0 - 1.0) * INITIAL_RANDOM, (this.rng() * 2.0 - 1.0) * INITIAL_RANDOM),
          true
        );

        if (this.enableWind) {
          this.windIdx = Math.floor(this.rng() * 19998.0) - 9999;
          this.torqueIdx = Math.floor(this.rng() * 19998.0) - 9999;
        }

        this.legs = [];
        for (const sideSign of [-1, +1]) {
          const leg = this.world.createDynamicBody({
            position: Vec2(initialX - (sideSign * LEG_AWAY) / SCALE, initialY),
            angle: sideSign * 0.05,
          });
          leg.createFixture(pl.Box(LEG_W / SCALE, LEG_H / SCALE), {
            density: 1.0,
            restitution: 0.0,
            filterCategoryBits: 0x0020,
            filterMaskBits: 0x001,
          });
          leg.ground_contact = false;
          this.world.createJoint(
            pl.RevoluteJoint(
              {
                localAnchorA: Vec2(0.0, 0.0),
                localAnchorB: Vec2((sideSign * LEG_AWAY) / SCALE, LEG_DOWN / SCALE),
                enableMotor: true,
                enableLimit: true,
                maxMotorTorque: LEG_SPRING_TORQUE,
                motorSpeed: +0.3 * sideSign,
                lowerAngle: sideSign === -1 ? +0.9 - 0.5 : -0.9,
                upperAngle: sideSign === -1 ? +0.9 : -0.9 + 0.5,
              },
              this.lander,
              leg
            )
          );
          this.legs.push(leg);
        }

        this.world.on("begin-contact", (contact) => {
          const bA = contact.getFixtureA().getBody();
          const bB = contact.getFixtureB().getBody();
          if (this.lander === bA || this.lander === bB) this.gameOver = true;
          for (let i = 0; i < 2; i++) {
            if (this.legs[i] === bA || this.legs[i] === bB) this.legs[i].ground_contact = true;
          }
        });
        this.world.on("end-contact", (contact) => {
          const bA = contact.getFixtureA().getBody();
          const bB = contact.getFixtureB().getBody();
          for (let i = 0; i < 2; i++) {
            if (this.legs[i] === bA || this.legs[i] === bB) this.legs[i].ground_contact = false;
          }
        });

        return this.step([0.0, 0.0]).obs;
      },

      applyNudge(nudge) {
        if (!this.lander) return;
        const pos = this.lander.getPosition();
        if (nudge === "left") {
          this.lander.applyLinearImpulse(Vec2(-14.0, 0.0), pos, true);
        } else if (nudge === "right") {
          this.lander.applyLinearImpulse(Vec2(+14.0, 0.0), pos, true);
        } else if (nudge === "up") {
          this.lander.applyLinearImpulse(Vec2(0.0, +16.0), pos, true);
        }
      },

      step(action) {
        if (this.enableWind && !(this.legs[0].ground_contact || this.legs[1].ground_contact)) {
          const windMag =
            Math.tanh(Math.sin(0.02 * this.windIdx) + Math.sin(Math.PI * 0.01 * this.windIdx)) *
            this.windPower;
          this.windIdx += 1;
          this.lander.applyForceToCenter(Vec2(windMag, 0.0), true);

          const torqueMag =
            Math.tanh(Math.sin(0.02 * this.torqueIdx) + Math.sin(Math.PI * 0.01 * this.torqueIdx)) *
            this.turbulencePower;
          this.torqueIdx += 1;
          this.lander.applyTorque(torqueMag, true);
        }

        const a0 = Math.max(-1.0, Math.min(1.0, Number(action[0]) || 0.0));
        const a1 = Math.max(-1.0, Math.min(1.0, Number(action[1]) || 0.0));

        const ang = this.lander.getAngle();
        const tip0 = Math.sin(ang);
        const tip1 = Math.cos(ang);
        const side0 = -tip1;
        const side1 = tip0;

        const disp0 = (this.rng() * 2.0 - 1.0) / SCALE;
        const disp1 = (this.rng() * 2.0 - 1.0) / SCALE;
        const p0 = this.lander.getPosition();

        let mPower = 0.0;
        if (a0 > 0.0) {
          mPower = (Math.min(1.0, Math.max(0.0, a0)) + 1.0) * 0.5;
          const ox = tip0 * (MAIN_ENGINE_Y_LOCATION / SCALE + 2.0 * disp0) + side0 * disp1;
          const oy = -tip1 * (MAIN_ENGINE_Y_LOCATION / SCALE + 2.0 * disp0) - side1 * disp1;
          const ip = Vec2(p0.x + ox, p0.y + oy);
          this.lander.applyLinearImpulse(
            Vec2(-ox * MAIN_ENGINE_POWER * mPower, -oy * MAIN_ENGINE_POWER * mPower),
            ip,
            true
          );
        }

        let sPower = 0.0;
        let sSigned = 0.0;
        if (Math.abs(a1) > 0.5) {
          const direction = a1 > 0.0 ? 1.0 : -1.0;
          sPower = Math.min(1.0, Math.max(0.5, Math.abs(a1)));
          sSigned = direction * sPower;
          const ox = tip0 * disp0 + side0 * (3.0 * disp1 + (direction * SIDE_ENGINE_AWAY) / SCALE);
          const oy = -tip1 * disp0 - side1 * (3.0 * disp1 + (direction * SIDE_ENGINE_AWAY) / SCALE);
          const ip = Vec2(
            p0.x + ox - (tip0 * 17.0) / SCALE,
            p0.y + oy + (tip1 * SIDE_ENGINE_HEIGHT) / SCALE
          );
          this.lander.applyLinearImpulse(
            Vec2(-ox * SIDE_ENGINE_POWER * sPower, -oy * SIDE_ENGINE_POWER * sPower),
            ip,
            true
          );
        }

        this.world.step(DT, SOLVER_VEL_ITERS, SOLVER_POS_ITERS);

        const pos = this.lander.getPosition();
        const vel = this.lander.getLinearVelocity();
        const obs = lunarlanderObsFromBody(
          pos.x,
          pos.y,
          vel.x,
          vel.y,
          this.lander.getAngle(),
          this.lander.getAngularVelocity(),
          this.legs[0].ground_contact,
          this.legs[1].ground_contact,
          this.helipadY
        );

        const shaping =
          -100.0 * Math.hypot(obs[0], obs[1]) -
          100.0 * Math.hypot(obs[2], obs[3]) -
          100.0 * Math.abs(obs[4]) +
          10.0 * obs[6] +
          10.0 * obs[7];
        let reward = this.prevShaping === null ? 0.0 : shaping - this.prevShaping;
        this.prevShaping = shaping;
        reward -= mPower * 0.30;
        reward -= sPower * 0.03;

        let terminated = false;
        if (this.gameOver || Math.abs(obs[0]) >= 1.0) {
          terminated = true;
          reward = -100.0;
        }
        if (!this.lander.isAwake()) {
          terminated = true;
          reward = +100.0;
        }
        return { obs, reward, terminated, mPower, sSigned };
      },

      captureFrame(mPower, sSigned, cumRet) {
        const pos = this.lander.getPosition();
        const vel = this.lander.getLinearVelocity();
        const l0 = this.legs[0];
        const l1 = this.legs[1];
        const p0 = l0.getPosition();
        const p1 = l1.getPosition();
        return [
          Math.round(pos.x * 10000) / 10000,
          Math.round(pos.y * 10000) / 10000,
          Math.round(vel.x * 10000) / 10000,
          Math.round(vel.y * 10000) / 10000,
          Math.round(this.lander.getAngle() * 10000) / 10000,
          Math.round(this.lander.getAngularVelocity() * 10000) / 10000,
          l0.ground_contact ? 1.0 : 0.0,
          l1.ground_contact ? 1.0 : 0.0,
          Math.round(mPower * 1000) / 1000,
          Math.round(sSigned * 1000) / 1000,
          Math.round(cumRet * 100) / 100,
          Math.round(p0.x * 10000) / 10000,
          Math.round(p0.y * 10000) / 10000,
          Math.round(l0.getAngle() * 10000) / 10000,
          Math.round(p1.x * 10000) / 10000,
          Math.round(p1.y * 10000) / 10000,
          Math.round(l1.getAngle() * 10000) / 10000,
        ];
      },
    };
    return sim;
  }

  function lunarlanderRolloutPlanck(policyFn, opts = {}) {
    const seed = Number(opts.seed ?? 0);
    const maxSteps = Math.max(1, Number(opts.maxSteps ?? MAX_STEPS));
    const nudge = opts.nudge || null;
    const captureFrames = opts.captureFrames !== false;
    const stride = Math.max(1, Number(opts.poseStride ?? 1));
    const sim = createLunarLanderV3Sim(opts.planck || global.planck, opts.envParams || null);
    let obs = sim.reset(seed);

    let totalRet = 0.0;
    const frames = captureFrames ? [sim.captureFrame(0.0, 0.0, 0.0)] : [];
    const trajSample = [];
    let steps = 0;
    let outcome = "flying";
    const nudgeStep = Math.min(22, Math.max(1, Math.floor(maxSteps / 6)));

    for (let stepIdx = 0; stepIdx < maxSteps; stepIdx++) {
      if (nudge && stepIdx === nudgeStep) sim.applyNudge(nudge);
      const act = policyFn(obs, stepIdx);
      const res = sim.step(act);
      obs = res.obs;
      totalRet += res.reward;
      steps += 1;

      if (stepIdx % 12 === 0 && trajSample.length < 32) {
        trajSample.push(obs.map((v) => Math.round(v * 10000) / 10000));
      }
      if (captureFrames && (res.terminated || steps === maxSteps || steps % stride === 0)) {
        frames.push(sim.captureFrame(res.mPower, res.sSigned, totalRet));
      }
      if (res.terminated) {
        if (!sim.lander.isAwake() && !sim.gameOver) {
          const px = sim.lander.getPosition().x;
          const onPad = px >= sim.helipadX1 && px <= sim.helipadX2;
          outcome = onPad ? "landed" : "settled";
        } else {
          outcome = "crashed";
        }
        break;
      }
    }
    if (outcome === "flying") outcome = "timeout";

    const pos = sim.lander.getPosition();
    const onPad = pos.x >= sim.helipadX1 && pos.x <= sim.helipadX2;
    const landedClean = (outcome === "landed" || outcome === "settled") && !sim.gameOver && !sim.lander.isAwake();

    return {
      task: "lunarlander",
      seed,
      steps,
      max_steps: maxSteps,
      return: Math.round(totalRet * 100) / 100,
      outcome,
      landed: landedClean,
      on_pad: onPad,
      crashed: outcome === "crashed",
      final_x: Math.round(pos.x * 10000) / 10000,
      final_y: Math.round(pos.y * 10000) / 10000,
      final_angle: Math.round(sim.lander.getAngle() * 10000) / 10000,
      helipad_x1: Math.round(sim.helipadX1 * 10000) / 10000,
      helipad_x2: Math.round(sim.helipadX2 * 10000) / 10000,
      helipad_y: Math.round(sim.helipadY * 10000) / 10000,
      terrain_x: sim.chunkX.map((v) => Math.round(v * 10000) / 10000),
      terrain_y: sim.smoothY.map((v) => Math.round(v * 10000) / 10000),
      heights: sim.smoothY.map((v) => Math.round(v * 10000) / 10000),
      pose_stride: stride,
      frames,
      traj_sample: trajSample,
      nudge,
    };
  }

  // --- Authentic Gymnasium LunarLander-v3 2D Canvas Renderer ---

  function drawSceneBackdrop(ctx, w, h, rollout, padTop, padBot) {
    const usableH = Math.max(1, h - padTop - padBot);
    const toPx = (wx, wy) => [
      (wx / WORLD_W) * w,
      padTop + (1.0 - wy / WORLD_H) * usableH,
    ];

    // Deep space black sky (authentic Gymnasium LunarLander-v3)
    ctx.fillStyle = "#04050a";
    ctx.fillRect(0, 0, w, h);

    // Subtle deterministic starfield
    const starRng = mulberry32(42);
    const starCount = w < 80 ? 10 : 32;
    for (let i = 0; i < starCount; i++) {
      const sx = starRng() * w;
      const sy = padTop + starRng() * usableH * 0.62;
      const r = starRng() > 0.8 ? 1.25 : 0.8;
      const alpha = 0.22 + starRng() * 0.5;
      ctx.fillStyle = `rgba(226, 232, 240, ${alpha.toFixed(2)})`;
      ctx.fillRect(sx, sy, r, r);
    }

    const txArr =
      rollout && Array.isArray(rollout.terrain_x) && rollout.terrain_x.length >= CHUNKS
        ? rollout.terrain_x
        : Array.from({ length: CHUNKS }, (_, i) => (WORLD_W / (CHUNKS - 1)) * i);
    const tyArr =
      rollout && Array.isArray(rollout.terrain_y) && rollout.terrain_y.length >= CHUNKS
        ? rollout.terrain_y
        : rollout && Array.isArray(rollout.heights) && rollout.heights.length >= CHUNKS
        ? rollout.heights
        : Array.from({ length: CHUNKS }, () => HELIPAD_Y);

    const helipadX1 = rollout && Number.isFinite(Number(rollout.helipad_x1)) ? Number(rollout.helipad_x1) : HELIPAD_X1;
    const helipadX2 = rollout && Number.isFinite(Number(rollout.helipad_x2)) ? Number(rollout.helipad_x2) : HELIPAD_X2;
    const helipadY = rollout && Number.isFinite(Number(rollout.helipad_y)) ? Number(rollout.helipad_y) : HELIPAD_Y;

    // Lunar surface polygon (crisp pale lunar regolith matching Gymnasium LunarLander-v3)
    ctx.beginPath();
    for (let i = 0; i < txArr.length; i++) {
      const [px, py] = toPx(txArr[i], tyArr[i]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    const moonGrad = ctx.createLinearGradient(0, padTop + usableH * 0.55, 0, h);
    moonGrad.addColorStop(0, "#e5e7ef");
    moonGrad.addColorStop(1, "#c7ccd9");
    ctx.fillStyle = moonGrad;
    ctx.fill();

    // Terrain ridge outline
    ctx.beginPath();
    for (let i = 0; i < txArr.length; i++) {
      const [px, py] = toPx(txArr[i], tyArr[i]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = "#8a92a8";
    ctx.lineWidth = Math.max(1, w / 260);
    ctx.stroke();

    // Helipad landing strip accent
    const [p1x, p1y] = toPx(helipadX1, helipadY);
    const [p2x, p2y] = toPx(helipadX2, helipadY);
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = Math.max(2, w / 160);
    ctx.beginPath();
    ctx.moveTo(p1x, p1y);
    ctx.lineTo(p2x, p2y);
    ctx.stroke();

    // Iconic Gymnasium yellow flags on white poles at helipad_x1 & helipad_x2
    if (w >= 72) {
      const flagHWorld = 50.0 / SCALE;
      const flagW = Math.max(5, (25.0 / VIEWPORT_W) * w);
      const flagH = Math.max(3, (10.0 / VIEWPORT_H) * usableH);
      for (const fxW of [helipadX1, helipadX2]) {
        const [bx, by] = toPx(fxW, helipadY);
        const [tx, ty] = toPx(fxW, helipadY + flagHWorld);
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = Math.max(1, w / 340);
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.lineTo(tx, ty);
        ctx.stroke();
        ctx.fillStyle = "#cccc00";
        ctx.beginPath();
        ctx.moveTo(tx, ty);
        ctx.lineTo(tx, ty + flagH);
        ctx.lineTo(tx + flagW, ty + flagH * 0.5);
        ctx.closePath();
        ctx.fill();
      }
    }

    return { toPx, usableH, helipadY };
  }

  function drawLanderCraft(ctx, toPx, frame, isCrashedEnd, stepIdx, compact) {
    const x = frame[0];
    const y = frame[1];
    const angle = frame[4];
    const leg0C = frame[6] > 0.5;
    const leg1C = frame[7] > 0.5;
    const mPower = frame[8] || 0.0;
    const sSigned = frame[9] || 0.0;
    const hasLegPoses = frame.length >= 17;

    const rotPx = (cx, cy, ang, lx, ly) => {
      const ca = Math.cos(ang);
      const sa = Math.sin(ang);
      return toPx(cx + lx * ca - ly * sa, cy + lx * sa + ly * ca);
    };

    // Main engine exhaust cone + particles
    if (mPower > 0.01) {
      const flicker = 0.86 + 0.28 * Math.sin(stepIdx * 1.9);
      const pLen = (0.38 + 0.95 * mPower) * flicker;
      const y0 = -10.0 / SCALE;
      const [o1x, o1y] = rotPx(x, y, angle, -0.22, y0);
      const [o2x, o2y] = rotPx(x, y, angle, +0.22, y0);
      const [otx, oty] = rotPx(x, y, angle, 0.0, y0 - pLen);
      ctx.fillStyle = "rgba(255, 118, 42, 0.92)";
      ctx.beginPath();
      ctx.moveTo(o1x, o1y);
      ctx.lineTo(o2x, o2y);
      ctx.lineTo(otx, oty);
      ctx.closePath();
      ctx.fill();

      const [i1x, i1y] = rotPx(x, y, angle, -0.10, y0);
      const [i2x, i2y] = rotPx(x, y, angle, +0.10, y0);
      const [itx, ity] = rotPx(x, y, angle, 0.0, y0 - pLen * 0.58);
      ctx.fillStyle = "rgba(255, 240, 148, 0.96)";
      ctx.beginPath();
      ctx.moveTo(i1x, i1y);
      ctx.lineTo(i2x, i2y);
      ctx.lineTo(itx, ity);
      ctx.closePath();
      ctx.fill();
    }

    // Side RCS plume
    if (Math.abs(sSigned) > 0.01) {
      const sDir = sSigned > 0.0 ? 1.0 : -1.0;
      const sLen = 0.28 + 0.48 * Math.abs(sSigned);
      const sy = SIDE_ENGINE_HEIGHT / SCALE;
      const sx = -sDir * (SIDE_ENGINE_AWAY / SCALE);
      const [a1x, a1y] = rotPx(x, y, angle, sx, sy - 0.07);
      const [a2x, a2y] = rotPx(x, y, angle, sx, sy + 0.07);
      const [atx, aty] = rotPx(x, y, angle, sx - sDir * sLen, sy);
      ctx.fillStyle = "rgba(255, 168, 72, 0.92)";
      ctx.beginPath();
      ctx.moveTo(a1x, a1y);
      ctx.lineTo(a2x, a2y);
      ctx.lineTo(atx, aty);
      ctx.closePath();
      ctx.fill();
    }

    // Articulated Box2D landing legs (exact per-leg world position & angle)
    const lw = LEG_W / SCALE;
    const lh = LEG_H / SCALE;
    const legBox = [
      [-lw, -lh],
      [lw, -lh],
      [lw, lh],
      [-lw, lh],
    ];
    for (let li = 0; li < 2; li++) {
      const contact = li === 0 ? leg0C : leg1C;
      let lxW;
      let lyW;
      let lAng;
      if (hasLegPoses) {
        lxW = frame[11 + li * 3];
        lyW = frame[12 + li * 3];
        lAng = frame[13 + li * 3];
      } else {
        const sgn = li === 0 ? -1.0 : 1.0;
        const ca = Math.cos(angle);
        const sa = Math.sin(angle);
        const ox = sgn * (LEG_AWAY / SCALE) * 0.72;
        const oy = -(LEG_DOWN / SCALE) * 0.75;
        lxW = x + ox * ca - oy * sa;
        lyW = y + ox * sa + oy * ca;
        lAng = angle - sgn * 0.45;
      }
      ctx.beginPath();
      for (let bi = 0; bi < legBox.length; bi++) {
        const [px, py] = rotPx(lxW, lyW, lAng, legBox[bi][0], legBox[bi][1]);
        if (bi === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fillStyle = contact ? "#34d399" : "#8066e6";
      ctx.fill();
      ctx.strokeStyle = contact ? "#059669" : "#4d4d80";
      ctx.lineWidth = compact ? 1.0 : 1.4;
      ctx.stroke();
    }

    // Iconic Gymnasium purple-indigo polygonal hull (LANDER_POLY)
    ctx.beginPath();
    for (let hi = 0; hi < LANDER_POLY.length; hi++) {
      const [px, py] = rotPx(x, y, angle, LANDER_POLY[hi][0] / SCALE, LANDER_POLY[hi][1] / SCALE);
      if (hi === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fillStyle = isCrashedEnd ? "#f87171" : "#8066e6";
    ctx.fill();
    ctx.strokeStyle = isCrashedEnd ? "#991b1b" : "#4d4d80";
    ctx.lineWidth = compact ? 1.1 : 1.6;
    ctx.stroke();

    // Crash flash ring
    if (isCrashedEnd) {
      const [cx, cy] = toPx(x, y);
      ctx.strokeStyle = "rgba(248, 113, 113, 0.85)";
      ctx.lineWidth = compact ? 1.4 : 2.2;
      ctx.beginPath();
      ctx.arc(cx, cy, compact ? 9 : 16, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  function renderLunarStaticCanvas(canvas, rollout, summary, maxSteps) {
    if (!canvas || !rollout || !Array.isArray(rollout.frames) || !rollout.frames.length) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const compact = w < 180;
    const padTop = compact ? 2 : 6;
    const padBot = compact ? 2 : 6;

    const { toPx } = drawSceneBackdrop(ctx, w, h, rollout, padTop, padBot);
    const frames = rollout.frames;
    const total = frames.length;

    // Color-coded descent trajectory
    if (total > 1) {
      const stride = Math.max(1, Math.floor(total / 80));
      for (let i = stride; i < total; i += stride) {
        const prev = frames[Math.max(0, i - stride)];
        const cur = frames[i];
        const [x0, y0] = toPx(prev[0], prev[1]);
        const [x1, y1] = toPx(cur[0], cur[1]);
        const burning = cur[8] > 0.01;
        ctx.strokeStyle = burning ? "rgba(251, 146, 60, 0.82)" : "rgba(96, 165, 250, 0.58)";
        ctx.lineWidth = compact ? 1.4 : 2.0;
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
      }
    }

    // Subtle mid-descent ghost
    if (total >= 12) {
      ctx.save();
      ctx.globalAlpha = 0.34;
      const midIdx = Math.floor(total * 0.48);
      drawLanderCraft(ctx, toPx, frames[midIdx], false, midIdx, compact);
      ctx.restore();
    }

    // Final craft pose
    const lastIdx = total - 1;
    const isCrashed = rollout.outcome === "crashed";
    drawLanderCraft(ctx, toPx, frames[lastIdx], isCrashed, lastIdx, compact);

    // Compact status tag
    if (w >= 110) {
      const ret = summary && typeof summary.return === "number" ? summary.return : rollout.return || 0;
      const tag =
        rollout.outcome === "landed"
          ? "LANDED"
          : rollout.outcome === "settled"
          ? "SETTLED"
          : rollout.outcome === "crashed"
          ? "CRASH"
          : "TIMEOUT";
      const tagColor =
        rollout.outcome === "landed"
          ? "#34d399"
          : rollout.outcome === "settled"
          ? "#fbbf24"
          : rollout.outcome === "crashed"
          ? "#f87171"
          : "#94a3b8";
      ctx.font = `600 ${compact ? 9 : 11}px ui-monospace, SFMono-Regular, Menlo, monospace`;
      ctx.fillStyle = "rgba(4, 5, 10, 0.72)";
      ctx.fillRect(6, 6, compact ? 52 : 64, compact ? 14 : 18);
      ctx.fillStyle = tagColor;
      ctx.fillText(tag, 10, compact ? 16 : 19);

      if (!compact) {
        const retTxt = `${ret >= 0 ? "+" : ""}${ret.toFixed(0)}`;
        ctx.fillStyle = "rgba(4, 5, 10, 0.72)";
        ctx.fillRect(w - 58, 6, 52, 18);
        ctx.fillStyle = ret >= SOLVED_RETURN ? "#34d399" : ret >= TIER_MID_RETURN ? "#fbbf24" : "#cbd5e1";
        ctx.fillText(retTxt, w - 52, 19);
      }
    }
  }

  function renderLunarLiveFrame(canvas, rollout, frameIdx, maxSteps) {
    if (!canvas || !rollout || !Array.isArray(rollout.frames) || !rollout.frames.length) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const compact = w < 220;
    const padTop = compact ? 2 : 8;
    const padBot = compact ? 2 : 8;

    const { toPx, helipadY } = drawSceneBackdrop(ctx, w, h, rollout, padTop, padBot);
    const frames = rollout.frames;
    const idx = Math.max(0, Math.min(frames.length - 1, frameIdx | 0));

    if (idx > 0) {
      const stride = Math.max(1, Math.floor(idx / 90));
      for (let i = stride; i <= idx; i += stride) {
        const prev = frames[Math.max(0, i - stride)];
        const cur = frames[i];
        const [x0, y0] = toPx(prev[0], prev[1]);
        const [x1, y1] = toPx(cur[0], cur[1]);
        ctx.strokeStyle = cur[8] > 0.01 ? "rgba(251, 146, 60, 0.78)" : "rgba(96, 165, 250, 0.50)";
        ctx.lineWidth = compact ? 1.2 : 1.8;
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
      }
    }

    const isCrashedEnd = rollout.outcome === "crashed" && idx >= frames.length - 1;
    drawLanderCraft(ctx, toPx, frames[idx], isCrashedEnd, idx, compact);

    if (!compact) {
      const fr = frames[idx];
      const alt = Math.max(0, fr[1] - (helipadY + LEG_DOWN / SCALE));
      const ret = fr[10] || 0;
      const stepNum = Math.min(rollout.steps || idx, idx * (rollout.pose_stride || 1));
      ctx.font = "600 12px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.fillStyle = "rgba(4, 5, 10, 0.76)";
      ctx.fillRect(10, 10, 326, 22);
      ctx.fillStyle = "#e2e8f0";
      ctx.fillText(
        `v3 step ${stepNum}/${maxSteps || MAX_STEPS}  ret ${ret >= 0 ? "+" : ""}${ret.toFixed(1)}  alt ${alt.toFixed(1)}m`,
        16,
        25
      );
    }
  }

  function renderLunarLineageThumb(canvas, rollout, summary) {
    renderLunarStaticCanvas(canvas, rollout, summary, MAX_STEPS);
  }

  const api = {
    FPS,
    DT,
    SCALE,
    VIEWPORT_W,
    VIEWPORT_H,
    WORLD_W,
    HALF_W,
    WORLD_H,
    GRAVITY,
    MAIN_ENGINE_POWER,
    SIDE_ENGINE_POWER,
    INITIAL_RANDOM,
    LANDER_POLY,
    LEG_AWAY,
    LEG_DOWN,
    LEG_W,
    LEG_H,
    LEG_SPRING_TORQUE,
    SIDE_ENGINE_HEIGHT,
    SIDE_ENGINE_AWAY,
    MAIN_ENGINE_Y_LOCATION,
    CHUNKS,
    HELIPAD_Y,
    HELIPAD_X1,
    HELIPAD_X2,
    REST_COM_Y,
    MAX_STEPS,
    SOLVED_RETURN,
    TIER_MID_RETURN,
    mulberry32,
    lunarlanderObsFromBody,
    lunarlanderObs: lunarlanderObsFromBody,
    lunarlanderHeuristic,
    resolveLunarEnvParams,
    createLunarLanderV3Sim,
    lunarlanderRolloutPlanck,
    renderLunarStaticCanvas,
    renderLunarLiveFrame,
    renderLunarLineageThumb,
  };

  global.DetLunarRenderer = api;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})(typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : globalThis);
