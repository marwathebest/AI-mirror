import {
  FilesetResolver,
  GestureRecognizer,
  ImageSegmenter,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { createDecartClient, models } from "https://cdn.jsdelivr.net/npm/@decartai/sdk@0.2.14/+esm";

const WASM_ROOT = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const SEGMENTER_MODEL =
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite";
const GESTURE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";
const AI_MODEL = models.realtime("lucy-2.6");

const PRESENT_COVERAGE = 0.03;
const CLOSE_AFTER_MS = 1500;
// Billing runs per second while connected, so one visit is capped.
const MAX_VISIT_MS = 3 * 60 * 1000;
const AI_FADE_MS = 1200;
const GESTURE_HOLD_MS = 700;
const GESTURE_COOLDOWN_MS = 1800;

const $ = (selector) => document.querySelector(selector);
const stage = $("#stage");
const ctx = stage.getContext("2d");
const camera = $("#camera");
const aiVideo = $("#ai");
const statusPill = $("#status");

const state = {
  W: 0,
  H: 0,
  unit: 1,
  world: 0,
  frame: 0,
  time: 0,
  stream: null,
  coverage: 0,
  lastSeen: -Infinity,
  visiting: false,
  visitStart: 0,
  expired: false,
  open: 0,
  held: { name: null, since: 0 },
  cooldownUntil: 0,
};

/* ---------- geometry ---------- */

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = window.innerWidth * dpr;
  const height = window.innerHeight * dpr;
  const scale = Math.min(1, 1280 / Math.max(width, height));
  state.W = stage.width = Math.max(2, Math.round(width * scale));
  state.H = stage.height = Math.max(2, Math.round(height * scale));
  state.unit = Math.min(state.W, state.H) / 720;
}

function drawCover(c, image, width, height) {
  const { W, H } = state;
  const scale = Math.max(W / width, H / height);
  c.drawImage(image, (W - width * scale) / 2, (H - height * scale) / 2, width * scale, height * scale);
}

/* ---------- drawing helpers ---------- */

function radial(c, x, y, radius, color, alpha) {
  const gradient = c.createRadialGradient(x, y, 0, x, y, radius);
  gradient.addColorStop(0, color.replace("ALPHA", alpha));
  gradient.addColorStop(1, color.replace("ALPHA", 0));
  c.fillStyle = gradient;
  c.fillRect(x - radius, y - radius, radius * 2, radius * 2);
}

function vertical(c, stops) {
  const gradient = c.createLinearGradient(0, 0, 0, state.H);
  stops.forEach((color, i) => gradient.addColorStop(i / (stops.length - 1), color));
  c.fillStyle = gradient;
  c.fillRect(0, 0, state.W, state.H);
}

const scatter = (count, make) => Array.from({ length: count }, (_, i) => make(i));
const wrap = (value) => ((value % 1) + 1) % 1;

/* ---------- the other worlds: drawn in code while the AI warms up, described to the AI ---------- */

const stars = scatter(260, () => ({ x: Math.random(), y: Math.random(), r: 0.4 + Math.random() * 1.6, p: Math.random() * 6 }));
const bubbles = scatter(60, () => ({ x: Math.random(), y: Math.random(), r: 3 + Math.random() * 10, s: 0.03 + Math.random() * 0.06, p: Math.random() * 6 }));
const embers = scatter(120, () => ({ x: Math.random(), y: Math.random(), r: 1 + Math.random() * 3, s: 0.04 + Math.random() * 0.1, p: Math.random() * 6 }));
const flakes = scatter(140, () => ({ x: Math.random(), y: Math.random(), r: 1 + Math.random() * 2.5, s: 0.02 + Math.random() * 0.05, p: Math.random() * 6 }));
const fireflies = scatter(70, () => ({ x: Math.random(), y: 0.3 + Math.random() * 0.7, p: Math.random() * 6, s: 0.3 + Math.random() * 0.6 }));
const peaks = scatter(14, (i) => ({ x: i / 13, h: 0.18 + Math.random() * 0.22 }));
const trees = scatter(16, () => ({ x: Math.random(), w: 0.02 + Math.random() * 0.03, depth: Math.random() }))
  .sort((a, b) => a.depth - b.depth);

function drawStars(c, t, alpha = 1) {
  const { W, H, unit } = state;
  c.fillStyle = "#ffffff";
  for (const s of stars) {
    c.globalAlpha = alpha * (0.35 + 0.65 * (0.5 + 0.5 * Math.sin(t * 2 + s.p)));
    c.fillRect(s.x * W, s.y * H, s.r * unit, s.r * unit);
  }
  c.globalAlpha = 1;
}

function drawRidge(c, base, height, color, t = 0) {
  const { W, H } = state;
  c.fillStyle = color;
  c.beginPath();
  c.moveTo(0, H);
  peaks.forEach((peak, i) => {
    const jitter = Math.sin(i * 12.9 + t) * 0.02;
    c.lineTo(peak.x * W, H * (base - (peak.h + jitter) * height));
  });
  c.lineTo(W, H);
  c.closePath();
  c.fill();
}

// Each prompt rebuilds the whole person (face, hair, skin, clothes) and the scene behind them.
const TRANSFORM = "Completely transform the person into a fantasy character so their real face, hair and clothes are no longer recognizable, keeping only their pose and movement.";

const worlds = [
  {
    name: "الفضاء",
    icon: "🪐",
    glow: "#8b5cf6",
    ring: ["#a78bfa", "#f0abfc", "#67e8f9"],
    prompt: `${TRANSFORM} They become a futuristic cosmic explorer in a sleek white and violet space suit with glowing neon seams, a reflective glass helmet visor and glowing eyes. Replace the background with deep outer space full of purple and pink nebulae, stars and a giant ringed planet. Cinematic lighting.`,
    draw(c, t) {
      const { W, H } = state;
      vertical(c, ["#0b0420", "#140b3d", "#02010a"]);
      c.globalCompositeOperation = "lighter";
      ["rgba(123, 47, 247, ALPHA)", "rgba(241, 7, 163, ALPHA)", "rgba(0, 198, 255, ALPHA)", "rgba(42, 157, 143, ALPHA)"].forEach((color, i) => {
        radial(c, W * (0.5 + 0.35 * Math.sin(t * 0.05 + i * 2)), H * (0.4 + 0.3 * Math.cos(t * 0.04 + i)), Math.max(W, H) * 0.45, color, 0.28);
      });
      c.globalCompositeOperation = "source-over";
      drawStars(c, t);
      const px = W * 0.8;
      const py = H * 0.74;
      const pr = Math.min(W, H) * 0.2;
      c.save();
      c.translate(px, py);
      c.rotate(-0.35);
      c.strokeStyle = "rgba(255, 220, 180, 0.55)";
      c.lineWidth = pr * 0.09;
      c.beginPath();
      c.ellipse(0, 0, pr * 1.9, pr * 0.45, 0, Math.PI, Math.PI * 2);
      c.stroke();
      const planet = c.createRadialGradient(-pr * 0.4, -pr * 0.4, pr * 0.1, 0, 0, pr);
      planet.addColorStop(0, "#ffd29b");
      planet.addColorStop(0.6, "#c2410c");
      planet.addColorStop(1, "#2a0a00");
      c.fillStyle = planet;
      c.beginPath();
      c.arc(0, 0, pr, 0, Math.PI * 2);
      c.fill();
      c.beginPath();
      c.ellipse(0, 0, pr * 1.9, pr * 0.45, 0, 0, Math.PI);
      c.stroke();
      c.restore();
    },
  },
  {
    name: "تحت البحر",
    icon: "🐠",
    glow: "#22d3ee",
    ring: ["#67e8f9", "#a5f3fc", "#0ea5e9"],
    prompt: `${TRANSFORM} They become royalty of an underwater kingdom with shimmering teal scale armor, pearl necklaces, a crown of seashells, long flowing blue hair floating in the water and iridescent skin. Replace the background with a deep blue underwater world with sun rays piercing the water, rising bubbles and glowing coral reefs.`,
    draw(c, t) {
      const { W, H, unit } = state;
      vertical(c, ["#0e7490", "#083344", "#020617"]);
      c.save();
      c.globalCompositeOperation = "lighter";
      for (let i = 0; i < 7; i++) {
        const x = W * (i / 6) + Math.sin(t * 0.4 + i) * W * 0.05;
        const sway = Math.sin(t * 0.3 + i * 1.7) * 80 * unit;
        const ray = c.createLinearGradient(0, 0, 0, H);
        ray.addColorStop(0, "rgba(186, 230, 253, 0.22)");
        ray.addColorStop(1, "rgba(186, 230, 253, 0)");
        c.fillStyle = ray;
        c.beginPath();
        c.moveTo(x - 18 * unit, 0);
        c.lineTo(x + 18 * unit, 0);
        c.lineTo(x + 130 * unit + sway, H);
        c.lineTo(x - 50 * unit + sway, H);
        c.closePath();
        c.fill();
      }
      for (let i = 0; i < 3; i++) {
        radial(c, W * (0.2 + i * 0.3), H * 1.02, W * 0.25, i % 2 ? "rgba(244, 114, 182, ALPHA)" : "rgba(251, 146, 60, ALPHA)", 0.35);
      }
      c.restore();
      c.strokeStyle = "rgba(255, 255, 255, 0.55)";
      c.lineWidth = 1.5 * unit;
      for (const b of bubbles) {
        const y = wrap(b.y - t * b.s) * H;
        const x = b.x * W + Math.sin(t * 2 + b.p) * 8 * unit;
        c.beginPath();
        c.arc(x, y, b.r * unit, 0, Math.PI * 2);
        c.stroke();
      }
    },
  },
  {
    name: "عالم النار",
    icon: "🌋",
    glow: "#f97316",
    ring: ["#fb923c", "#fde047", "#ef4444"],
    prompt: `${TRANSFORM} They become a fire warrior in dark obsidian armor with glowing lava cracks, hair made of living flames and glowing ember eyes. Replace the background with a volcanic world with glowing lava rivers, a burning orange sky and floating embers.`,
    draw(c, t) {
      const { W, H, unit } = state;
      vertical(c, ["#1a0500", "#3b0a00", "#7c2d12"]);
      c.globalCompositeOperation = "lighter";
      radial(c, W * 0.5, H * 0.95, Math.max(W, H) * 0.7, "rgba(249, 115, 22, ALPHA)", 0.45);
      c.globalCompositeOperation = "source-over";
      drawRidge(c, 0.82, 1.1, "#120300");
      c.save();
      c.globalCompositeOperation = "lighter";
      c.lineCap = "round";
      for (let i = 0; i < 3; i++) {
        c.strokeStyle = ["#f97316", "#fde047", "#ef4444"][i];
        c.lineWidth = (10 - i * 3) * unit;
        c.shadowColor = c.strokeStyle;
        c.shadowBlur = 20 * unit;
        c.beginPath();
        for (let x = 0; x <= W; x += 16 * unit) {
          const y = H * 0.9 + Math.sin(x * 0.01 / unit + t * 1.5 + i) * 14 * unit + i * 10 * unit;
          if (x === 0) c.moveTo(x, y);
          else c.lineTo(x, y);
        }
        c.stroke();
      }
      c.shadowBlur = 0;
      for (const e of embers) {
        const y = wrap(e.y - t * e.s) * H;
        const x = e.x * W + Math.sin(t * 1.5 + e.p) * 20 * unit;
        c.globalAlpha = 0.4 + 0.6 * (y / H);
        c.fillStyle = e.p > 3 ? "#fdba74" : "#f97316";
        c.beginPath();
        c.arc(x, y, e.r * unit, 0, Math.PI * 2);
        c.fill();
      }
      c.restore();
    },
  },
  {
    name: "الشفق القطبي",
    icon: "🌌",
    glow: "#34d399",
    ring: ["#6ee7b7", "#a78bfa", "#e0f2fe"],
    prompt: `${TRANSFORM} They become ice royalty in a white fur-trimmed cloak with crystal ornaments, a crown of ice crystals, pale icy-blue skin and frosted white hair. Replace the background with a frozen arctic night with dancing green and purple northern lights, snowy mountains and falling snow.`,
    draw(c, t) {
      const { W, H, unit } = state;
      vertical(c, ["#020617", "#0f172a", "#1e293b"]);
      drawStars(c, t, 0.6);
      c.save();
      c.globalCompositeOperation = "lighter";
      c.lineCap = "round";
      [
        ["rgba(52, 211, 153, 0.12)", 0.22],
        ["rgba(167, 139, 250, 0.1)", 0.3],
        ["rgba(45, 212, 191, 0.1)", 0.38],
      ].forEach(([color, base], i) => {
        c.strokeStyle = color;
        for (const width of [140, 80, 30]) {
          c.lineWidth = width * unit;
          c.beginPath();
          for (let x = -40 * unit; x <= W + 40 * unit; x += 20 * unit) {
            const y = H * base + Math.sin(x * 0.004 / unit + t * 0.6 + i * 2) * 50 * unit + Math.sin(x * 0.011 / unit - t * 0.9) * 18 * unit;
            if (x < 0) c.moveTo(x, y);
            else c.lineTo(x, y);
          }
          c.stroke();
        }
      });
      c.restore();
      drawRidge(c, 0.95, 1.4, "#0b1120");
      drawRidge(c, 1.0, 0.9, "#cbd5e1", 1);
      c.fillStyle = "#ffffff";
      for (const f of flakes) {
        const y = wrap(f.y + t * f.s) * H;
        const x = wrap(f.x + Math.sin(t * 0.8 + f.p) * 0.01) * W;
        c.globalAlpha = 0.8;
        c.beginPath();
        c.arc(x, y, f.r * unit, 0, Math.PI * 2);
        c.fill();
      }
      c.globalAlpha = 1;
    },
  },
  {
    name: "الغابة السحرية",
    icon: "🌳",
    glow: "#a3e635",
    ring: ["#bef264", "#fde68a", "#4ade80"],
    prompt: `${TRANSFORM} They become an elven forest guardian wearing armor woven from leaves, moss and vines, a crown of flowers and branches, pointed ears and softly glowing green eyes. Replace the background with an enchanted night forest with giant ancient trees, soft mist and hundreds of glowing fireflies.`,
    draw(c, t) {
      const { W, H, unit } = state;
      vertical(c, ["#03140d", "#064e3b", "#010805"]);
      c.globalCompositeOperation = "lighter";
      radial(c, W * 0.5, H * 0.35, Math.max(W, H) * 0.5, "rgba(190, 242, 100, ALPHA)", 0.18);
      c.globalCompositeOperation = "source-over";
      for (const tree of trees) {
        const shade = Math.round(4 + tree.depth * 10);
        c.fillStyle = `rgb(${shade - 2}, ${shade + 12}, ${shade + 4})`;
        const x = tree.x * W;
        const width = tree.w * W * (0.6 + tree.depth);
        c.fillRect(x - width / 2, 0, width, H);
        c.beginPath();
        c.arc(x, H * 0.12, width * 3.2, 0, Math.PI * 2);
        c.fill();
      }
      const mist = c.createLinearGradient(0, H * 0.6, 0, H);
      mist.addColorStop(0, "rgba(209, 250, 229, 0)");
      mist.addColorStop(1, "rgba(209, 250, 229, 0.18)");
      c.fillStyle = mist;
      c.fillRect(0, H * 0.6, W, H * 0.4);
      c.save();
      c.globalCompositeOperation = "lighter";
      for (const f of fireflies) {
        const x = wrap(f.x + Math.sin(t * f.s + f.p) * 0.04) * W;
        const y = f.y * H + Math.cos(t * f.s * 1.3 + f.p) * 30 * unit;
        const pulse = 0.4 + 0.6 * (0.5 + 0.5 * Math.sin(t * 3 + f.p * 4));
        radial(c, x, y, 14 * unit, "rgba(217, 249, 157, ALPHA)", pulse * 0.9);
      }
      c.restore();
    },
  },
];

/* ---------- live AI: the camera goes to Decart, only the transformed video comes back ---------- */

const decart = createDecartClient({
  apiKeyProvider: async () => {
    const response = await fetch("/api/realtime-token", { method: "POST" });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
    return body.apiKey;
  },
});

const ai = { id: 0, session: null, starting: false, connection: "disconnected", connectedAt: 0, liveAt: 0 };

function showStatus(text) {
  statusPill.textContent = text;
  statusPill.hidden = !text;
}

async function startAI() {
  if (ai.session || ai.starting) return;
  ai.starting = true;
  const id = ++ai.id;
  showStatus("✨ البوابة تنقلك…");
  try {
    const session = await decart.realtime.connect(state.stream, {
      model: AI_MODEL,
      mirror: true,
      onRemoteStream: (stream) => {
        aiVideo.srcObject = stream;
        aiVideo.play().catch(() => {});
      },
      initialState: { prompt: { text: worlds[state.world].prompt, enhance: true } },
    });
    if (id !== ai.id) {
      session.disconnect();
      return;
    }
    ai.session = session;
    ai.connectedAt = performance.now();
    session.on("connectionChange", (connection) => {
      ai.connection = connection;
    });
    session.on("error", (error) => console.error("Decart error", error.code, error.message));
    session.on("sessionEnded", ({ reason }) => {
      console.warn("Decart session ended:", reason);
      stopAI();
      if (reason === "insufficient_credits") showStatus("رصيد Decart خلص، اشحنيه من لوحة التحكم");
    });
  } catch (error) {
    console.error(error);
    if (id === ai.id) showStatus(error.message || "تعذّر الاتصال بالذكاء الاصطناعي");
  } finally {
    if (id === ai.id) ai.starting = false;
  }
}

function stopAI() {
  ai.id++;
  ai.starting = false;
  ai.session?.disconnect();
  ai.session = null;
  ai.connection = "disconnected";
  ai.liveAt = 0;
  aiVideo.srcObject = null;
}

// Only show the AI picture once it is actually generating, so the raw camera never leaks through.
function aiAlpha(now) {
  const ready =
    ai.session &&
    aiVideo.readyState >= 2 &&
    aiVideo.videoWidth > 0 &&
    (ai.connection === "generating" || now - ai.connectedAt > 1500);
  if (!ready) {
    ai.liveAt = 0;
    return 0;
  }
  if (!ai.liveAt) {
    ai.liveAt = now;
    showStatus("");
  }
  return Math.min(1, (now - ai.liveAt) / AI_FADE_MS);
}

/* ---------- the portal ---------- */

const ringParticles = scatter(280, () => ({
  angle: Math.random() * Math.PI * 2,
  speed: 0.6 + Math.random() * 1.4,
  offset: (Math.random() - 0.5) * 0.08,
  size: 1 + Math.random() * 3,
  color: Math.floor(Math.random() * 3),
}));
const sparks = [];

function portalShape(t) {
  const { W, H } = state;
  const base = Math.min(W * 0.46, H * 0.4);
  const radius = base * (0.04 + 0.96 * state.open);
  return { cx: W / 2, cy: H * 0.52, rx: radius, ry: radius * 1.18, t };
}

function portalPath(c, shape) {
  const { cx, cy, rx, ry, t } = shape;
  c.beginPath();
  for (let i = 0; i <= 120; i++) {
    const a = (i / 120) * Math.PI * 2;
    const wobble = 1 + 0.015 * Math.sin(a * 5 + t * 2) + 0.01 * Math.sin(a * 9 - t * 3);
    const x = cx + Math.cos(a) * rx * wobble;
    const y = cy + Math.sin(a) * ry * wobble;
    if (i === 0) c.moveTo(x, y);
    else c.lineTo(x, y);
  }
  c.closePath();
}

function drawOutside(world, shape, t) {
  const { W, H } = state;
  ctx.fillStyle = "#020205";
  ctx.fillRect(0, 0, W, H);
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  const light = ctx.createRadialGradient(shape.cx, shape.cy, shape.rx * 0.5, shape.cx, shape.cy, Math.max(shape.rx, 40) * 2.6);
  light.addColorStop(0, world.glow + "55");
  light.addColorStop(1, "transparent");
  ctx.fillStyle = light;
  ctx.fillRect(0, 0, W, H);
  // Slow smoke swirling around the portal.
  for (let i = 0; i < 5; i++) {
    const a = t * 0.15 + i * 1.3;
    radial(ctx, shape.cx + Math.cos(a) * W * 0.35, shape.cy + Math.sin(a * 0.8) * H * 0.3, Math.max(W, H) * 0.3, "rgba(148, 163, 184, ALPHA)", 0.05);
  }
  ctx.restore();
}

function drawInside(world, shape, t, now) {
  ctx.save();
  portalPath(ctx, shape);
  ctx.clip();
  world.draw(ctx, t);
  const alpha = aiAlpha(now) * Math.min(1, Math.max(0, (state.open - 0.3) / 0.7));
  if (alpha > 0) {
    ctx.globalAlpha = alpha;
    drawCover(ctx, aiVideo, aiVideo.videoWidth, aiVideo.videoHeight);
    ctx.globalAlpha = 1;
  }
  // Darken the inner edge so the world feels deeper than the opening.
  const depth = ctx.createRadialGradient(shape.cx, shape.cy, Math.min(shape.rx, shape.ry) * 0.6, shape.cx, shape.cy, Math.max(shape.rx, shape.ry) * 1.05);
  depth.addColorStop(0, "rgba(0, 0, 0, 0)");
  depth.addColorStop(1, "rgba(0, 0, 0, 0.7)");
  ctx.fillStyle = depth;
  ctx.fillRect(0, 0, state.W, state.H);
  ctx.restore();
}

function emitSparks(shape, count) {
  for (let i = 0; i < count; i++) {
    const a = Math.random() * Math.PI * 2;
    const speed = (2 + Math.random() * 5) * state.unit;
    sparks.push({
      x: shape.cx + Math.cos(a) * shape.rx,
      y: shape.cy + Math.sin(a) * shape.ry,
      vx: Math.cos(a) * speed - Math.sin(a) * speed,
      vy: Math.sin(a) * speed + Math.cos(a) * speed,
      life: 1,
      color: Math.floor(Math.random() * 3),
    });
  }
  if (sparks.length > 500) sparks.splice(0, sparks.length - 500);
}

function drawRing(world, shape, t, dt) {
  const { unit } = state;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.lineWidth = 6 * unit;
  ctx.strokeStyle = world.ring[0];
  ctx.shadowColor = world.glow;
  ctx.shadowBlur = 40 * unit;
  portalPath(ctx, shape);
  ctx.stroke();
  ctx.lineWidth = 2 * unit;
  ctx.strokeStyle = "#ffffff";
  ctx.shadowBlur = 12 * unit;
  portalPath(ctx, shape);
  ctx.stroke();
  ctx.shadowBlur = 0;

  for (const p of ringParticles) {
    p.angle += p.speed * dt * (0.6 + state.open);
    const r = 1 + p.offset + Math.sin(t * 4 + p.angle * 3) * 0.01;
    ctx.fillStyle = world.ring[p.color];
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    ctx.arc(shape.cx + Math.cos(p.angle) * shape.rx * r, shape.cy + Math.sin(p.angle) * shape.ry * r, p.size * unit, 0, Math.PI * 2);
    ctx.fill();
  }

  emitSparks(shape, state.open > 0.1 ? 3 : 1);
  for (let i = sparks.length - 1; i >= 0; i--) {
    const s = sparks[i];
    s.x += s.vx * dt * 60;
    s.y += s.vy * dt * 60;
    s.vx *= 0.96;
    s.vy = s.vy * 0.96 + 0.15 * unit;
    s.life -= dt * 1.6;
    if (s.life <= 0) {
      sparks.splice(i, 1);
      continue;
    }
    ctx.globalAlpha = s.life;
    ctx.fillStyle = world.ring[s.color];
    ctx.beginPath();
    ctx.arc(s.x, s.y, 2.2 * unit * s.life, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

/* ---------- interface ---------- */

const toast = $("#toast");
const hint = $("#hint");
let toastTimer = 0;

function setWorld(index, announce = true) {
  state.world = (index + worlds.length) % worlds.length;
  const world = worlds[state.world];
  hint.textContent = `${world.icon} ${world.name}`;
  ai.session?.setPrompt(world.prompt, { enhance: true });
  if (announce) {
    toast.textContent = `${world.icon} ${world.name}`;
    toast.classList.remove("show");
    void toast.offsetWidth;
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("show"), 1600);
  }
}

function showOverlay(id, visible) {
  $(id).hidden = !visible;
}

/* ---------- gestures ---------- */

const gestureMeter = $("#gesture-meter");

function handleGestures(result, now) {
  const names = (result?.gestures || [])
    .map((list) => list[0])
    .filter((top) => top && top.score > 0.6)
    .map((top) => top.categoryName);
  const name = names.includes("Victory") ? "Victory" : null;

  if (name !== state.held.name) state.held = { name, since: now };
  const progress = name && now > state.cooldownUntil ? (now - state.held.since) / GESTURE_HOLD_MS : 0;
  gestureMeter.style.setProperty("--progress", Math.min(progress, 1));

  if (progress >= 1) {
    state.cooldownUntil = now + GESTURE_COOLDOWN_MS;
    state.held = { name: null, since: now };
    setWorld(state.world + 1);
  }
}

/* ---------- camera & presence ---------- */

async function openCamera() {
  state.stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: "user", width: AI_MODEL.width, height: AI_MODEL.height, frameRate: AI_MODEL.fps },
  });
  camera.srcObject = state.stream;
  await camera.play();
}

async function createTask(Task, vision, options) {
  try {
    return await Task.createFromOptions(vision, { ...options, baseOptions: { ...options.baseOptions, delegate: "GPU" } });
  } catch (error) {
    console.warn("GPU delegate failed, falling back to CPU", error);
    return Task.createFromOptions(vision, { ...options, baseOptions: { ...options.baseOptions, delegate: "CPU" } });
  }
}

let segmenter;
let recognizer;

// The segmenter only answers "is someone standing here?"; its mask is never drawn.
function readCoverage(result) {
  const masks = result.confidenceMasks;
  if (!masks?.length) return;
  const values = masks[masks.length - 1].getAsFloat32Array();
  let sum = 0;
  for (let i = 0; i < values.length; i++) sum += values[i];
  state.coverage = sum / values.length;
}

// Someone walking in opens the portal and starts the AI; leaving closes both.
function updatePortal(now, dt) {
  const here = state.coverage > PRESENT_COVERAGE;
  if (here) state.lastSeen = now;
  const present = now - state.lastSeen < CLOSE_AFTER_MS;

  if (present && !state.visiting && !state.expired) {
    state.visiting = true;
    state.visitStart = now;
    setWorld(state.world + 1);
    emitSparks(portalShape(state.time), 80);
    startAI();
  }
  if (state.visiting && now - state.visitStart > MAX_VISIT_MS) {
    state.visiting = false;
    state.expired = true;
    stopAI();
    showStatus("شكراً لزيارتك ✨ خلّ غيرك يجرّب");
  }
  if (!present && (state.visiting || state.expired)) {
    state.visiting = false;
    state.expired = false;
    stopAI();
    showStatus("");
  }

  const target = state.visiting ? 1 : 0;
  state.open += (target - state.open) * Math.min(1, dt * (target ? 2.4 : 1.6));
  document.body.classList.toggle("idle", !present && state.open < 0.2);
}

let lastTime = performance.now();
let lastTimestamp = 0;

function loop() {
  requestAnimationFrame(loop);
  if (camera.readyState < 2) return;
  const now = performance.now();
  const dt = Math.min((now - lastTime) / 1000, 0.1);
  lastTime = now;
  state.time += dt;
  state.frame++;

  const timestamp = Math.max(now, lastTimestamp + 1);
  lastTimestamp = timestamp;
  segmenter.segmentForVideo(camera, timestamp, readCoverage);
  if (state.frame % 3 === 0) handleGestures(recognizer.recognizeForVideo(camera, timestamp), now);

  updatePortal(now, dt);
  const world = worlds[state.world];
  const shape = portalShape(state.time);
  drawOutside(world, shape, state.time);
  drawInside(world, shape, state.time, now);
  drawRing(world, shape, state.time, dt);
}

async function start() {
  resize();
  setWorld(worlds.length - 1, false);
  window.addEventListener("resize", resize);
  window.addEventListener("beforeunload", stopAI);

  try {
    await openCamera();
  } catch (error) {
    console.error(error);
    const reasons = {
      NotAllowedError: "المتصفح ما سمح بالكاميرا. اضغطي على أيقونة القفل بجانب الرابط واسمحي بالكاميرا، ثم حدّثي الصفحة.",
      NotFoundError: "ما فيه كاميرا موصولة بالجهاز.",
      NotReadableError: "الكاميرا مستخدمة في برنامج ثاني (مثل Teams أو Zoom). سكّريه وحدّثي الصفحة.",
    };
    $("#camera-error-reason").textContent =
      reasons[error.name] || (window.isSecureContext ? `${error.name}: ${error.message}` : "افتحي الصفحة من http://localhost:3000 عشان الكاميرا تشتغل.");
    showOverlay("#loading", false);
    showOverlay("#camera-error", true);
    return;
  }

  try {
    const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
    [segmenter, recognizer] = await Promise.all([
      createTask(ImageSegmenter, vision, {
        baseOptions: { modelAssetPath: SEGMENTER_MODEL },
        runningMode: "VIDEO",
        outputCategoryMask: false,
        outputConfidenceMasks: true,
      }),
      createTask(GestureRecognizer, vision, {
        baseOptions: { modelAssetPath: GESTURE_MODEL },
        runningMode: "VIDEO",
        numHands: 2,
      }),
    ]);
  } catch (error) {
    console.error(error);
    $("#loading-text").textContent = "تعذّر تحميل الذكاء الاصطناعي. تأكد من الاتصال بالإنترنت ثم حدّث الصفحة.";
    return;
  }

  showOverlay("#loading", false);
  loop();
}

$("#fullscreen").addEventListener("click", () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen?.();
});
window.addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft") setWorld(state.world + 1);
  else if (event.key === "ArrowRight") setWorld(state.world - 1);
});

start();
