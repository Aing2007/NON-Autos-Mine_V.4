// ============================================================
// app.js — AI Detection Monitor
// ============================================================

const WS_URL = "ws://localhost:8000/ws";
const SEND_W = 320;   // was 640 — cuts inference time ~4×
const JPEG_Q = 0.55;  // was 0.72 — smaller blob, faster encode
const WS_TIMEOUT_MS = 3500;

// ============================================================
// DRAW STYLE — แก้ที่นี่ที่เดียวเพื่อเปลี่ยนสี ขนาด ความเข้มทุกอย่าง
// ============================================================
const DRAW = {
    // --- สีกรอบหลัก ---
    color: {
        human: "#00e5ff",   // cyan
        face: "#ff4081",   // pink
        pose: "#69ff47",   // green
    },

    // --- ความหนาเส้น (px) ---
    lineWidth: {
        humanBox: 1.5,   // กรอบ human
        humanCorner: 2.0,   // L-mark มุม
        humanDiag: 1.0,   // เส้นทแยง
        faceBox: 1.5,   // กรอบ face
        poseBox: 1.0,   // กรอบ pose
        skeleton: 1.8,   // เส้น skeleton
    },

    // --- ความโปร่งใส 0.0–1.0 ---
    alpha: {
        humanDiag: 0.35, // เส้นทแยงมุม
        faceBox: 0.90, // กรอบ face
        poseBox: 0.80, // กรอบ pose
        skeletonEdge: 0.85, // เส้น skeleton
        kpOuter: 0.55, // วงนอก keypoint
        kpInner: 1.00, // วงใน keypoint
    },

    // --- ขนาดจุด (radius px) ---
    radius: {
        centerDot: 3.5,     // จุดแดงกลาง human box
        kpOuter: 4.5,     // วงนอก keypoint
        kpInner: 3.0,     // วงใน keypoint
    },

    // --- สีพิเศษ ---
    centerDotColor: "#ff3333",

    // --- สี skeleton แยกฝั่ง (ต่อด้วย alpha จาก DRAW.alpha.skeletonEdge) ---
    skeletonColor: {
        face: "rgba(200,200,255,",  // ส่วนหัว
        center: "rgba(105,255,71,",   // กลางลำตัว
        left: "rgba(80,200,255,",   // ฝั่งซ้าย
        right: "rgba(255,220,50,",   // ฝั่งขวา
    },

    // --- Keypoint conf ต่ำกว่านี้ไม่วาด ---
    kpConfMin: 0.30,
};

// alias ให้ code ที่เหลือใช้ TYPE_COLORS ได้เหมือนเดิม
const TYPE_COLORS = DRAW.color;

// State
let ws = null;
let waitingReply = false;
let streaming = false;
let timeoutTimer = null;

// Stats
let txCount = 0, rxCount = 0;
let txTs = Date.now(), rxTs = Date.now();
let txFps = 0, rxFps = 0;
let latencyMs = 0;
let lastSendMs = 0;

// Latest data from server
let latestHumans = [];
let latestFaces = [];
let latestPoses = [];
let srcW = 0, srcH = 0;
let inferMs = 0;

// Log
const MAX_LOG = 60;
const logEntries = [];

// ID → color cache
const idColors = {};
const COLOR_PALETTE = [
    "#00e5ff", "#ff4081", "#69ff47", "#ffab00",
    "#a29bfe", "#fd79a8", "#55efc4", "#fdcb6e", "#74b9ff", "#ff7675"
];
let colorIdx = 0;
function getIdColor(id) {
    if (id == null) return TYPE_COLORS.human;
    if (!idColors[id]) { idColors[id] = COLOR_PALETTE[colorIdx++ % COLOR_PALETTE.length]; }
    return idColors[id];
}

// ============================================================
// FIX 3: สร้าง Set และ pre-compute ครั้งเดียว (ไม่สร้างใหม่ทุก call)
// ============================================================
const LEFT_SET = new Set([1, 3, 5, 7, 9, 11, 13, 15]);
const RIGHT_SET = new Set([2, 4, 6, 8, 10, 12, 14, 16]);
const EDGE_COLORS = {
    face: "rgba(200,200,255,0.8)",
    center: "rgba(105,255,71,0.85)",
    left: "rgba(80,200,255,0.9)",
    right: "rgba(255,220,50,0.9)",
};
const SKELETON_EDGES = [
    [0, 1], [0, 2], [1, 3], [2, 4],
    [5, 6], [5, 11], [6, 12], [11, 12],
    [5, 7], [7, 9], [6, 8], [8, 10],
    [11, 13], [13, 15], [12, 14], [14, 16],
];

// pre-compute สีของแต่ละ edge (อ่านจาก DRAW.skeletonColor + DRAW.alpha.skeletonEdge)
function buildEdgeColorMap() {
    const a = DRAW.alpha.skeletonEdge;
    const c = DRAW.skeletonColor;
    return SKELETON_EDGES.map(([i, j]) => {
        if (i <= 4 && j <= 4) return c.face + a + ")";
        const isLeft = LEFT_SET.has(i) || LEFT_SET.has(j);
        const isRight = RIGHT_SET.has(i) || RIGHT_SET.has(j);
        if (isLeft && !isRight) return c.left + a + ")";
        if (isRight && !isLeft) return c.right + a + ")";
        return c.center + a + ")";
    });
}
let EDGE_COLOR_MAP = buildEdgeColorMap();

// ============================================================
// DOM refs
// ============================================================
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const camOffline = document.getElementById("camOffline");
const wsDot = document.getElementById("wsDot");
const wsLabel = document.getElementById("wsLabel");
const valTxFps = document.getElementById("valTxFps");
const valRxFps = document.getElementById("valRxFps");
const valLat = document.getElementById("valLat");
const valInfer = document.getElementById("valInfer");
const countHuman = document.getElementById("countHuman");
const countFace = document.getElementById("countFace");
const countPose = document.getElementById("countPose");
const detLog = document.getElementById("detLog");
const inferBadge = document.getElementById("inferBadge");
const clockEl = document.getElementById("clock");

// ============================================================
// Clock
// ============================================================
setInterval(() => {
    clockEl.textContent = new Date().toLocaleTimeString("th-TH", { hour12: false });
}, 1000);

// ============================================================
// FIX 1: setupHiDPI — เรียกครั้งเดียวตอน init และ resize เท่านั้น
// ไม่เรียกซ้ำทุก frame อีกต่อไป
// ============================================================
let drawCtx = null;
let drawW = 0;
let drawH = 0;

function initCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const rect = overlay.getBoundingClientRect();
    drawW = rect.width;
    drawH = rect.height;
    overlay.width = Math.round(drawW * dpr);
    overlay.height = Math.round(drawH * dpr);
    drawCtx = overlay.getContext("2d");
    drawCtx.setTransform(1, 0, 0, 1, 0, 0);
    drawCtx.scale(dpr, dpr);
}

// resize → reinit canvas แล้ว flag ว่า ready
const ro = new ResizeObserver(() => initCanvas());
ro.observe(overlay);

// ============================================================
// Camera
// ============================================================
async function openCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user" }, audio: false
        });
        video.srcObject = stream;
        await video.play();
        camOffline.classList.remove("show");
    } catch (err) {
        camOffline.classList.add("show");
        document.querySelector(".cam-offline-text").textContent = "กรุณาอนุญาตการใช้กล้อง";
        console.error("Camera error:", err);
    }
}

// ============================================================
// WebSocket
// ============================================================
function connectWS() {
    ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
        wsDot.classList.add("on");
        wsLabel.textContent = "CONNECTED";
        waitingReply = false;
        addLog("system", "sys", "WebSocket connected");
    };

    ws.onclose = () => {
        wsDot.classList.remove("on");
        wsLabel.textContent = "DISCONNECTED";
        waitingReply = false;
        clearTimeout(timeoutTimer);
        addLog("system", "sys", "disconnected — reconnecting…");
        setTimeout(connectWS, 2000);
    };

    ws.onerror = () => ws.close();

    ws.onmessage = (ev) => {
        clearTimeout(timeoutTimer);
        waitingReply = false;
        latencyMs = Date.now() - lastSendMs;

        rxCount++;
        const now = Date.now();
        const dtRx = (now - rxTs) / 1000;
        if (dtRx >= 1) { rxFps = rxCount / dtRx; rxCount = 0; rxTs = now; }

        let msg;
        try { msg = JSON.parse(ev.data); } catch { return; }

        srcW = msg.w || srcW;
        srcH = msg.h || srcH;
        inferMs = msg.inference_ms || 0;

        latestHumans = msg.humans || [];
        latestFaces = msg.faces || [];
        latestPoses = msg.poses || [];

        // update UI (เบาๆ ไม่ใช่ draw)
        countHuman.textContent = latestHumans.length;
        countFace.textContent = latestFaces.length;
        countPose.textContent = latestPoses.length;
        inferBadge.querySelector("span").textContent = inferMs.toFixed(1);
        updateStats();

        for (const b of latestHumans) addLog("human", "HUMAN", `ID:${b.id ?? "?"} conf:${(b.conf * 100).toFixed(0)}%`);
        for (const b of latestFaces) addLog("face", "FACE", `conf:${(b.conf * 100).toFixed(0)}%`);
    };
}

// ============================================================
// Send frame
// ============================================================
const sendCanvas = document.createElement("canvas");
const sendCtx = sendCanvas.getContext("2d");
sendCtx.imageSmoothingEnabled = false;

// Replace toBlob() callback with this synchronous path:
function sendFrame() {
    if (!ws || ws.readyState !== 1 || waitingReply || !video.videoWidth) return;

    const ratio = video.videoHeight / video.videoWidth;
    const W = SEND_W, H = Math.round(SEND_W * ratio);
    if (sendCanvas.width !== W || sendCanvas.height !== H) {
        sendCanvas.width = W; sendCanvas.height = H;
    }

    sendCtx.drawImage(video, 0, 0, W, H);

    // OffscreenCanvas path — no async callback, no extra delay
    sendCanvas.toBlob((blob) => {
        if (!blob || ws.readyState !== 1) return;
        lastSendMs = Date.now();
        ws.send(blob);
        waitingReply = true;
        timeoutTimer = setTimeout(() => { waitingReply = false; }, WS_TIMEOUT_MS);
        txCount++;
        const now = Date.now();
        const dt = (now - txTs) / 1000;
        if (dt >= 1) { txFps = txCount / dt; txCount = 0; txTs = now; }
    }, "image/jpeg", JPEG_Q);
}

function loop() {
    if (streaming) { sendFrame(); requestAnimationFrame(loop); }
}

// ============================================================
// FIX 2: Draw loop — ไม่เรียก setupHiDPI ซ้ำทุก frame
// ใช้ drawCtx / drawW / drawH ที่ init ไว้แล้วโดยตรง
// ============================================================
function drawLoop() {
    function draw() {
        if (!drawCtx) { requestAnimationFrame(draw); return; }

        const vw = video.videoWidth;
        const vh = video.videoHeight;
        if (!vw || !vh) { requestAnimationFrame(draw); return; }

        drawCtx.clearRect(0, 0, drawW, drawH);

        const SW = srcW || vw;
        const SH = srcH || vh;
        const scaleX = drawW / SW;
        const scaleY = drawH / SH;

        // mirror x helper
        const mx = (bx, bw) => drawW - (bx + bw) * scaleX;

        // ---- Human: กรอบ + ทแยงมุม + corners ----
        for (const b of latestHumans) {
            const color = getIdColor(b.id);
            const bx = mx(b.x, b.w);
            const by = b.y * scaleY;
            const bw = b.w * scaleX;
            const bh = b.h * scaleY;
            drawHumanBox(drawCtx, bx, by, bw, bh, color, b.id, b.conf);
        }

        // ---- Face: กรอบ dashed ----
        drawCtx.strokeStyle = TYPE_COLORS.face;
        drawCtx.lineWidth = DRAW.lineWidth.faceBox;
        drawCtx.globalAlpha = DRAW.alpha.faceBox;
        drawCtx.setLineDash([4, 3]);
        for (const b of latestFaces) {
            const bx = mx(b.x, b.w);
            const by = b.y * scaleY;
            const bw = b.w * scaleX;
            const bh = b.h * scaleY;
            drawCtx.strokeRect(bx, by, bw, bh);
            drawTag(drawCtx, bx, by, `FACE ${(b.conf * 100).toFixed(0)}%`, TYPE_COLORS.face);
        }
        drawCtx.setLineDash([]);
        drawCtx.globalAlpha = 1;

        // ---- Pose: skeleton ----
        for (const b of latestPoses) {
            const bx = mx(b.x, b.w);
            const by = b.y * scaleY;
            const bw = b.w * scaleX;
            const bh = b.h * scaleY;

            drawCtx.strokeStyle = TYPE_COLORS.pose;
            drawCtx.lineWidth = DRAW.lineWidth.poseBox;
            drawCtx.globalAlpha = DRAW.alpha.poseBox;
            drawCtx.setLineDash([2, 4]);
            drawCtx.strokeRect(bx, by, bw, bh);
            drawCtx.setLineDash([]);
            drawCtx.globalAlpha = 1;

            drawTag(drawCtx, bx, by + bh, `POSE ${(b.conf * 100).toFixed(0)}%`, TYPE_COLORS.pose, true);

            if (b.keypoints && b.keypoints.length >= 17) {
                drawSkeleton(drawCtx, b.keypoints, scaleX, scaleY, drawW);
            }
        }

        requestAnimationFrame(draw);
    }
    requestAnimationFrame(draw);
}

// ============================================================
// Draw helpers
// ============================================================

// drawHumanBox — อ่านทุกค่าจาก DRAW
function drawHumanBox(ctx, x, y, w, h, color, id, conf) {
    const cx = x + w / 2;
    const cy = y + h / 2;
    const s = Math.min(w, h) * 0.18;

    ctx.save();

    // กรอบหลัก
    ctx.strokeStyle = color;
    ctx.lineWidth = DRAW.lineWidth.humanBox;
    ctx.setLineDash([]);
    ctx.strokeRect(x, y, w, h);

    // เส้นทแยง
    ctx.globalAlpha = DRAW.alpha.humanDiag;
    ctx.lineWidth = DRAW.lineWidth.humanDiag;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(x, y); ctx.lineTo(x + w, y + h);
    ctx.moveTo(x + w, y); ctx.lineTo(x, y + h);
    ctx.stroke();
    ctx.globalAlpha = 1;

    // จุดกลาง
    ctx.fillStyle = DRAW.centerDotColor;
    ctx.beginPath();
    ctx.arc(cx, cy, DRAW.radius.centerDot, 0, Math.PI * 2);
    ctx.fill();

    // Corner L-marks
    ctx.strokeStyle = color;
    ctx.lineWidth = DRAW.lineWidth.humanCorner;
    ctx.lineCap = "square";
    ctx.beginPath();
    ctx.moveTo(x + s, y); ctx.lineTo(x, y); ctx.lineTo(x, y + s);
    ctx.moveTo(x + w - s, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + s);
    ctx.moveTo(x, y + h - s); ctx.lineTo(x, y + h); ctx.lineTo(x + s, y + h);
    ctx.moveTo(x + w - s, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - s);
    ctx.stroke();

    drawTag(ctx, x, y, `ID ${id ?? "?"} ${(conf * 100).toFixed(0)}%`, color);
    ctx.restore();
}

// drawSkeleton — อ่านทุกค่าจาก DRAW
function drawSkeleton(ctx, keypoints, scaleX, scaleY, canvasW) {
    const minConf = DRAW.kpConfMin;
    const pts = keypoints.map(kp => ({
        x: canvasW - kp.x * scaleX,
        y: kp.y * scaleY,
        vis: (kp.x > 0 || kp.y > 0) && (kp.conf ?? 1) >= minConf,
    }));

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = DRAW.lineWidth.skeleton;

    // group เส้นตามสี → stroke ครั้งเดียวต่อกลุ่ม
    const groups = {};
    SKELETON_EDGES.forEach(([a, b], i) => {
        if (!pts[a]?.vis || !pts[b]?.vis) return;
        const col = EDGE_COLOR_MAP[i];
        if (!groups[col]) groups[col] = [];
        groups[col].push([pts[a].x, pts[a].y, pts[b].x, pts[b].y]);
    });
    for (const [col, segs] of Object.entries(groups)) {
        ctx.strokeStyle = col;
        ctx.beginPath();
        for (const [x1, y1, x2, y2] of segs) { ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); }
        ctx.stroke();
    }

    // keypoint pass 1: วงนอก
    ctx.fillStyle = `rgba(0,0,0,${DRAW.alpha.kpOuter})`;
    ctx.beginPath();
    for (const p of pts) {
        if (!p.vis) continue;
        ctx.moveTo(p.x + DRAW.radius.kpOuter, p.y);
        ctx.arc(p.x, p.y, DRAW.radius.kpOuter, 0, Math.PI * 2);
    }
    ctx.fill();

    // keypoint pass 2: วงใน
    ctx.globalAlpha = DRAW.alpha.kpInner;
    ctx.fillStyle = TYPE_COLORS.pose;
    ctx.beginPath();
    for (const p of pts) {
        if (!p.vis) continue;
        ctx.moveTo(p.x + DRAW.radius.kpInner, p.y);
        ctx.arc(p.x, p.y, DRAW.radius.kpInner, 0, Math.PI * 2);
    }
    ctx.fill();
    ctx.globalAlpha = 1;

    ctx.restore();
}

function drawTag(ctx, x, y, text, color, below = false) {
    ctx.font = "bold 10px 'Share Tech Mono', monospace";
    const tw = ctx.measureText(text).width;
    const pad = 4;
    const tagH = 16;
    const tagY = below ? y + 2 : y - tagH - 2;
    ctx.fillStyle = color + "22";
    ctx.fillRect(x, tagY, tw + pad * 2, tagH);
    ctx.fillStyle = color;
    ctx.fillText(text, x + pad, tagY + 11);
}

// ============================================================
// Stats
// ============================================================
function updateStats() {
    valTxFps.textContent = txFps.toFixed(1);
    valRxFps.textContent = rxFps.toFixed(1);
    valLat.textContent = latencyMs;
    valInfer.textContent = inferMs.toFixed(0);
    valLat.className = "stat-value" + (latencyMs > 300 ? " warn" : " ok");
    valInfer.className = "stat-value" + (inferMs > 200 ? " warn" : " ok");
}
setInterval(updateStats, 1000);

// ============================================================
// Detection Log
// ============================================================
function addLog(type, typeLabel, detail) {
    const now = new Date().toLocaleTimeString("th-TH", { hour12: false });
    logEntries.unshift({ ts: now, type, typeLabel, detail });
    if (logEntries.length > MAX_LOG) logEntries.pop();
    renderLog();
}

// FIX: ใช้ DocumentFragment แทน innerHTML ทุก frame
function renderLog() {
    const frag = document.createDocumentFragment();
    for (const e of logEntries) {
        const row = document.createElement("div");
        row.className = "log-entry";
        row.innerHTML = `<span class="log-ts">${e.ts}</span><span class="log-type ${e.type}">${e.typeLabel}</span><span class="log-detail">${e.detail}</span>`;
        frag.appendChild(row);
    }
    detLog.replaceChildren(frag);
}

// ============================================================
// Init
// ============================================================
window.addEventListener("load", async () => {
    initCanvas();
    await openCamera();
    connectWS();
    streaming = true;
    loop();
    drawLoop();
});