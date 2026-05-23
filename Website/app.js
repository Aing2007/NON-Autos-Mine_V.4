// ============================================================
// app.js — AI Detection Monitor
// จัดการ Camera, WebSocket, และวาด bounding boxes
// ============================================================

const WS_URL = "ws://localhost:8000/ws";
const SEND_W = 640;
const JPEG_Q = 0.72;
const WS_TIMEOUT_MS = 3500;  // reset waitingReply ถ้าไม่ได้ reply

// สีต่อ type
const TYPE_COLORS = {
    human: "#00e5ff",
    face: "#ff4081",
    pose: "#69ff47",
};

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

// Latest boxes from server
let latestHumans = [];
let latestFaces = [];
let latestPoses = [];
let srcW = 0, srcH = 0;
let inferMs = 0;

// Log entries (sidebar)
const MAX_LOG = 60;
const logEntries = [];

// Color cache (ID → color, for human tracking)
const idColors = {};
const COLOR_PALETTE = [
    "#00e5ff", "#ff4081", "#69ff47", "#ffab00",
    "#a29bfe", "#fd79a8", "#55efc4", "#fdcb6e", "#74b9ff", "#ff7675"
];
let colorIdx = 0;
function getIdColor(id) {
    if (id == null) return TYPE_COLORS.human;
    if (!idColors[id]) {
        idColors[id] = COLOR_PALETTE[colorIdx % COLOR_PALETTE.length];
        colorIdx++;
    }
    return idColors[id];
}

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
function updateClock() {
    const now = new Date();
    clockEl.textContent = now.toLocaleTimeString("th-TH", { hour12: false });
}
setInterval(updateClock, 1000);
updateClock();

// ============================================================
// HiDPI Canvas
// ============================================================
function setupHiDPI(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
    return { ctx, w: rect.width, h: rect.height, dpr };
}

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
        document.querySelector(".cam-offline-text").textContent =
            "กรุณาอนุญาตการใช้กล้อง";
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
        // ยกเลิก timeout timer
        clearTimeout(timeoutTimer);
        waitingReply = false;

        // Latency
        latencyMs = Date.now() - lastSendMs;

        // Stats
        rxCount++;
        const now = Date.now();
        const dtRx = (now - rxTs) / 1000;
        if (dtRx >= 1) {
            rxFps = rxCount / dtRx;
            rxCount = 0;
            rxTs = now;
        }

        // Parse
        let msg;
        try { msg = JSON.parse(ev.data); }
        catch { return; }

        srcW = msg.w || srcW;
        srcH = msg.h || srcH;
        inferMs = msg.inference_ms || 0;

        latestHumans = msg.humans || [];
        latestFaces = msg.faces || [];
        latestPoses = msg.poses || [];

        // Update sidebar counts
        countHuman.textContent = latestHumans.length;
        countFace.textContent = latestFaces.length;
        countPose.textContent = latestPoses.length;
        inferBadge.querySelector("span").textContent = inferMs.toFixed(1);

        // Update stat cells
        updateStats();

        // Log new detections (human only to avoid flooding)
        for (const b of latestHumans) {
            addLog("human", "HUMAN", `ID:${b.id ?? "?"} conf:${(b.conf * 100).toFixed(0)}%`);
        }
        for (const b of latestFaces) {
            addLog("face", "FACE", `conf:${(b.conf * 100).toFixed(0)}%`);
        }
    };
}

// ============================================================
// Send frame
// ============================================================
const sendCanvas = document.createElement("canvas");
const sendCtx = sendCanvas.getContext("2d");
sendCtx.imageSmoothingEnabled = false;

function sendFrame() {
    if (!ws || ws.readyState !== 1) return;
    if (waitingReply) return;
    if (!video.videoWidth) return;

    const ratio = video.videoHeight / video.videoWidth;
    const targetW = SEND_W;
    const targetH = Math.round(SEND_W * ratio);

    if (sendCanvas.width !== targetW || sendCanvas.height !== targetH) {
        sendCanvas.width = targetW;
        sendCanvas.height = targetH;
    }

    sendCtx.drawImage(video, 0, 0, targetW, targetH);
    sendCanvas.toBlob((blob) => {
        if (!blob || ws.readyState !== 1) return;
        lastSendMs = Date.now();
        ws.send(blob);
        waitingReply = true;

        // Safety timeout
        timeoutTimer = setTimeout(() => {
            waitingReply = false;
        }, WS_TIMEOUT_MS);

        // Tx stats
        txCount++;
        const now = Date.now();
        const dtTx = (now - txTs) / 1000;
        if (dtTx >= 1) {
            txFps = txCount / dtTx;
            txCount = 0;
            txTs = now;
        }
    }, "image/jpeg", JPEG_Q);
}

function loop() {
    if (streaming) {
        sendFrame();
        requestAnimationFrame(loop);
    }
}

// ============================================================
// Draw bounding boxes
// ============================================================
function drawBoxes() {
    const { ctx, w, h } = setupHiDPI(overlay);

    function draw() {
        ctx.clearRect(0, 0, w, h);

        const vw = video.videoWidth;
        const vh = video.videoHeight;
        if (!vw || !vh) { requestAnimationFrame(draw); return; }

        const SW = srcW || vw;
        const SH = srcH || vh;
        const scaleX = w / SW;
        const scaleY = h / SH;

        // Helper: mirror x (เพราะ video ถูก mirror ด้วย CSS scaleX(-1))
        const mx = (bx, bw) => w - (bx + bw) * scaleX;

        // ---- Human boxes (cyan, with ID) ----
        for (const b of latestHumans) {
            const color = getIdColor(b.id);
            const bx = mx(b.x, b.w);
            const by = b.y * scaleY;
            const bw = b.w * scaleX;
            const bh = b.h * scaleY;

            // กรอบ
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;
            ctx.strokeRect(bx, by, bw, bh);

            // Corner marks
            drawCorners(ctx, bx, by, bw, bh, color);

            // Label
            drawTag(ctx, bx, by, `ID ${b.id ?? "?"} ${(b.conf * 100).toFixed(0)}%`, color);
        }

        // ---- Face boxes (pink, solid) ----
        for (const b of latestFaces) {
            const bx = mx(b.x, b.w);
            const by = b.y * scaleY;
            const bw = b.w * scaleX;
            const bh = b.h * scaleY;

            ctx.strokeStyle = TYPE_COLORS.face;
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 3]);
            ctx.strokeRect(bx, by, bw, bh);
            ctx.setLineDash([]);

            drawTag(ctx, bx, by, `FACE ${(b.conf * 100).toFixed(0)}%`, TYPE_COLORS.face);
        }

        // ---- Pose boxes + keypoints (green) ----
        for (const b of latestPoses) {
            const bx = mx(b.x, b.w);
            const by = b.y * scaleY;
            const bw = b.w * scaleX;
            const bh = b.h * scaleY;

            ctx.strokeStyle = TYPE_COLORS.pose;
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 4]);
            ctx.strokeRect(bx, by, bw, bh);
            ctx.setLineDash([]);

            drawTag(ctx, bx, by + bh, `POSE ${(b.conf * 100).toFixed(0)}%`, TYPE_COLORS.pose, true);

            // Keypoints
            if (b.keypoints && b.keypoints.length > 0) {
                for (const kp of b.keypoints) {
                    if (!kp.x && !kp.y) continue;
                    const kx = w - kp.x * scaleX;
                    const ky = kp.y * scaleY;
                    ctx.beginPath();
                    ctx.arc(kx, ky, 2.5, 0, Math.PI * 2);
                    ctx.fillStyle = TYPE_COLORS.pose;
                    ctx.fill();
                }
            }
        }

        requestAnimationFrame(draw);
    }
    requestAnimationFrame(draw);
}

// Corner L-marks
function drawCorners(ctx, x, y, w, h, color) {
    const s = Math.min(w, h) * 0.2;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineCap = "square";
    const corners = [
        [x, y, 1, 1], [x + w, y, -1, 1],
        [x, y + h, 1, -1], [x + w, y + h, -1, -1],
    ];
    for (const [cx, cy, dx, dy] of corners) {
        ctx.beginPath();
        ctx.moveTo(cx + dx * s, cy);
        ctx.lineTo(cx, cy);
        ctx.lineTo(cx, cy + dy * s);
        ctx.stroke();
    }
}

// Tag label above/below box
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
// Stats update
// ============================================================
function updateStats() {
    valTxFps.textContent = txFps.toFixed(1);
    valRxFps.textContent = rxFps.toFixed(1);
    valLat.textContent = latencyMs;
    valInfer.textContent = inferMs.toFixed(0);

    // Color coding
    valLat.className = "stat-value" + (latencyMs > 300 ? " warn" : " ok");
    valInfer.className = "stat-value" + (inferMs > 200 ? " warn" : " ok");
}

// Stats polling (ทุก 1s)
setInterval(updateStats, 1000);

// ============================================================
// Detection Log (sidebar)
// ============================================================
function addLog(type, typeLabel, detail) {
    const now = new Date().toLocaleTimeString("th-TH", { hour12: false });
    logEntries.unshift({ ts: now, type, typeLabel, detail });
    if (logEntries.length > MAX_LOG) logEntries.pop();
    renderLog();
}

function renderLog() {
    detLog.innerHTML = logEntries.map(e => `
    <div class="log-entry">
      <span class="log-ts">${e.ts}</span>
      <span class="log-type ${e.type}">${e.typeLabel}</span>
      <span class="log-detail">${e.detail}</span>
    </div>
  `).join("");
}

// ============================================================
// ResizeObserver — recalc canvas on resize
// ============================================================
const ro = new ResizeObserver(() => setupHiDPI(overlay));
ro.observe(overlay);

// ============================================================
// Init
// ============================================================
window.addEventListener("load", async () => {
    await openCamera();
    connectWS();
    streaming = true;
    loop();
    drawBoxes();
});