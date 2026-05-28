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

    // --- สี skeleton แยกฝั่ง ---
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

// ============================================================
// รับค่า Parameter จาก setup.html ผ่าน URL Query String
// ============================================================
const setupParams = new URLSearchParams(window.location.search);

const SETUP = {
    playMethod: setupParams.get("playMethod") || "sitting",
    objective:  setupParams.get("objective")  || "attention",
    gameName:   setupParams.get("gameName")   || "—",
};

// Map ค่า value → ข้อความภาษาไทย
const PLAY_LABEL = {
    sitting:  "นั่งเล่น",
    standing: "ยืนเล่น",
};
const OBJ_LABEL = {
    attention:   "ฝึกสมาธิและความนิ่ง",
    fine_motor:  "กล้ามเนื้อมัดเล็ก",
    gross_motor: "กล้ามเนื้อมัดใหญ่",
};

function applySetupParams() {
    // แสดงค่าใน info block
    const elPlay = document.getElementById("infoPlayMethod");
    const elObj  = document.getElementById("infoObjective");
    const elGame = document.getElementById("infoGameName");

    if (elPlay) elPlay.textContent = PLAY_LABEL[SETUP.playMethod] || SETUP.playMethod;
    if (elObj)  elObj.textContent  = OBJ_LABEL[SETUP.objective]   || SETUP.objective;
    if (elGame) elGame.textContent = decodeURIComponent(SETUP.gameName);
}

// เรียกหลัง DOM โหลดเสร็จ
document.addEventListener("DOMContentLoaded", applySetupParams);

// ============================================================
// State
// ============================================================
let ws = null;
let waitingReply = false;
let streaming = false;
let timeoutTimer = null;
let mode = "camera";
let videoFileEl = document.getElementById("videoFile");
let recorder = null;
let chunks = [];
let currentVideoObjectURL = null;

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
let trackedChildren = [];
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
// Pre-computed skeleton edges
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
// Mode / Video input handling
// ============================================================
document.querySelectorAll('input[name="mode"]').forEach(r => {
    r.addEventListener("change", (e) => {
        mode = e.target.value;

        if (mode === "video") {
            if (video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
                video.srcObject = null;
            }
            video.pause();
            if (videoFileEl) {
                videoFileEl.style.display = "block";
            }
        } else {
            if (videoFileEl) {
                videoFileEl.style.display = "none";
                videoFileEl.value = "";
            }
            if (currentVideoObjectURL) {
                URL.revokeObjectURL(currentVideoObjectURL);
                currentVideoObjectURL = null;
            }
            if (video.src && video.src.startsWith("blob:")) {
                video.pause();
                video.removeAttribute("src");
                video.load();
            }
            openCamera();
        }
    });
});

if (videoFileEl) {
    videoFileEl.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (currentVideoObjectURL) {
            URL.revokeObjectURL(currentVideoObjectURL);
        }

        currentVideoObjectURL = URL.createObjectURL(file);

        video.srcObject = null;
        video.src = currentVideoObjectURL;
        video.loop = true;
        video.muted = true;
        video.play().catch((err) => {
            console.warn("Unable to play selected video file:", err);
        });
    });
}

// ============================================================
// Recorder helpers
// ============================================================
function startRecording() {
    if (!overlay || typeof overlay.captureStream !== "function") {
        console.warn("overlay.captureStream is not available");
        return;
    }

    if (recorder && recorder.state !== "inactive") {
        recorder.stop();
    }

    chunks = [];
    const stream = overlay.captureStream(30);
    recorder = new MediaRecorder(stream);

    recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
            chunks.push(e.data);
        }
    };

    recorder.onstop = () => {
        const blob = new Blob(chunks, { type: "video/webm" });
        const url = URL.createObjectURL(blob);

        const a = document.createElement("a");
        a.href = url;
        a.download = "result.webm";
        document.body.appendChild(a);
        a.click();
        a.remove();

        URL.revokeObjectURL(url);
        chunks = [];
        recorder = null;
    };

    recorder.onerror = (event) => {
        console.error("MediaRecorder error:", event.error);
    };

    recorder.start();
}

function stopRecording() {
    if (recorder && recorder.state !== "inactive") {
        recorder.stop();
    }
}

// ============================================================
// Clock
// ============================================================
setInterval(() => {
    clockEl.textContent = new Date().toLocaleTimeString("th-TH", { hour12: false });
}, 1000);

// ============================================================
// Canvas initialization
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
        
        // ส่ง parameters ไปยัง server
        const initMsg = JSON.stringify({
            playMethod: SETUP.playMethod,
            objective: SETUP.objective,
            gameName: SETUP.gameName
        });
        ws.send(initMsg);
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
        trackedChildren = msg.tracked_children || [];

        // update UI
        countHuman.textContent = latestHumans.length;
        countFace.textContent = latestFaces.length;
        countPose.textContent = latestPoses.length;
        inferBadge.querySelector("span").textContent = inferMs.toFixed(1);
        updateStats();

        for (const b of latestHumans) addLog("human", "HUMAN", `ID:${b.id ?? "?"} conf:${(b.conf * 100).toFixed(0)}%`);
        for (const b of latestFaces) addLog("face", "FACE", `conf:${(b.conf * 100).toFixed(0)}%`);

        // Update chart data and scores
        collectChartData();
        updateScoreData();
        if (currentPanel === "graph") drawCharts();
        if (currentPanel === "score") renderScorePanel();
    };
}

// ============================================================
// Send frame
// ============================================================
const sendCanvas = document.createElement("canvas");
const sendCtx = sendCanvas.getContext("2d");
sendCtx.imageSmoothingEnabled = false;

function sendFrame() {
    if (!ws || ws.readyState !== 1 || waitingReply || !video.videoWidth) return;

    const ratio = video.videoHeight / video.videoWidth;
    const W = SEND_W, H = Math.round(SEND_W * ratio);
    if (sendCanvas.width !== W || sendCanvas.height !== H) {
        sendCanvas.width = W; sendCanvas.height = H;
    }

    sendCtx.drawImage(video, 0, 0, W, H);

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
// Draw loop
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

// ============================================================
// HAMBURGER DRAWER SYSTEM
// ============================================================

const hamburgerBtn  = document.getElementById("hamburgerBtn");
const drawer        = document.getElementById("drawer");
const drawerOverlay = document.getElementById("drawerOverlay");
const drawerBack    = document.getElementById("drawerBack");
const drawerBackLabel = document.getElementById("drawerBackLabel");
const drawerTitle   = document.getElementById("drawerTitle");
const drawerMain    = document.getElementById("drawerMain");
const drawerWsStatus = document.getElementById("drawerWsStatus");

// Panel elements
const panelGraph    = document.getElementById("panelGraph");
const panelScore    = document.getElementById("panelScore");
const panelSettings = document.getElementById("panelSettings");

let drawerOpen = false;
let currentPanel = null;

function openDrawer() {
    drawerOpen = true;
    drawer.classList.add("open");
    drawerOverlay.classList.add("show");
    hamburgerBtn.classList.add("open");
    showMainMenu();
}

function closeDrawer() {
    drawerOpen = false;
    drawer.classList.remove("open");
    drawerOverlay.classList.remove("show");
    hamburgerBtn.classList.remove("open");
    currentPanel = null;
}

function showMainMenu() {
    currentPanel = null;
    drawerMain.style.display = "";
    [panelGraph, panelScore, panelSettings].forEach(p => {
        p.classList.remove("active");
        p.style.display = "";
    });
    drawerBackLabel.textContent = "ปิดเมนู";
    drawerTitle.textContent = "NON-Autos\u00a0Mine";
}

function showPanel(panelId) {
    const panelMap = {
        graph:    { el: panelGraph,    title: "ข้อมูลกราฟ" },
        score:    { el: panelScore,    title: "คะแนนผู้เข้าร่วม" },
        settings: { el: panelSettings, title: "ตั้งค่าพารามิเตอร์" },
    };
    const p = panelMap[panelId];
    if (!p) return;

    currentPanel = panelId;
    drawerMain.style.display = "none";
    [panelGraph, panelScore, panelSettings].forEach(el => el.classList.remove("active"));
    p.el.classList.add("active");
    drawerBackLabel.textContent = "ย้อนกลับ";
    drawerTitle.textContent = p.title;

    if (panelId === "score") renderScorePanel();
    if (panelId === "graph") initCharts();
}

hamburgerBtn.addEventListener("click", () => {
    drawerOpen ? closeDrawer() : openDrawer();
});

drawerOverlay.addEventListener("click", closeDrawer);

drawerBack.addEventListener("click", () => {
    if (currentPanel) {
        showMainMenu();
    } else {
        closeDrawer();
    }
});

document.querySelectorAll(".menu-item").forEach(btn => {
    btn.addEventListener("click", () => showPanel(btn.dataset.panel));
});

document.addEventListener("keydown", e => {
    if (e.key === "Escape") {
        if (currentPanel) showMainMenu();
        else if (drawerOpen) closeDrawer();
    }
});

function syncDrawerWsStatus() {
    if (!drawerWsStatus) return;
    const connected = wsDot.classList.contains("on");
    drawerWsStatus.textContent = connected ? "CONNECTED" : "DISCONNECTED";
    drawerWsStatus.style.color = connected ? "var(--col-pose)" : "var(--red)";
}
setInterval(syncDrawerWsStatus, 1500);

// ============================================================
// CHIPS (single-select)
// ============================================================
document.querySelectorAll(".setting-chips").forEach(group => {
    group.addEventListener("click", e => {
        const chip = e.target.closest(".chip");
        if (!chip) return;
        group.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
        chip.classList.add("active");
    });
});

// ============================================================
// SCORE PANEL
// ============================================================
const scoreData = {};   // track_id → { color, scores }

function updateScoreData() {
    for (const child of trackedChildren) {
        const id = child.track_id ?? "?";
        if (!scoreData[id]) {
            scoreData[id] = {
                color: getIdColor(child.track_id),
                name: child.name,
                scores: { attention: 0, fine_motor: 0, gross_motor: 0 }
            };
        }
        // Update scores from server
        if (child.scores) {
            scoreData[id].scores = child.scores;
        }
    }
}

function renderScorePanel() {
    const list = document.getElementById("scoreList");
    if (!list) return;
    const ids = Object.keys(scoreData);
    if (ids.length === 0) {
        list.innerHTML = '<div class="score-empty">ยังไม่มีข้อมูลผู้เข้าร่วม<br>รอให้ตรวจจับ Human…</div>';
        return;
    }
    list.innerHTML = ids.map(id => {
        const d = scoreData[id];
        const scores = d.scores;
        const avgScore = Math.round(
            (scores.attention + scores.fine_motor + scores.gross_motor) / 3
        );
        return `
        <div class="score-card">
          <div class="score-card-header">
            <span class="score-id-dot" style="background:${d.color}"></span>
            <span class="score-id-label">${d.name}</span>
            <span class="score-badge">${avgScore} pt</span>
          </div>
          <div class="score-breakdown">
            <div class="score-row">
              <span>สมาธิ</span>
              <span>${scores.attention}</span>
            </div>
            <div class="score-row">
              <span>กล้ามเนื้อมัดเล็ก</span>
              <span>${scores.fine_motor}</span>
            </div>
            <div class="score-row">
              <span>กล้ามเนื้อมัดใหญ่</span>
              <span>${scores.gross_motor}</span>
            </div>
          </div>
          <div class="score-bar-wrap">
            <div class="score-bar-fill" style="width:${avgScore}%;background:${d.color}"></div>
          </div>
        </div>`;
    }).join("");
}

// ============================================================
// GRAPH PANEL
// ============================================================
const CHART_LEN = 60;

const speedHistory = new Array(CHART_LEN).fill(0);
let lastCenterX = null, lastCenterY = null;

const kpHistory = {
    head:  new Array(CHART_LEN).fill(null),
    armL:  new Array(CHART_LEN).fill(null),
    armR:  new Array(CHART_LEN).fill(null),
    torso: new Array(CHART_LEN).fill(null),
};

let chartsInitialized = false;

function pushVal(arr, val) {
    arr.push(val);
    if (arr.length > CHART_LEN) arr.shift();
}

function collectChartData() {
    if (latestHumans.length > 0) {
        const h = latestHumans[0];
        const cx = h.x + h.w / 2;
        const cy = h.y + h.h / 2;
        if (lastCenterX !== null) {
            const dx = cx - lastCenterX, dy = cy - lastCenterY;
            pushVal(speedHistory, Math.sqrt(dx * dx + dy * dy));
        }
        lastCenterX = cx; lastCenterY = cy;
    } else {
        pushVal(speedHistory, 0);
    }

    if (latestPoses.length > 0 && latestPoses[0].keypoints) {
        const kp = latestPoses[0].keypoints;
        const get = (i) => (kp[i] && (kp[i].conf ?? 0) >= DRAW.kpConfMin) ? kp[i].y : null;
        pushVal(kpHistory.head,  get(0));
        pushVal(kpHistory.armL,  get(9));
        pushVal(kpHistory.armR,  get(10));
        pushVal(kpHistory.torso, (get(5) !== null && get(11) !== null) ? (kp[5].y + kp[11].y) / 2 : null);
    } else {
        Object.values(kpHistory).forEach(arr => pushVal(arr, null));
    }
}

setInterval(() => {
    collectChartData();
    updateScoreData();
    if (currentPanel === "graph") drawCharts();
    if (currentPanel === "score") renderScorePanel();
}, 200);

function initCharts() {
    if (!chartsInitialized) chartsInitialized = true;
    drawCharts();
}

function drawCharts() {
    drawLineChart(
        document.getElementById("chartSpeed"),
        [{ data: speedHistory, color: "#ff3333" }],
        { bg: "rgba(255,51,51,0.04)", gridColor: "rgba(26,24,20,0.06)" }
    );
    drawLineChart(
        document.getElementById("chartKeypoint"),
        [
            { data: kpHistory.head,  color: "rgba(200,200,255,0.85)" },
            { data: kpHistory.armL,  color: "rgba(80,200,255,0.9)" },
            { data: kpHistory.armR,  color: "rgba(255,220,50,0.9)" },
            { data: kpHistory.torso, color: "rgba(105,255,71,0.85)" },
        ],
        { bg: "rgba(105,255,71,0.03)", gridColor: "rgba(26,24,20,0.06)", invert: true }
    );
}

function drawLineChart(canvas, series, opts = {}) {
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const W = rect.width || canvas.parentElement.clientWidth || 260;
    const H = canvas.height;

    canvas.width  = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    canvas.style.height = H + "px";

    const ctx = canvas.getContext("2d");
    ctx.setTransform(1,0,0,1,0,0);
    ctx.scale(dpr, dpr);

    ctx.fillStyle = opts.bg || "rgba(242,239,233,0.6)";
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = opts.gridColor || "rgba(0,0,0,0.06)";
    ctx.lineWidth = 0.5;
    for (let i = 1; i < 4; i++) {
        const gy = (H / 4) * i;
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
    }

    const padT = 4, padB = 4;
    const plotH = H - padT - padB;

    let allVals = [];
    series.forEach(s => s.data.forEach(v => { if (v !== null) allVals.push(v); }));
    let minV = allVals.length ? Math.min(...allVals) : 0;
    let maxV = allVals.length ? Math.max(...allVals) : 1;
    if (maxV === minV) { maxV = minV + 1; }

    series.forEach(s => {
        const len = s.data.length;
        if (len < 2) return;
        const step = W / (CHART_LEN - 1);

        ctx.strokeStyle = s.color;
        ctx.lineWidth = 1.5;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < len; i++) {
            const v = s.data[i];
            if (v === null) { started = false; continue; }
            const nx = i * step;
            const pct = (v - minV) / (maxV - minV);
            const ny = opts.invert
                ? padT + pct * plotH
                : padT + (1 - pct) * plotH;
            if (!started) { ctx.moveTo(nx, ny); started = true; }
            else ctx.lineTo(nx, ny);
        }
        ctx.stroke();
    });
}

// ============================================================
// Dashboard Export
// ============================================================
const goDashboardBtn = document.getElementById("goDashboardBtn");

function exportToDashboard() {
    // Prepare data with colors
    const dataForExport = trackedChildren.map(child => ({
        ...child,
        color: idColors[child.track_id] || COLOR_PALETTE[0]
    }));

    // Save to sessionStorage
    sessionStorage.setItem('sessionScores', JSON.stringify(dataForExport));
    sessionStorage.setItem('playMethod', SETUP.playMethod);
    sessionStorage.setItem('objective', SETUP.objective);
    sessionStorage.setItem('gameName', SETUP.gameName);

    // Navigate to Dashboard
    window.location.href = 'Dashboard.html';
}

if (goDashboardBtn) {
    goDashboardBtn.addEventListener('click', exportToDashboard);
}