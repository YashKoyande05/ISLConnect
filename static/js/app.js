/* ─── ISLENS · Live Translation Frontend ─────────────────────── */

const $ = (id) => document.getElementById(id);

/* DOM refs */
const video          = $("video");
const canvas         = $("hiddenCanvas");
const startBtn       = $("startBtn");
const stopBtn        = $("stopBtn");
const liveIndicator  = $("liveIndicator");
const prediction     = $("prediction");
const confidenceVal  = $("confidenceVal");
const confidenceBar  = $("confidenceBar");
const fpsDisplay     = $("fpsDisplay");
const message        = $("message");
const translatedText = $("translatedText");
const modelStatus    = $("modelStatus");
const classGrid      = $("classGrid");
const classCount     = $("classCount");

/* ── State ─────────────────────────────────────────────────────── */
let stream       = null;
let liveInterval = null;
let isLive       = false;

let text             = "";
let recentPrediction = [];   // rolling buffer for stability
let lastAccepted     = null;

let frameCount = 0;
let fpsTimer   = Date.now();

const INTERVAL_MS   = 120;  // ~8 FPS prediction rate
const STABLE_NEEDED = 3;    // how many matching frames to accept a letter
const BUFFER_SIZE   = 5;    // rolling window size
const MIN_CONFIDENCE = 0.70;

/* ── API status ─────────────────────────────────────────────────── */
async function loadStatus() {
    try {
        const data = await fetch("/api/status").then(r => r.json());
        if (data.model_ready) {
            modelStatus.classList.add("ready");
            modelStatus.innerHTML = '<span class="status-dot"></span>Model ready';
        } else {
            modelStatus.innerHTML = '<span class="status-dot"></span>Model not trained';
        }
        classGrid.innerHTML = "";
        data.classes.forEach(label => {
            const el = document.createElement("div");
            el.className = "class-item";
            el.textContent = label.toUpperCase();
            classGrid.appendChild(el);
        });
        classCount.textContent = `${data.classes.length} classes`;
    } catch {
        modelStatus.innerHTML = '<span class="status-dot"></span>Backend offline';
    }
}

/* ── Camera ──────────────────────────────────────────────────────── */
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }
        });
        video.srcObject = stream;
        await video.play();
        startLive();
    } catch (err) {
        setMessage("⚠ Camera unavailable: " + err.message);
    }
}

function stopCamera() {
    stopLive();
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        stream = null;
    }
    video.srcObject = null;
    resetPrediction();
    setMessage("Camera stopped.");
}

/* ── Live loop ───────────────────────────────────────────────────── */
function startLive() {
    isLive = true;
    liveIndicator.classList.add("active");
    startBtn.disabled = true;
    stopBtn.disabled  = false;
    recentPrediction  = [];
    lastAccepted      = null;
    frameCount        = 0;
    fpsTimer          = Date.now();

    liveInterval = setInterval(sendFrame, INTERVAL_MS);
    setMessage("Live translation active — show a hand sign.");
}

function stopLive() {
    isLive = false;
    clearInterval(liveInterval);
    liveInterval = null;
    liveIndicator.classList.remove("active");
    startBtn.disabled = false;
    stopBtn.disabled  = true;
}

async function sendFrame() {
    if (!video.videoWidth) return;

    /* capture frame */
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    /* FPS counter */
    frameCount++;
    const now = Date.now();
    if (now - fpsTimer >= 1000) {
        fpsDisplay.textContent = `${frameCount} FPS`;
        frameCount = 0;
        fpsTimer   = now;
    }

    let blob;
    try {
        blob = await new Promise(res => canvas.toBlob(res, "image/jpeg", 0.80));
    } catch { return; }

    const form = new FormData();
    form.append("image", blob, "frame.jpg");

    try {
        const resp = await fetch("/api/predict", { method: "POST", body: form });
        const data = await resp.json();

        if (!resp.ok || !data.success) {
            setMessage(data.error || "Prediction error.");
            return;
        }

        if (!data.hand_detected) {
            resetPrediction();
            setMessage("No hand detected — hold a sign up to the camera.");
            return;
        }

        updatePrediction(data.prediction, data.confidence);

        if (data.confidence >= MIN_CONFIDENCE) {
            stabilize(data.prediction);
        } else {
            recentPrediction = [];
        }

    } catch { /* network blip — ignore */ }
}

/* ── UI helpers ──────────────────────────────────────────────────── */
function updatePrediction(label, conf) {
    prediction.textContent = label ? label.toUpperCase() : "—";
    const pct = Math.round(conf * 100);
    confidenceVal.textContent = `${pct}%`;
    confidenceBar.style.width = `${pct}%`;
    confidenceBar.style.background =
        pct >= 90 ? "var(--accent)" :
        pct >= 70 ? "#f5d020"       : "#ff6b6b";
    setMessage("Detecting…");

    /* highlight matching class badge */
    document.querySelectorAll(".class-item").forEach(el => {
        el.classList.toggle("active", el.textContent === label?.toUpperCase());
    });
}

function resetPrediction() {
    prediction.textContent    = "—";
    confidenceVal.textContent = "—";
    confidenceBar.style.width = "0";
    recentPrediction          = [];
    document.querySelectorAll(".class-item.active")
        .forEach(el => el.classList.remove("active"));
}

function setMessage(msg) {
    message.textContent = msg;
}

/* ── Stability buffer ────────────────────────────────────────────── */
function stabilize(label) {
    recentPrediction.push(label);
    if (recentPrediction.length > BUFFER_SIZE) recentPrediction.shift();

    const counts = {};
    recentPrediction.forEach(x => counts[x] = (counts[x] || 0) + 1);
    const [top, cnt] = Object.entries(counts).sort((a,b) => b[1]-a[1])[0];

    if (cnt >= STABLE_NEEDED && top !== lastAccepted) {
        lastAccepted = top;
        text += top;
        renderText();
        /* brief visual flash */
        prediction.classList.add("flash");
        setTimeout(() => prediction.classList.remove("flash"), 300);
    }
}

/* ── Text controls ───────────────────────────────────────────────── */
$("btnSpace").addEventListener("click", () => {
    if (text && !text.endsWith(" ")) {
        text       += " ";
        lastAccepted = null;
        recentPrediction = [];
        renderText();
    }
});

$("btnBackspace").addEventListener("click", () => {
    text = text.slice(0, -1);
    lastAccepted     = null;
    recentPrediction = [];
    renderText();
});

$("btnClear").addEventListener("click", () => {
    text             = "";
    lastAccepted     = null;
    recentPrediction = [];
    renderText();
});

function renderText() {
    translatedText.textContent = text || "Start signing…";
}

/* ── Button wiring ───────────────────────────────────────────────── */
startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click",  stopCamera);

/* ── Init ────────────────────────────────────────────────────────── */
loadStatus();
