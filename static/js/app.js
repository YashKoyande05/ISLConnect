/* ─── ISLENS · Live Translation Frontend with AI Sentence Modeling ─── */

const $ = (id) => document.getElementById(id);

/* DOM refs */
const video                  = $("video");
const canvas                 = $("hiddenCanvas");
const startBtn               = $("startBtn");
const stopBtn                = $("stopBtn");
const liveIndicator          = $("liveIndicator");
const prediction             = $("prediction");
const confidenceVal          = $("confidenceVal");
const confidenceBar          = $("confidenceBar");
const fpsDisplay             = $("fpsDisplay");
const message                = $("message");
const glossContainer         = $("glossContainer");
const translatedText         = $("translatedText");
const translationSourceBadge = $("translationSourceBadge");
const modelStatus            = $("modelStatus");
const aiStatus               = $("aiStatus");
const aiStatusText           = $("aiStatusText");
const classGrid              = $("classGrid");
const classCount             = $("classCount");

const btnTranslate           = $("btnTranslate");
const btnSpeak               = $("btnSpeak");
const btnCopy                = $("btnCopy");
const btnClear               = $("btnClear");
const btnBackspace           = $("btnBackspace");
const btnSpace               = $("btnSpace");

/* ── State ─────────────────────────────────────────────────────── */
let stream           = null;
let liveInterval     = null;
let isLive           = false;

let detectedGlosses  = [];     // List of recognized glosses/words
let currentSentence  = "";     // Translated English sentence
let recentPrediction = [];     // Rolling window for frame stability
let lastAccepted     = null;
let lastSignTime     = 0;
let isTranslating    = false;
let translationTimer = null;

let frameCount = 0;
let fpsTimer   = Date.now();

const INTERVAL_MS    = 130;    // ~7.5 FPS prediction rate
const STABLE_NEEDED  = 2;      // Frame consensus needed to accept sign
const BUFFER_SIZE    = 4;      // Rolling window size
const MIN_CONFIDENCE = 0.58;
const SIGN_COOLDOWN  = 800;    // ms between registering the same consecutive sign

/* ── API status ─────────────────────────────────────────────────── */
async function loadStatus() {
    try {
        const data = await fetch("/api/status").then(r => r.json());
        
        // Model status
        if (data.model_ready) {
            modelStatus.classList.add("ready");
            modelStatus.innerHTML = '<span class="status-dot"></span>Model loaded';
        } else {
            modelStatus.classList.remove("ready");
            modelStatus.innerHTML = '<span class="status-dot"></span>Model not trained';
        }

        // AI Translation status
        if (data.gemini_enabled) {
            aiStatus.classList.add("gemini-active");
            aiStatusText.textContent = "AI: Gemini Active";
            translationSourceBadge.textContent = "Gemini AI";
            translationSourceBadge.classList.add("gemini");
        } else {
            aiStatus.classList.remove("gemini-active");
            aiStatusText.textContent = "AI: Smart NLP Engine";
            translationSourceBadge.textContent = "Smart NLP Engine";
            translationSourceBadge.classList.remove("gemini");
        }

        // Active Classes Grid
        classGrid.innerHTML = "";
        (data.classes || []).forEach(label => {
            const el = document.createElement("div");
            el.className = "class-item";
            el.textContent = label.toUpperCase();
            classGrid.appendChild(el);
        });
        classCount.textContent = `${(data.classes || []).length} signs`;
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
    setMessage("Live translation active — position hand in camera.");
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
        const resp = await fetch("/api/recognize-sequence", { method: "POST", body: form });
        const data = await resp.json();

        if (!resp.ok || !data.success) {
            setMessage(data.error || "Prediction error.");
            return;
        }

        const frameResult = data.frame_prediction;

        if (!frameResult.hand_detected) {
            resetPrediction();
            setMessage("No hand detected — hold hand up to the camera.");
            return;
        }

        updatePrediction(frameResult.prediction, frameResult.confidence);

        // Stabilize individual signs and add to gloss sequence
        if (frameResult.confidence >= MIN_CONFIDENCE && frameResult.prediction) {
            stabilize(frameResult.prediction);
        } else {
            recentPrediction = [];
        }

    } catch { /* network blip — ignore */ }
}

/* ── Stability buffer & Sequence builder ─────────────────────────── */
function stabilize(label) {
    recentPrediction.push(label);
    if (recentPrediction.length > BUFFER_SIZE) recentPrediction.shift();

    const counts = {};
    recentPrediction.forEach(x => counts[x] = (counts[x] || 0) + 1);
    const [top, cnt] = Object.entries(counts).sort((a,b) => b[1]-a[1])[0];

    const now = Date.now();
    const isNewSign = top !== lastAccepted;
    const cooldownElapsed = (now - lastSignTime) > SIGN_COOLDOWN;

    if (cnt >= STABLE_NEEDED && (isNewSign || cooldownElapsed)) {
        lastAccepted = top;
        lastSignTime = now;
        
        // Add recognized sign gloss to sequence
        addGloss(top);

        /* brief visual feedback flash */
        prediction.classList.add("flash");
        setTimeout(() => prediction.classList.remove("flash"), 300);
    }
}

function addGloss(gloss) {
    const isLetter = gloss.length === 1 && /[A-Z]/i.test(gloss);
    
    // Check if the previous token was a single letter - if user is fingerspelling, group consecutive letters
    if (isLetter && detectedGlosses.length > 0) {
        const last = detectedGlosses[detectedGlosses.length - 1];
        // If last element is also single letter, keep both as separate chips or together
    }
    
    detectedGlosses.push(gloss);
    renderGlosses();

    // Trigger debounced AI translation automatically
    scheduleTranslation();
}

function renderGlosses() {
    glossContainer.innerHTML = "";
    if (detectedGlosses.length === 0) {
        glossContainer.innerHTML = '<span class="placeholder-text">Signed words and letters will collect here…</span>';
        return;
    }

    detectedGlosses.forEach((gloss, idx) => {
        const chip = document.createElement("span");
        chip.className = "gloss-chip";
        const isLetter = gloss.length === 1 && /[A-Z]/i.test(gloss);
        if (isLetter) {
            chip.classList.add("letter-chip");
        }
        chip.textContent = gloss.toUpperCase();
        chip.title = isLetter ? `Letter '${gloss}' (Click to remove)` : `Sign '${gloss}' (Click to remove)`;
        chip.addEventListener("click", () => {
            detectedGlosses.splice(idx, 1);
            renderGlosses();
            scheduleTranslation();
        });
        glossContainer.appendChild(chip);
    });
}

function scheduleTranslation() {
    clearTimeout(translationTimer);
    translationTimer = setTimeout(() => {
        fetchTranslation(detectedGlosses);
    }, 400); // 400ms debounce
}

/* ── AI Translation API ──────────────────────────────────────────── */
async function fetchTranslation(sequence, manual = false) {
    if (!sequence || sequence.length === 0) {
        translatedText.textContent = "Start signing to generate full sentences…";
        currentSentence = "";
        return;
    }

    if (isTranslating && !manual) return;
    isTranslating = true;

    if (manual) {
        btnTranslate.disabled = true;
        btnTranslate.textContent = "Translating…";
    }

    try {
        const resp = await fetch("/api/translate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sequence: sequence })
        });
        const data = await resp.json();

        if (data.success && data.translation) {
            currentSentence = data.translation;
            translatedText.textContent = currentSentence;
            
            // Update source badge
            if (data.source === "gemini") {
                translationSourceBadge.textContent = "Gemini AI";
                translationSourceBadge.classList.add("gemini");
            } else {
                translationSourceBadge.textContent = "Smart NLP Engine";
                translationSourceBadge.classList.remove("gemini");
            }
        }
    } catch (err) {
        console.error("Translation error:", err);
    } finally {
        isTranslating = false;
        if (manual) {
            btnTranslate.disabled = false;
            btnTranslate.textContent = "✨ Translate Now";
        }
    }
}

/* ── UI Helpers ──────────────────────────────────────────────────── */
function updatePrediction(label, conf) {
    const text = label ? label.toUpperCase() : "—";
    prediction.textContent = text;
    
    // Scale font size dynamically according to word length so box never resizes
    prediction.classList.remove("long-word", "medium-word");
    if (text.length > 7) {
        prediction.classList.add("long-word");
    } else if (text.length > 3) {
        prediction.classList.add("medium-word");
    }

    const pct = Math.round(conf * 100);
    confidenceVal.textContent = `${pct}%`;
    confidenceBar.style.width = `${pct}%`;
    confidenceBar.style.background =
        pct >= 85 ? "var(--accent)" :
        pct >= 65 ? "#f5d020"       : "#ff6b6b";
    setMessage("Detecting sign…");

    /* highlight matching class badge */
    document.querySelectorAll(".class-item").forEach(el => {
        el.classList.toggle("active", el.textContent === label?.toUpperCase());
    });
}

function resetPrediction() {
    prediction.textContent    = "—";
    prediction.classList.remove("long-word", "medium-word");
    confidenceVal.textContent = "—";
    confidenceBar.style.width = "0";
    recentPrediction          = [];
    document.querySelectorAll(".class-item.active")
        .forEach(el => el.classList.remove("active"));
}

function setMessage(msg) {
    message.textContent = msg;
}

/* ── Controls Wiring ─────────────────────────────────────────────── */
btnTranslate.addEventListener("click", () => {
    fetchTranslation(detectedGlosses, true);
});

btnSpeak.addEventListener("click", () => {
    if (!currentSentence) return;
    if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel(); // Stop any pending speech
        const utterance = new SpeechSynthesisUtterance(currentSentence);
        utterance.rate = 0.95;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
    } else {
        alert("Speech synthesis is not supported in this browser.");
    }
});

btnCopy.addEventListener("click", async () => {
    if (!currentSentence) return;
    try {
        await navigator.clipboard.writeText(currentSentence);
        const originalText = btnCopy.textContent;
        btnCopy.textContent = "✓ Copied!";
        setTimeout(() => btnCopy.textContent = originalText, 1500);
    } catch (err) {
        console.error("Could not copy:", err);
    }
});

btnClear.addEventListener("click", () => {
    detectedGlosses = [];
    currentSentence = "";
    lastAccepted = null;
    recentPrediction = [];
    renderGlosses();
    translatedText.textContent = "Start signing to generate full sentences…";
    fetch("/api/clear-sequence", { method: "POST" });
});

btnBackspace.addEventListener("click", () => {
    if (detectedGlosses.length > 0) {
        detectedGlosses.pop();
        lastAccepted = null;
        recentPrediction = [];
        renderGlosses();
        scheduleTranslation();
    }
});

btnSpace.addEventListener("click", () => {
    // Inserts a pause / forces immediate translation
    fetchTranslation(detectedGlosses, true);
});

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click",  stopCamera);

/* ── Init ────────────────────────────────────────────────────────── */
loadStatus();
