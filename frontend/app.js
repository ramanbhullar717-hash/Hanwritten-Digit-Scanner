/**
 * app.js - Client-Side Controller for OmniScan AI
 * Manages:
 * 1. High-DPI HTML5 Canvas (Drawing, Touch, Smoothing)
 * 2. REST API Integration (/api/predict, /api/benchmark, /api/explain/knn, /api/scan/document)
 * 3. Real-Time Telemetry & Probability Bar Rendering
 * 4. Multi-Digit Document Upload & Contour Chip Rendering
 */

document.addEventListener("DOMContentLoaded", () => {
  // --- Canvas Setup ---
  const canvas = document.getElementById("drawingCanvas");
  const ctx = canvas.getContext("2d");
  const brushSizeInput = document.getElementById("brushSize");
  const brushVal = document.getElementById("brushValue");
  const clearBtn = document.getElementById("clearBtn");
  const predictBtn = document.getElementById("predictBtn");
  const modelSelect = document.getElementById("modelSelect");

  // Telemetry elements
  const predictedDigit = document.getElementById("predictedDigit");
  const confidencePercent = document.getElementById("confidencePercent");
  const confidenceBar = document.getElementById("confidenceBar");
  const activeEngineBadge = document.getElementById("activeEngineBadge");
  const latencyBadge = document.getElementById("latencyBadge");
  const tensorPreview = document.getElementById("tensorPreview");
  const comVal = document.getElementById("comVal");
  const probGrid = document.getElementById("probGrid");
  const knnPrototypes = document.getElementById("knnPrototypes");
  const clusterCentroidImg = document.getElementById("clusterCentroidImg");
  const clusterStyleTitle = document.getElementById("clusterStyleTitle");
  const clusterDist = document.getElementById("clusterDist");

  // Tab navigation
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  // Multi-Digit Document elements
  const docDropzone = document.getElementById("docDropzone");
  const docFileInput = document.getElementById("docFileInput");
  const sampleDocBtn = document.getElementById("sampleDocBtn");
  const docPreviewWrapper = document.getElementById("docPreviewWrapper");
  const annotatedDocPreview = document.getElementById("annotatedDocPreview");
  const docSequenceValue = document.getElementById("docSequenceValue");
  const docDigitsCount = document.getElementById("docDigitsCount");
  const docAvgConf = document.getElementById("docAvgConf");
  const segmentedChips = document.getElementById("segmentedChips");
  const docLatencyBadge = document.getElementById("docLatencyBadge");

  // Benchmark elements
  const runArenaBtn = document.getElementById("runArenaBtn");
  const consensusDigit = document.getElementById("consensusDigit");
  const lrDigit = document.getElementById("lrDigit");
  const lrConf = document.getElementById("lrConf");
  const lrLatency = document.getElementById("lrLatency");
  const knnDigit = document.getElementById("knnDigit");
  const knnConf = document.getElementById("knnConf");
  const knnLatency = document.getElementById("knnLatency");
  const annDigit = document.getElementById("annDigit");
  const annConf = document.getElementById("annConf");
  const annLatency = document.getElementById("annLatency");
  const cnnDigit = document.getElementById("cnnDigit");
  const cnnConf = document.getElementById("cnnConf");
  const cnnLatency = document.getElementById("cnnLatency");

  let isDrawing = false;
  let hasDrawn = false;
  let brushSize = parseInt(brushSizeInput.value, 10);

  // Initialize Canvas
  function resetCanvas() {
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    hasDrawn = false;
  }
  resetCanvas();

  // Brush controls
  brushSizeInput.addEventListener("input", (e) => {
    brushSize = parseInt(e.target.value, 10);
    brushVal.textContent = `${brushSize}px`;
  });

  clearBtn.addEventListener("click", () => {
    resetCanvas();
    predictedDigit.textContent = "—";
    confidencePercent.textContent = "0.0%";
    confidenceBar.style.width = "0%";
    latencyBadge.textContent = "0.0 ms";
    tensorPreview.src = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
    comVal.textContent = "(13.5, 13.5)";
    renderProbabilities(new Array(10).fill(0));
    knnPrototypes.innerHTML = `<p class="placeholder-text">Draw a digit and click Scan to inspect closest training prototypes.</p>`;
    clusterStyleTitle.textContent = "Archetype #—";
    clusterDist.textContent = "Distance: —";
  });

  // Drawing event listeners (Mouse & Touch)
  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: (clientX - rect.left) * (canvas.width / rect.width),
      y: (clientY - rect.top) * (canvas.height / rect.height)
    };
  }

  function startDraw(e) {
    e.preventDefault();
    isDrawing = true;
    hasDrawn = true;
    const pos = getPos(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  }

  function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const pos = getPos(e);
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = brushSize;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
  }

  function stopDraw(e) {
    if (!isDrawing) return;
    isDrawing = false;
    ctx.closePath();
  }

  canvas.addEventListener("mousedown", startDraw);
  canvas.addEventListener("mousemove", draw);
  canvas.addEventListener("mouseup", stopDraw);
  canvas.addEventListener("mouseleave", stopDraw);

  canvas.addEventListener("touchstart", startDraw, { passive: false });
  canvas.addEventListener("touchmove", draw, { passive: false });
  canvas.addEventListener("touchend", stopDraw);

  // Tab Switching
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const targetPane = document.getElementById(btn.dataset.tab);
      if (targetPane) targetPane.classList.add("active");
    });
  });

  // Render 0-9 Probability Spectrum
  function renderProbabilities(probs) {
    probGrid.innerHTML = "";
    const maxVal = Math.max(...probs);

    for (let i = 0; i < 10; i++) {
      const p = probs[i] || 0;
      const isHighest = p === maxVal && p > 0;
      const item = document.createElement("div");
      item.className = `prob-item ${isHighest ? "highest" : ""}`;
      item.innerHTML = `
        <div class="prob-item-header">
          <span>Digit ${i}</span>
          <span>${(p * 100).toFixed(1)}%</span>
        </div>
        <div class="mini-bar-bg">
          <div class="mini-bar-fill" style="width: ${(p * 100).toFixed(1)}%"></div>
        </div>
      `;
      probGrid.appendChild(item);
    }
  }
  renderProbabilities(new Array(10).fill(0));

  // --- API 1: Predict Single Digit ---
  async function scanCurrentCanvas() {
    if (!hasDrawn) return;
    const b64 = canvas.toDataURL("image/png");
    const activeModel = modelSelect.value;

    try {
      predictBtn.disabled = true;
      predictBtn.innerHTML = `Scanning...`;

      // 1. Predict
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: b64, model: activeModel })
      });
      const data = await res.json();

      if (data.status === "success") {
        predictedDigit.textContent = data.digit;
        const confPercent = (data.confidence * 100).toFixed(1);
        confidencePercent.textContent = `${confPercent}%`;
        confidenceBar.style.width = `${confPercent}%`;
        latencyBadge.textContent = `${data.total_latency_ms} ms`;
        activeEngineBadge.textContent = data.model_used;
        tensorPreview.src = data.preprocessed_image;
        comVal.textContent = `(${data.center_of_mass[0]}, ${data.center_of_mass[1]})`;
        renderProbabilities(data.probabilities);

        // 2. Fetch KNN Explainability Prototypes
        fetchKNNExplains(b64);

        // 3. Fetch K-Means Cluster Archetype
        fetchClusterStyle(b64);
      }
    } catch (err) {
      console.error("Inference Error:", err);
    } finally {
      predictBtn.disabled = false;
      predictBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        Scan Digit
      `;
    }
  }
  predictBtn.addEventListener("click", scanCurrentCanvas);

  // --- API 2: Explainable AI (KNN Top-5) ---
  async function fetchKNNExplains(b64) {
    try {
      const res = await fetch("/api/explain/knn", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: b64, k: 5 })
      });
      const data = await res.json();

      if (data.status === "success" && data.neighbors.length > 0) {
        knnPrototypes.innerHTML = "";
        data.neighbors.forEach((n) => {
          const card = document.createElement("div");
          card.className = "prototype-card";
          card.innerHTML = `
            <img src="${n.image_base64}" alt="Neighbor">
            <span class="proto-label">#${n.label}</span>
            <span class="proto-dist">d=${n.distance}</span>
          `;
          knnPrototypes.appendChild(card);
        });
      }
    } catch (e) {
      console.error("KNN Explain Error:", e);
    }
  }

  // --- API 3: K-Means Handwriting Style ---
  async function fetchClusterStyle(b64) {
    try {
      const res = await fetch("/api/cluster/inspect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: b64 })
      });
      const data = await res.json();

      if (data.status === "success") {
        const diag = data.cluster_diagnostics;
        clusterCentroidImg.src = diag.centroid_preview_base64;
        clusterStyleTitle.textContent = diag.style_description;
        clusterDist.textContent = `Centroid Dist: ${diag.distance_to_centroid}`;
      }
    } catch (e) {
      console.error("Cluster Error:", e);
    }
  }

  // --- API 4: Benchmark Arena ---
  runArenaBtn.addEventListener("click", async () => {
    if (!hasDrawn) {
      alert("Please draw a digit on the canvas first!");
      return;
    }
    const b64 = canvas.toDataURL("image/png");

    try {
      runArenaBtn.disabled = true;
      runArenaBtn.textContent = "Running Benchmark...";

      const res = await fetch("/api/benchmark", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: b64 })
      });
      const data = await res.json();

      if (data.status === "success") {
        consensusDigit.textContent = `Digit ${data.consensus_digit}`;
        const b = data.benchmark_results;

        // Logistic
        lrDigit.textContent = b.logistic_regression.digit;
        lrConf.textContent = `Conf: ${(b.logistic_regression.confidence * 100).toFixed(1)}%`;
        lrLatency.textContent = `${b.logistic_regression.latency_ms.toFixed(1)} ms`;

        // KNN
        knnDigit.textContent = b.knn.digit;
        knnConf.textContent = `Conf: ${(b.knn.confidence * 100).toFixed(1)}%`;
        knnLatency.textContent = `${b.knn.latency_ms.toFixed(1)} ms`;

        // ANN
        annDigit.textContent = b.ann.digit;
        annConf.textContent = `Conf: ${(b.ann.confidence * 100).toFixed(1)}%`;
        annLatency.textContent = `${b.ann.latency_ms.toFixed(1)} ms`;

        // CNN
        cnnDigit.textContent = b.cnn.digit;
        cnnConf.textContent = `Conf: ${(b.cnn.confidence * 100).toFixed(1)}%`;
        cnnLatency.textContent = `${b.cnn.latency_ms.toFixed(1)} ms`;
      }
    } catch (e) {
      console.error("Benchmark Error:", e);
    } finally {
      runArenaBtn.disabled = false;
      runArenaBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        Run Benchmark on Current Sketch
      `;
    }
  });

  // --- API 5: Multi-Digit Document Reader ---
  docDropzone.addEventListener("click", () => docFileInput.click());

  docFileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      const reader = new FileReader();
      reader.onload = (evt) => processDocumentScan(evt.target.result);
      reader.readAsDataURL(e.target.files[0]);
    }
  });

  // Generate Synthetic "7042" Sequence for instant testing
  sampleDocBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 400;
    tempCanvas.height = 120;
    const tCtx = tempCanvas.getContext("2d");
    tCtx.fillStyle = "#FFFFFF";
    tCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
    tCtx.fillStyle = "#000000";
    tCtx.font = "bold 65px monospace";
    tCtx.fillText("7 0 4 2", 40, 85);
    processDocumentScan(tempCanvas.toDataURL("image/png"));
  });

  async function processDocumentScan(b64) {
    try {
      docSequenceValue.textContent = "Scanning...";
      const res = await fetch("/api/scan/document", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: b64, model: "cnn" })
      });
      const data = await res.json();

      if (data.status === "success") {
        docPreviewWrapper.style.display = "block";
        annotatedDocPreview.src = data.annotated_preview;
        docSequenceValue.textContent = data.recognized_sequence;
        docDigitsCount.textContent = data.total_digits;
        docAvgConf.textContent = `${(data.average_confidence * 100).toFixed(1)}%`;
        docLatencyBadge.textContent = `${data.total_latency_ms} ms`;

        // Render chips
        segmentedChips.innerHTML = "";
        data.digit_details.forEach((d) => {
          const chip = document.createElement("div");
          chip.className = "digit-chip";
          chip.innerHTML = `
            <span class="chip-idx">Digit #${d.index + 1}</span>
            <span class="chip-val">${d.digit}</span>
            <span class="chip-conf">${(d.confidence * 100).toFixed(1)}%</span>
          `;
          segmentedChips.appendChild(chip);
        });
      } else {
        docSequenceValue.textContent = "No Digits Found";
      }
    } catch (e) {
      console.error("Document Scan Error:", e);
      docSequenceValue.textContent = "Error Scanning";
    }
  }
});
