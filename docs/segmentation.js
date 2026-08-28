/**
 * Client-side port of the Python segmentation pipeline (maxflow/graph_builder.py,
 * maxflow/edmonds_karp.py, segmentation/segmenter.py) so the GitHub Pages site can
 * run real interactive segmentation with no backend server.
 *
 * Algorithm section mirrors the Python implementation function-for-function so
 * behavior stays consistent between the two. UI section below handles the
 * upload / canvas-scribble / segment / download interaction.
 */

(() => {
  "use strict";

  // ---- maxflow/graph_builder.py equivalent ----

  const SOURCE = "S";
  const SINK = "T";
  const UNMARKED = 0;
  const FOREGROUND = 1;
  const BACKGROUND = 2;

  function toGrayscale(rgba, pixelCount) {
    const gray = new Float64Array(pixelCount);
    for (let i = 0; i < pixelCount; i++) {
      const r = rgba[i * 4];
      const g = rgba[i * 4 + 1];
      const b = rgba[i * 4 + 2];
      gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    }
    return gray;
  }

  function estimateBeta(gray, width, height) {
    let sumSquares = 0;
    let count = 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width - 1; x++) {
        const diff = gray[y * width + x] - gray[y * width + x + 1];
        sumSquares += diff * diff;
        count++;
      }
    }
    for (let y = 0; y < height - 1; y++) {
      for (let x = 0; x < width; x++) {
        const diff = gray[y * width + x] - gray[(y + 1) * width + x];
        sumSquares += diff * diff;
        count++;
      }
    }
    const meanSquare = count > 0 ? sumSquares / count : 0;
    return meanSquare === 0 ? 1.0 : 1.0 / (2.0 * meanSquare);
  }

  // Graphs are Map<node, Map<node, capacity>>. Plain objects would force V8
  // into slow "dictionary mode" for a dynamically-built key set this large;
  // Map is what V8 actually optimizes for this access pattern.
  function getOrCreateAdjacency(graph, node) {
    let adjacency = graph.get(node);
    if (!adjacency) {
      adjacency = new Map();
      graph.set(node, adjacency);
    }
    return adjacency;
  }

  function buildNLinks(gray, width, height, nLinkWeight) {
    const beta = estimateBeta(gray, width, height);
    const graph = new Map();
    const neighborWeightSum = new Float64Array(width * height);

    function addEdge(p, q, weight) {
      getOrCreateAdjacency(graph, p).set(q, weight);
      getOrCreateAdjacency(graph, q).set(p, weight);
      neighborWeightSum[p] += weight;
      neighborWeightSum[q] += weight;
    }

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const p = y * width + x;
        if (x + 1 < width) {
          const q = y * width + (x + 1);
          const diff = gray[p] - gray[q];
          addEdge(p, q, nLinkWeight * Math.exp(-beta * diff * diff));
        }
        if (y + 1 < height) {
          const q = (y + 1) * width + x;
          const diff = gray[p] - gray[q];
          addEdge(p, q, nLinkWeight * Math.exp(-beta * diff * diff));
        }
      }
    }
    return { graph, neighborWeightSum };
  }

  function colorHistogram(gray, selected, numBins) {
    const histogram = new Float64Array(numBins).fill(1.0); // Laplace smoothing
    let total = numBins;
    const binWidth = 256.0 / numBins;
    for (let i = 0; i < gray.length; i++) {
      if (!selected[i]) continue;
      let bin = Math.floor(gray[i] / binWidth);
      bin = Math.max(0, Math.min(numBins - 1, bin));
      histogram[bin] += 1;
      total += 1;
    }
    for (let i = 0; i < numBins; i++) histogram[i] /= total;
    return histogram;
  }

  function binIndex(value, numBins) {
    const binWidth = 256.0 / numBins;
    return Math.max(0, Math.min(numBins - 1, Math.floor(value / binWidth)));
  }

  function buildGraph(gray, width, height, scribbleMask, nLinkWeight, numBins) {
    const { graph, neighborWeightSum } = buildNLinks(gray, width, height, nLinkWeight);

    const fgSelected = new Uint8Array(width * height);
    const bgSelected = new Uint8Array(width * height);
    let hasForeground = false;
    let hasBackground = false;
    for (let i = 0; i < scribbleMask.length; i++) {
      if (scribbleMask[i] === FOREGROUND) {
        fgSelected[i] = 1;
        hasForeground = true;
      } else if (scribbleMask[i] === BACKGROUND) {
        bgSelected[i] = 1;
        hasBackground = true;
      }
    }
    if (!hasForeground || !hasBackground) {
      throw new Error("Scribble at least one foreground (red) and one background (blue) area first.");
    }

    const fgHistogram = colorHistogram(gray, fgSelected, numBins);
    const bgHistogram = colorHistogram(gray, bgSelected, numBins);

    let maxNeighborSum = 0;
    for (let i = 0; i < neighborWeightSum.length; i++) {
      if (neighborWeightSum[i] > maxNeighborSum) maxNeighborSum = neighborWeightSum[i];
    }
    const hardConstraint = 1.0 + maxNeighborSum;

    const sourceAdjacency = getOrCreateAdjacency(graph, SOURCE);
    for (let i = 0; i < width * height; i++) {
      const label = scribbleMask[i];
      let sourceCapacity;
      let sinkCapacity;

      if (label === FOREGROUND) {
        sourceCapacity = hardConstraint;
        sinkCapacity = 0;
      } else if (label === BACKGROUND) {
        sourceCapacity = 0;
        sinkCapacity = hardConstraint;
      } else {
        const bin = binIndex(gray[i], numBins);
        sourceCapacity = -Math.log(bgHistogram[bin]);
        sinkCapacity = -Math.log(fgHistogram[bin]);
      }

      if (sourceCapacity > 0) sourceAdjacency.set(i, sourceCapacity);
      if (sinkCapacity > 0) getOrCreateAdjacency(graph, i).set(SINK, sinkCapacity);
    }

    return graph;
  }

  // ---- maxflow/edmonds_karp.py equivalent ----

  function buildResidualGraph(graph) {
    const residual = new Map();
    for (const [u, neighbors] of graph) {
      const uAdjacency = getOrCreateAdjacency(residual, u);
      for (const [v, capacity] of neighbors) {
        uAdjacency.set(v, (uAdjacency.get(v) || 0) + capacity);
        const vAdjacency = getOrCreateAdjacency(residual, v);
        if (!vAdjacency.has(u)) vAdjacency.set(u, 0);
      }
    }
    return residual;
  }

  function bfsFindAugmentingPath(residual, source, sink) {
    const parent = new Map([[source, source]]);
    const queue = [source];
    let head = 0;
    while (head < queue.length) {
      const u = queue[head++];
      if (u === sink) break;
      const neighbors = residual.get(u);
      if (!neighbors) continue;
      for (const [v, capacity] of neighbors) {
        if (capacity > 0 && !parent.has(v)) {
          parent.set(v, u);
          queue.push(v);
        }
      }
    }
    if (!parent.has(sink)) return null;
    const path = [sink];
    while (path[path.length - 1] !== source) {
      path.push(parent.get(path[path.length - 1]));
    }
    path.reverse();
    return path;
  }

  function pathBottleneck(residual, path) {
    let bottleneck = Infinity;
    for (let i = 0; i < path.length - 1; i++) {
      bottleneck = Math.min(bottleneck, residual.get(path[i]).get(path[i + 1]));
    }
    return bottleneck;
  }

  function augmentAlongPath(residual, path, bottleneck) {
    for (let i = 0; i < path.length - 1; i++) {
      const u = path[i];
      const v = path[i + 1];
      const uAdjacency = residual.get(u);
      uAdjacency.set(v, uAdjacency.get(v) - bottleneck);
      const vAdjacency = getOrCreateAdjacency(residual, v);
      vAdjacency.set(u, (vAdjacency.get(u) || 0) + bottleneck);
    }
  }

  function edmondsKarp(graph, source, sink) {
    const residual = buildResidualGraph(graph);
    let maxFlow = 0;
    while (true) {
      const path = bfsFindAugmentingPath(residual, source, sink);
      if (!path) break;
      const bottleneck = pathBottleneck(residual, path);
      augmentAlongPath(residual, path, bottleneck);
      maxFlow += bottleneck;
    }
    return { maxFlow, residual };
  }

  function findReachableFromSource(residual, source) {
    const visited = new Set([source]);
    const queue = [source];
    let head = 0;
    while (head < queue.length) {
      const u = queue[head++];
      const neighbors = residual.get(u);
      if (!neighbors) continue;
      for (const [v, capacity] of neighbors) {
        if (capacity > 0 && !visited.has(v)) {
          visited.add(v);
          queue.push(v);
        }
      }
    }
    return visited;
  }

  // ---- segmentation/segmenter.py equivalent ----

  function segment(gray, width, height, scribbleMask, nLinkWeight = 50.0, numBins = 32) {
    const graph = buildGraph(gray, width, height, scribbleMask, nLinkWeight, numBins);
    const { residual } = edmondsKarp(graph, SOURCE, SINK);
    const reachable = findReachableFromSource(residual, SOURCE);

    const mask = new Uint8Array(width * height);
    for (const node of reachable) {
      if (node !== SOURCE) mask[node] = 1;
    }
    return mask;
  }

  // ---- UI ----

  const PROCESSING_MAX_DIM = 30;
  const DISPLAY_MAX_DIM = 320;

  const fileInput = document.getElementById("tryFileInput");
  const scribbleControls = document.getElementById("scribbleControls");
  const canvasWrap = document.getElementById("canvasWrap");
  const tryCanvas = document.getElementById("tryCanvas");
  const tryActions = document.getElementById("tryActions");
  const segmentBtn = document.getElementById("segmentBtn");
  const clearBtn = document.getElementById("clearScribblesBtn");
  const tryStatus = document.getElementById("tryStatus");
  const tryResults = document.getElementById("tryResults");
  const originalCanvas = document.getElementById("originalResultCanvas");
  const scribbleCanvas = document.getElementById("scribbleResultCanvas");
  const resultCanvas = document.getElementById("resultCanvas");
  const downloadBtn = document.getElementById("downloadBtn");
  const brushSizeInput = document.getElementById("brushSize");

  if (!fileInput) return; // section not present on this page

  const ctx = tryCanvas.getContext("2d");
  let procWidth = 0;
  let procHeight = 0;
  let cellSize = 1;
  let grayscale = null;
  let cleanImageBitmap = null; // small offscreen canvas holding the clean (unscribbled) image
  let scribbleMask = null;
  let isDrawing = false;
  let lastPoint = null;

  function currentScribbleLabel() {
    const checked = document.querySelector('input[name="scribbleMode"]:checked');
    return checked && checked.value === "bg" ? BACKGROUND : FOREGROUND;
  }

  function brushRadiusInProcPixels() {
    const displayPx = Number(brushSizeInput.value) || 6;
    return Math.max(1, Math.round(displayPx / cellSize / 2));
  }

  function markBrush(px, py, label, radius) {
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        if (dx * dx + dy * dy > radius * radius) continue;
        const x = px + dx;
        const y = py + dy;
        if (x >= 0 && x < procWidth && y >= 0 && y < procHeight) {
          scribbleMask[y * procWidth + x] = label;
        }
      }
    }
  }

  function markLine(x0, y0, x1, y1, label, radius) {
    const dx = x1 - x0;
    const dy = y1 - y0;
    const steps = Math.max(Math.abs(dx), Math.abs(dy), 1);
    for (let i = 0; i <= steps; i++) {
      const x = Math.round(x0 + (dx * i) / steps);
      const y = Math.round(y0 + (dy * i) / steps);
      markBrush(x, y, label, radius);
    }
  }

  function redrawTryCanvas() {
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(cleanImageBitmap, 0, 0, tryCanvas.width, tryCanvas.height);
    for (let y = 0; y < procHeight; y++) {
      for (let x = 0; x < procWidth; x++) {
        const label = scribbleMask[y * procWidth + x];
        if (label === UNMARKED) continue;
        ctx.fillStyle = label === FOREGROUND ? "#ef4444" : "#3b82f6";
        ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
      }
    }
  }

  function getProcCoords(evt) {
    const rect = tryCanvas.getBoundingClientRect();
    const cssX = evt.clientX - rect.left;
    const cssY = evt.clientY - rect.top;
    const x = (cssX * tryCanvas.width) / rect.width;
    const y = (cssY * tryCanvas.height) / rect.height;
    return [Math.floor(x / cellSize), Math.floor(y / cellSize)];
  }

  function setStatus(text, isError) {
    tryStatus.textContent = text;
    tryStatus.classList.toggle("try-status-error", Boolean(isError));
  }

  function loadImageFile(file) {
    const img = new Image();
    img.onload = () => {
      const scale = PROCESSING_MAX_DIM / Math.max(img.width, img.height);
      procWidth = Math.max(1, Math.round(img.width * scale));
      procHeight = Math.max(1, Math.round(img.height * scale));

      const displayScale = DISPLAY_MAX_DIM / Math.max(procWidth, procHeight);
      const displayWidth = Math.round(procWidth * displayScale);
      const displayHeight = Math.round(procHeight * displayScale);
      cellSize = displayWidth / procWidth;

      const offscreen = document.createElement("canvas");
      offscreen.width = procWidth;
      offscreen.height = procHeight;
      const offCtx = offscreen.getContext("2d");
      offCtx.drawImage(img, 0, 0, procWidth, procHeight);
      cleanImageBitmap = offscreen;

      const imageData = offCtx.getImageData(0, 0, procWidth, procHeight);
      grayscale = toGrayscale(imageData.data, procWidth * procHeight);

      scribbleMask = new Uint8Array(procWidth * procHeight);

      tryCanvas.width = displayWidth;
      tryCanvas.height = displayHeight;
      tryCanvas.style.aspectRatio = `${displayWidth} / ${displayHeight}`;
      redrawTryCanvas();

      scribbleControls.hidden = false;
      canvasWrap.hidden = false;
      tryActions.hidden = false;
      tryResults.hidden = true;
      setStatus(`Processing at ${procWidth}x${procHeight}.`, false);
    };
    img.onerror = () => setStatus("Couldn't load that file as an image.", true);
    img.src = URL.createObjectURL(file);
  }

  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0];
    if (file) loadImageFile(file);
  });

  tryCanvas.style.touchAction = "none";

  tryCanvas.addEventListener("pointerdown", (evt) => {
    isDrawing = true;
    tryCanvas.setPointerCapture(evt.pointerId);
    const [px, py] = getProcCoords(evt);
    markBrush(px, py, currentScribbleLabel(), brushRadiusInProcPixels());
    lastPoint = [px, py];
    redrawTryCanvas();
  });

  tryCanvas.addEventListener("pointermove", (evt) => {
    if (!isDrawing) return;
    const [px, py] = getProcCoords(evt);
    const radius = brushRadiusInProcPixels();
    if (lastPoint) {
      markLine(lastPoint[0], lastPoint[1], px, py, currentScribbleLabel(), radius);
    } else {
      markBrush(px, py, currentScribbleLabel(), radius);
    }
    lastPoint = [px, py];
    redrawTryCanvas();
  });

  function stopDrawing() {
    isDrawing = false;
    lastPoint = null;
  }
  tryCanvas.addEventListener("pointerup", stopDrawing);
  tryCanvas.addEventListener("pointerleave", stopDrawing);
  tryCanvas.addEventListener("pointercancel", stopDrawing);

  clearBtn.addEventListener("click", () => {
    if (!scribbleMask) return;
    scribbleMask.fill(UNMARKED);
    redrawTryCanvas();
    setStatus("", false);
    tryResults.hidden = true;
  });

  function drawCutout(destCanvas, mask) {
    destCanvas.width = tryCanvas.width;
    destCanvas.height = tryCanvas.height;
    const destCtx = destCanvas.getContext("2d");
    destCtx.imageSmoothingEnabled = false;

    const cutoutOffscreen = document.createElement("canvas");
    cutoutOffscreen.width = procWidth;
    cutoutOffscreen.height = procHeight;
    const cutoutCtx = cutoutOffscreen.getContext("2d");
    cutoutCtx.drawImage(cleanImageBitmap, 0, 0);
    const imageData = cutoutCtx.getImageData(0, 0, procWidth, procHeight);
    for (let i = 0; i < procWidth * procHeight; i++) {
      if (!mask[i]) {
        imageData.data[i * 4] = imageData.data[i * 4] >> 2;
        imageData.data[i * 4 + 1] = imageData.data[i * 4 + 1] >> 2;
        imageData.data[i * 4 + 2] = imageData.data[i * 4 + 2] >> 2;
      }
    }
    cutoutCtx.putImageData(imageData, 0, 0);
    destCtx.drawImage(cutoutOffscreen, 0, 0, destCanvas.width, destCanvas.height);
  }

  segmentBtn.addEventListener("click", () => {
    if (!scribbleMask) return;
    segmentBtn.disabled = true;
    setStatus("Segmenting…", false);

    // Defer so the "Segmenting…" status actually paints before the
    // synchronous, blocking Edmonds-Karp computation runs.
    setTimeout(() => {
      const start = performance.now();
      try {
        const mask = segment(grayscale, procWidth, procHeight, scribbleMask);
        const elapsedSeconds = (performance.now() - start) / 1000;

        originalCanvas.width = tryCanvas.width;
        originalCanvas.height = tryCanvas.height;
        const originalCtx = originalCanvas.getContext("2d");
        originalCtx.imageSmoothingEnabled = false;
        originalCtx.drawImage(cleanImageBitmap, 0, 0, originalCanvas.width, originalCanvas.height);

        scribbleCanvas.width = tryCanvas.width;
        scribbleCanvas.height = tryCanvas.height;
        scribbleCanvas.getContext("2d").drawImage(tryCanvas, 0, 0);

        drawCutout(resultCanvas, mask);

        tryResults.hidden = false;
        setStatus(`Segmentation took ${elapsedSeconds.toFixed(2)} seconds.`, false);
      } catch (err) {
        setStatus(err.message || "Segmentation failed.", true);
      } finally {
        segmentBtn.disabled = false;
      }
    }, 0);
  });

  downloadBtn.addEventListener("click", () => {
    resultCanvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "segmentation-result.png";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }, "image/png");
  });
})();
