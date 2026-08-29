/**
 * Client-side port of the Python segmentation pipeline (maxflow/graph_builder.py,
 * maxflow/grid_maxflow.py, segmentation/segmenter.py) so the GitHub Pages site can
 * run real interactive segmentation with no backend server.
 *
 * Mirrors the Python two-stage design: the min-cut is solved on a small grid
 * (Edmonds-Karp is O(V * E^2), so a per-pixel graph on a real photo is not
 * viable), then the resulting mask is upsampled and snapped back onto the
 * full-resolution image's edges with a guided filter. The pixels you get out
 * are the pixels you put in -- the solver's coarse grid never touches them.
 */

(() => {
  "use strict";

  // ---- tunables -------------------------------------------------------

  // Longest side of the grid the cut is solved on.
  const PROCESSING_MAX_DIM = 64;
  // Longest side the guided-filter refinement runs at. The mask is upsampled
  // from here to true full resolution; capping it keeps memory bounded on
  // large photos without touching output resolution.
  const REFINE_MAX_DIM = 1400;
  // On-screen canvas size. Purely a UI choice -- scribbles are rasterised at
  // the image's own resolution regardless.
  const DISPLAY_MAX_DIM = 520;

  const UNMARKED = 0;
  const FOREGROUND = 1;
  const BACKGROUND = 2;

  const FROM_SOURCE = 0;
  const FROM_WEST = 1;
  const FROM_EAST = 2;
  const FROM_NORTH = 3;
  const FROM_SOUTH = 4;
  const UNREACHED = -1;

  // ---- maxflow/graph_builder.py equivalent ----------------------------

  function toGrayscale(rgba, pixelCount) {
    const gray = new Float32Array(pixelCount);
    for (let i = 0; i < pixelCount; i++) {
      const o = i * 4;
      gray[i] = 0.299 * rgba[o] + 0.587 * rgba[o + 1] + 0.114 * rgba[o + 2];
    }
    return gray;
  }

  function estimateBeta(gray, width, height) {
    let sumSquares = 0;
    let count = 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width - 1; x++) {
        const d = gray[y * width + x] - gray[y * width + x + 1];
        sumSquares += d * d;
        count++;
      }
    }
    for (let y = 0; y < height - 1; y++) {
      for (let x = 0; x < width; x++) {
        const d = gray[y * width + x] - gray[(y + 1) * width + x];
        sumSquares += d * d;
        count++;
      }
    }
    const meanSquare = count > 0 ? sumSquares / count : 0;
    return meanSquare === 0 ? 1.0 : 1.0 / (2.0 * meanSquare);
  }

  function buildGridCapacities(gray, width, height, scribbleMask, nLinkWeight, numBins) {
    const n = width * height;
    const beta = estimateBeta(gray, width, height);

    const east = new Float64Array(n);
    const south = new Float64Array(n);
    const neighborSum = new Float64Array(n);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const p = y * width + x;
        if (x + 1 < width) {
          const d = gray[p] - gray[p + 1];
          const w = nLinkWeight * Math.exp(-beta * d * d);
          east[p] = w;
          neighborSum[p] += w;
          neighborSum[p + 1] += w;
        }
        if (y + 1 < height) {
          const d = gray[p] - gray[p + width];
          const w = nLinkWeight * Math.exp(-beta * d * d);
          south[p] = w;
          neighborSum[p] += w;
          neighborSum[p + width] += w;
        }
      }
    }

    // Histograms over the scribbled pixels, with Laplace smoothing so no bin
    // is ever zero (log(0) would be infinite).
    const fgHist = new Float64Array(numBins).fill(1.0);
    const bgHist = new Float64Array(numBins).fill(1.0);
    let fgTotal = numBins;
    let bgTotal = numBins;
    let hasForeground = false;
    let hasBackground = false;
    const binWidth = 256.0 / numBins;

    for (let i = 0; i < n; i++) {
      const label = scribbleMask[i];
      if (label === UNMARKED) continue;
      let bin = Math.floor(gray[i] / binWidth);
      if (bin < 0) bin = 0;
      else if (bin >= numBins) bin = numBins - 1;
      if (label === FOREGROUND) {
        fgHist[bin] += 1;
        fgTotal += 1;
        hasForeground = true;
      } else {
        bgHist[bin] += 1;
        bgTotal += 1;
        hasBackground = true;
      }
    }
    if (!hasForeground || !hasBackground) {
      throw new Error("Scribble at least one foreground (red) and one background (blue) area first.");
    }
    for (let b = 0; b < numBins; b++) {
      fgHist[b] /= fgTotal;
      bgHist[b] /= bgTotal;
    }

    let maxNeighborSum = 0;
    for (let i = 0; i < n; i++) {
      if (neighborSum[i] > maxNeighborSum) maxNeighborSum = neighborSum[i];
    }
    // K must exceed any pixel's total N-link weight so a scribbled pixel's
    // t-link is never the cheapest thing to cut.
    const hardConstraint = 1.0 + maxNeighborSum;

    const source = new Float64Array(n);
    const sink = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const label = scribbleMask[i];
      if (label === FOREGROUND) {
        source[i] = hardConstraint;
        sink[i] = 0;
      } else if (label === BACKGROUND) {
        source[i] = 0;
        sink[i] = hardConstraint;
      } else {
        let bin = Math.floor(gray[i] / binWidth);
        if (bin < 0) bin = 0;
        else if (bin >= numBins) bin = numBins - 1;
        // A label costs little when the pixel looks like the *other* model.
        source[i] = -Math.log(bgHist[bin]);
        sink[i] = -Math.log(fgHist[bin]);
      }
    }

    return { source, sink, east, south };
  }

  // ---- maxflow/grid_maxflow.py equivalent -----------------------------

  function solveGridMinCut(capacities, width, height) {
    const n = width * height;
    const capSource = capacities.source;
    const capSink = capacities.sink;

    // An undirected N-link of weight w is a pair of directed edges of weight w.
    const capEast = capacities.east;
    const capSouth = capacities.south;
    const capWest = new Float64Array(n);
    const capNorth = new Float64Array(n);
    for (let y = 0; y < height; y++) {
      for (let x = 1; x < width; x++) {
        capWest[y * width + x] = capEast[y * width + x - 1];
      }
    }
    for (let y = 1; y < height; y++) {
      for (let x = 0; x < width; x++) {
        capNorth[y * width + x] = capSouth[(y - 1) * width + x];
      }
    }

    // Every SOURCE -> pixel -> SINK path is a shortest path and they share no
    // edges, so they can all be saturated at once instead of one BFS each.
    for (let i = 0; i < n; i++) {
      const direct = Math.min(capSource[i], capSink[i]);
      if (direct > 0) {
        capSource[i] -= direct;
        capSink[i] -= direct;
      }
    }

    const parent = new Int8Array(n);
    const queue = new Int32Array(n);

    for (;;) {
      parent.fill(UNREACHED);
      let head = 0;
      let tail = 0;
      for (let i = 0; i < n; i++) {
        if (capSource[i] > 0) {
          parent[i] = FROM_SOURCE;
          queue[tail++] = i;
        }
      }

      // FIFO order means the first dequeued pixel that can still reach the
      // sink sits on a shortest augmenting path -- the Edmonds-Karp rule.
      let sinkPixel = -1;
      while (head < tail) {
        const p = queue[head++];
        if (capSink[p] > 0) {
          sinkPixel = p;
          break;
        }
        const y = (p / width) | 0;
        const x = p - y * width;

        if (x + 1 < width && capEast[p] > 0 && parent[p + 1] === UNREACHED) {
          parent[p + 1] = FROM_WEST;
          queue[tail++] = p + 1;
        }
        if (x > 0 && capWest[p] > 0 && parent[p - 1] === UNREACHED) {
          parent[p - 1] = FROM_EAST;
          queue[tail++] = p - 1;
        }
        if (y > 0 && capNorth[p] > 0 && parent[p - width] === UNREACHED) {
          parent[p - width] = FROM_SOUTH;
          queue[tail++] = p - width;
        }
        if (y + 1 < height && capSouth[p] > 0 && parent[p + width] === UNREACHED) {
          parent[p + width] = FROM_NORTH;
          queue[tail++] = p + width;
        }
      }
      if (sinkPixel < 0) break;

      // Walk the path back to the source, then push its bottleneck along it.
      const path = [sinkPixel];
      let cur = sinkPixel;
      while (parent[cur] !== FROM_SOURCE) {
        switch (parent[cur]) {
          case FROM_WEST: cur -= 1; break;
          case FROM_EAST: cur += 1; break;
          case FROM_NORTH: cur -= width; break;
          default: cur += width; break; // FROM_SOUTH
        }
        path.push(cur);
      }
      path.reverse();

      let bottleneck = Math.min(capSource[path[0]], capSink[path[path.length - 1]]);
      for (let i = 0; i < path.length - 1; i++) {
        const a = path[i];
        const b = path[i + 1];
        let capacity;
        if (b === a + 1) capacity = capEast[a];
        else if (b === a - 1) capacity = capWest[a];
        else if (b === a + width) capacity = capSouth[a];
        else capacity = capNorth[a];
        if (capacity < bottleneck) bottleneck = capacity;
      }

      capSource[path[0]] -= bottleneck;
      capSink[path[path.length - 1]] -= bottleneck;
      for (let i = 0; i < path.length - 1; i++) {
        const a = path[i];
        const b = path[i + 1];
        if (b === a + 1) { capEast[a] -= bottleneck; capWest[b] += bottleneck; }
        else if (b === a - 1) { capWest[a] -= bottleneck; capEast[b] += bottleneck; }
        else if (b === a + width) { capSouth[a] -= bottleneck; capNorth[b] += bottleneck; }
        else { capNorth[a] -= bottleneck; capSouth[b] += bottleneck; }
      }
    }

    // Whatever the source can still reach is the source side of the min-cut.
    const reached = new Uint8Array(n);
    let head = 0;
    let tail = 0;
    for (let i = 0; i < n; i++) {
      if (capSource[i] > 0) {
        reached[i] = 1;
        queue[tail++] = i;
      }
    }
    while (head < tail) {
      const p = queue[head++];
      const y = (p / width) | 0;
      const x = p - y * width;
      if (x + 1 < width && capEast[p] > 0 && !reached[p + 1]) { reached[p + 1] = 1; queue[tail++] = p + 1; }
      if (x > 0 && capWest[p] > 0 && !reached[p - 1]) { reached[p - 1] = 1; queue[tail++] = p - 1; }
      if (y > 0 && capNorth[p] > 0 && !reached[p - width]) { reached[p - width] = 1; queue[tail++] = p - width; }
      if (y + 1 < height && capSouth[p] > 0 && !reached[p + width]) { reached[p + width] = 1; queue[tail++] = p + width; }
    }
    return reached;
  }

  // ---- guided-filter refinement (segmenter.py equivalent) -------------

  /** Mean over a (2r+1) square window, via a summed-area table: O(n) per call. */
  function boxFilter(src, width, height, radius) {
    const integralWidth = width + 1;
    const integral = new Float64Array(integralWidth * (height + 1));
    for (let y = 0; y < height; y++) {
      let rowSum = 0;
      for (let x = 0; x < width; x++) {
        rowSum += src[y * width + x];
        integral[(y + 1) * integralWidth + x + 1] = integral[y * integralWidth + x + 1] + rowSum;
      }
    }

    const out = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
      const y0 = Math.max(0, y - radius);
      const y1 = Math.min(height - 1, y + radius);
      for (let x = 0; x < width; x++) {
        const x0 = Math.max(0, x - radius);
        const x1 = Math.min(width - 1, x + radius);
        const total =
          integral[(y1 + 1) * integralWidth + x1 + 1] -
          integral[y0 * integralWidth + x1 + 1] -
          integral[(y1 + 1) * integralWidth + x0] +
          integral[y0 * integralWidth + x0];
        out[y * width + x] = total / ((y1 - y0 + 1) * (x1 - x0 + 1));
      }
    }
    return out;
  }

  /**
   * Edge-aware smoothing of `target` under the structure of `guide`
   * (He et al., "Guided Image Filtering"). Fits target ~= a*guide + b in every
   * window; where the guide has an edge the fit follows it, which is what pulls
   * a coarse mask boundary onto the real edges of the full-resolution image.
   */
  function guidedFilter(guide, target, width, height, radius, eps) {
    const n = width * height;
    const guideTarget = new Float32Array(n);
    const guideSq = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      guideTarget[i] = guide[i] * target[i];
      guideSq[i] = guide[i] * guide[i];
    }

    const meanGuide = boxFilter(guide, width, height, radius);
    const meanTarget = boxFilter(target, width, height, radius);
    const meanGuideTarget = boxFilter(guideTarget, width, height, radius);
    const meanGuideSq = boxFilter(guideSq, width, height, radius);

    const a = new Float32Array(n);
    const b = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const covariance = meanGuideTarget[i] - meanGuide[i] * meanTarget[i];
      const variance = meanGuideSq[i] - meanGuide[i] * meanGuide[i];
      a[i] = covariance / (variance + eps);
      b[i] = meanTarget[i] - a[i] * meanGuide[i];
    }

    const meanA = boxFilter(a, width, height, radius);
    const meanB = boxFilter(b, width, height, radius);

    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = meanA[i] * guide[i] + meanB[i];
    return out;
  }

  /** Bilinear resample of a single-channel float image. */
  function resampleBilinear(src, srcWidth, srcHeight, dstWidth, dstHeight) {
    const out = new Float32Array(dstWidth * dstHeight);
    const scaleX = srcWidth / dstWidth;
    const scaleY = srcHeight / dstHeight;
    for (let y = 0; y < dstHeight; y++) {
      const sy = Math.min(srcHeight - 1, Math.max(0, (y + 0.5) * scaleY - 0.5));
      const y0 = Math.floor(sy);
      const y1 = Math.min(srcHeight - 1, y0 + 1);
      const wy = sy - y0;
      for (let x = 0; x < dstWidth; x++) {
        const sx = Math.min(srcWidth - 1, Math.max(0, (x + 0.5) * scaleX - 0.5));
        const x0 = Math.floor(sx);
        const x1 = Math.min(srcWidth - 1, x0 + 1);
        const wx = sx - x0;
        const top = src[y0 * srcWidth + x0] * (1 - wx) + src[y0 * srcWidth + x1] * wx;
        const bottom = src[y1 * srcWidth + x0] * (1 - wx) + src[y1 * srcWidth + x1] * wx;
        out[y * dstWidth + x] = top * (1 - wy) + bottom * wy;
      }
    }
    return out;
  }

  // ---- UI -------------------------------------------------------------

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
  const backgroundSelect = document.getElementById("backgroundStyle");

  if (!fileInput) return; // section not present on this page

  const displayCtx = tryCanvas.getContext("2d");

  let fullCanvas = null; // the uploaded image at its own resolution
  let strokeCanvas = null; // scribbles, also at full resolution
  let strokeCtx = null;
  let fullWidth = 0;
  let fullHeight = 0;
  let displayScale = 1; // full-res pixels per display pixel
  let latestMask = null; // Uint8Array at full resolution
  let isDrawing = false;
  let lastPoint = null;

  function currentStrokeColor() {
    const checked = document.querySelector('input[name="scribbleMode"]:checked');
    return checked && checked.value === "bg" ? "#0000ff" : "#ff0000";
  }

  function setStatus(text, isError) {
    tryStatus.textContent = text;
    tryStatus.classList.toggle("try-status-error", Boolean(isError));
  }

  function redrawDisplay() {
    displayCtx.clearRect(0, 0, tryCanvas.width, tryCanvas.height);
    displayCtx.drawImage(fullCanvas, 0, 0, tryCanvas.width, tryCanvas.height);
    displayCtx.drawImage(strokeCanvas, 0, 0, tryCanvas.width, tryCanvas.height);
  }

  function displayToFull(evt) {
    const rect = tryCanvas.getBoundingClientRect();
    const x = ((evt.clientX - rect.left) / rect.width) * fullWidth;
    const y = ((evt.clientY - rect.top) / rect.height) * fullHeight;
    return [x, y];
  }

  function drawStroke(from, to) {
    // Brush size is chosen in display pixels, so scale it up to match.
    strokeCtx.strokeStyle = currentStrokeColor();
    strokeCtx.fillStyle = currentStrokeColor();
    strokeCtx.lineWidth = Math.max(1, Number(brushSizeInput.value) * displayScale);
    strokeCtx.lineCap = "round";
    strokeCtx.lineJoin = "round";
    strokeCtx.beginPath();
    strokeCtx.moveTo(from[0], from[1]);
    strokeCtx.lineTo(to[0], to[1]);
    strokeCtx.stroke();
  }

  function loadImageFile(file) {
    const img = new Image();
    img.onload = () => {
      fullWidth = img.naturalWidth;
      fullHeight = img.naturalHeight;

      fullCanvas = document.createElement("canvas");
      fullCanvas.width = fullWidth;
      fullCanvas.height = fullHeight;
      fullCanvas.getContext("2d").drawImage(img, 0, 0);

      strokeCanvas = document.createElement("canvas");
      strokeCanvas.width = fullWidth;
      strokeCanvas.height = fullHeight;
      strokeCtx = strokeCanvas.getContext("2d", { willReadFrequently: true });

      const scale = Math.min(1, DISPLAY_MAX_DIM / Math.max(fullWidth, fullHeight));
      tryCanvas.width = Math.max(1, Math.round(fullWidth * scale));
      tryCanvas.height = Math.max(1, Math.round(fullHeight * scale));
      displayScale = fullWidth / tryCanvas.width;

      latestMask = null;
      redrawDisplay();

      scribbleControls.hidden = false;
      canvasWrap.hidden = false;
      tryActions.hidden = false;
      tryResults.hidden = true;
      setStatus(`Loaded ${fullWidth}x${fullHeight}. Results keep this resolution.`, false);
      URL.revokeObjectURL(img.src);
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
    if (!strokeCtx) return;
    isDrawing = true;
    tryCanvas.setPointerCapture(evt.pointerId);
    lastPoint = displayToFull(evt);
    drawStroke(lastPoint, lastPoint);
    redrawDisplay();
  });

  tryCanvas.addEventListener("pointermove", (evt) => {
    if (!isDrawing) return;
    const point = displayToFull(evt);
    drawStroke(lastPoint || point, point);
    lastPoint = point;
    redrawDisplay();
  });

  function stopDrawing() {
    isDrawing = false;
    lastPoint = null;
  }
  tryCanvas.addEventListener("pointerup", stopDrawing);
  tryCanvas.addEventListener("pointerleave", stopDrawing);
  tryCanvas.addEventListener("pointercancel", stopDrawing);

  clearBtn.addEventListener("click", () => {
    if (!strokeCtx) return;
    strokeCtx.clearRect(0, 0, fullWidth, fullHeight);
    latestMask = null;
    redrawDisplay();
    tryResults.hidden = true;
    setStatus("", false);
  });

  /** Read the stroke layer back as per-pixel FOREGROUND/BACKGROUND labels. */
  function readScribbleLabels(width, height) {
    const scaled = document.createElement("canvas");
    scaled.width = width;
    scaled.height = height;
    const ctx = scaled.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(strokeCanvas, 0, 0, width, height);
    const data = ctx.getImageData(0, 0, width, height).data;

    const labels = new Uint8Array(width * height);
    for (let i = 0; i < labels.length; i++) {
      const o = i * 4;
      if (data[o + 3] < 32) continue; // untouched
      labels[i] = data[o] > data[o + 2] ? FOREGROUND : BACKGROUND;
    }
    return labels;
  }

  function imageDataAt(width, height) {
    const scaled = document.createElement("canvas");
    scaled.width = width;
    scaled.height = height;
    const ctx = scaled.getContext("2d", { willReadFrequently: true });
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(fullCanvas, 0, 0, width, height);
    return ctx.getImageData(0, 0, width, height).data;
  }

  function fitWithin(width, height, maxDim) {
    const scale = Math.min(1, maxDim / Math.max(width, height));
    return [Math.max(1, Math.round(width * scale)), Math.max(1, Math.round(height * scale))];
  }

  /** Solve the cut coarsely, then refine the mask up to full resolution. */
  function computeFullResolutionMask() {
    const [procWidth, procHeight] = fitWithin(fullWidth, fullHeight, PROCESSING_MAX_DIM);

    const procRgba = imageDataAt(procWidth, procHeight);
    const procGray = toGrayscale(procRgba, procWidth * procHeight);
    const procLabels = readScribbleLabels(procWidth, procHeight);

    const capacities = buildGridCapacities(procGray, procWidth, procHeight, procLabels, 50.0, 32);
    const coarse = solveGridMinCut(capacities, procWidth, procHeight);

    const coarseFloat = new Float32Array(coarse.length);
    for (let i = 0; i < coarse.length; i++) coarseFloat[i] = coarse[i];

    // Refine against real image structure at a bounded resolution...
    const [refWidth, refHeight] = fitWithin(fullWidth, fullHeight, REFINE_MAX_DIM);
    const refRgba = imageDataAt(refWidth, refHeight);
    const refGray = toGrayscale(refRgba, refWidth * refHeight);
    const guide = new Float32Array(refGray.length);
    for (let i = 0; i < refGray.length; i++) guide[i] = refGray[i] / 255;

    const upsampled = resampleBilinear(coarseFloat, procWidth, procHeight, refWidth, refHeight);
    const radius = Math.max(2, Math.round(refWidth / procWidth));
    const refined = guidedFilter(guide, upsampled, refWidth, refHeight, radius, 1e-4);

    // ...then carry the refined mask the rest of the way to true full size.
    const atFullSize =
      refWidth === fullWidth && refHeight === fullHeight
        ? refined
        : resampleBilinear(refined, refWidth, refHeight, fullWidth, fullHeight);

    const mask = new Uint8Array(fullWidth * fullHeight);
    for (let i = 0; i < mask.length; i++) mask[i] = atFullSize[i] >= 0.5 ? 1 : 0;

    // Hard constraints are part of the problem statement, so they must hold
    // after refinement too.
    const fullLabels = readScribbleLabels(fullWidth, fullHeight);
    for (let i = 0; i < mask.length; i++) {
      if (fullLabels[i] === FOREGROUND) mask[i] = 1;
      else if (fullLabels[i] === BACKGROUND) mask[i] = 0;
    }
    return mask;
  }

  function renderResult() {
    const style = backgroundSelect ? backgroundSelect.value : "black";

    resultCanvas.width = fullWidth;
    resultCanvas.height = fullHeight;
    const ctx = resultCanvas.getContext("2d");
    ctx.drawImage(fullCanvas, 0, 0);

    const imageData = ctx.getImageData(0, 0, fullWidth, fullHeight);
    const data = imageData.data;
    for (let i = 0; i < latestMask.length; i++) {
      if (latestMask[i]) continue; // foreground pixels stay exactly as they were
      const o = i * 4;
      if (style === "transparent") {
        data[o + 3] = 0;
      } else if (style === "dim") {
        data[o] >>= 2;
        data[o + 1] >>= 2;
        data[o + 2] >>= 2;
      } else {
        data[o] = 0;
        data[o + 1] = 0;
        data[o + 2] = 0;
      }
    }
    ctx.putImageData(imageData, 0, 0);

    originalCanvas.width = fullWidth;
    originalCanvas.height = fullHeight;
    originalCanvas.getContext("2d").drawImage(fullCanvas, 0, 0);

    scribbleCanvas.width = fullWidth;
    scribbleCanvas.height = fullHeight;
    const scribbleCtx = scribbleCanvas.getContext("2d");
    scribbleCtx.drawImage(fullCanvas, 0, 0);
    scribbleCtx.drawImage(strokeCanvas, 0, 0);
  }

  segmentBtn.addEventListener("click", () => {
    if (!strokeCtx) return;
    segmentBtn.disabled = true;
    setStatus("Segmenting…", false);

    // Defer so the status paints before the synchronous solve blocks the thread.
    setTimeout(() => {
      const start = performance.now();
      try {
        latestMask = computeFullResolutionMask();
        const elapsed = (performance.now() - start) / 1000;
        renderResult();
        tryResults.hidden = false;
        setStatus(
          `Segmented in ${elapsed.toFixed(2)}s — result is ${fullWidth}x${fullHeight}, ` +
            "same as the upload.",
          false
        );
      } catch (err) {
        setStatus(err.message || "Segmentation failed.", true);
      } finally {
        segmentBtn.disabled = false;
      }
    }, 0);
  });

  if (backgroundSelect) {
    backgroundSelect.addEventListener("change", () => {
      if (latestMask) renderResult();
    });
  }

  downloadBtn.addEventListener("click", () => {
    resultCanvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "segmentation-result.png";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }, "image/png");
  });
})();
