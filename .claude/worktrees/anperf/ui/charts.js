// charts.js — lightweight Canvas-based chart engine for anperf
// No dependencies. Pure Canvas API with retina support, bezier curves,
// gradient fills, hover interaction, and smooth 60fps animations.

'use strict';

// ─── Utility ───────────────────────────────────────────────────────

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function lerp(a, b, t) { return a + (b - a) * t; }
function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

function formatNum(v, decimals) {
    if (v === null || v === undefined || isNaN(v)) return '—';
    if (Math.abs(v) >= 1e6) return (v / 1e6).toFixed(1) + 'M';
    if (Math.abs(v) >= 1e3) return (v / 1e3).toFixed(1) + 'k';
    return v.toFixed(decimals !== undefined ? decimals : 2);
}

function formatMs(v) {
    if (v >= 1000) return (v / 1000).toFixed(1) + 's';
    return v.toFixed(1) + 'ms';
}

function niceStep(range, targetSteps) {
    const rough = range / targetSteps;
    const mag = Math.pow(10, Math.floor(Math.log10(rough)));
    const frac = rough / mag;
    let nice;
    if (frac <= 1.5) nice = 1;
    else if (frac <= 3) nice = 2;
    else if (frac <= 7) nice = 5;
    else nice = 10;
    return nice * mag;
}

// ─── Base Chart ────────────────────────────────────────────────────

class BaseChart {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.dpr = window.devicePixelRatio || 1;
        this.options = Object.assign({
            padding: { top: 20, right: 20, bottom: 40, left: 60 },
            bgColor: 'transparent',
            gridColor: 'rgba(255,255,255,0.04)',
            axisColor: 'rgba(255,255,255,0.15)',
            textColor: 'rgba(255,255,255,0.4)',
            font: '11px SF Mono, monospace',
            animDuration: 400,
        }, options);

        this.width = 0;
        this.height = 0;
        this.plotX = 0;
        this.plotY = 0;
        this.plotW = 0;
        this.plotH = 0;
        this.animStart = 0;
        this.animProgress = 1;
        this._resizeObserver = null;
        this._hoverX = -1;
        this._hoverY = -1;
        this._tooltip = null;
        this._rafId = 0;

        this._bindEvents();
        this.resize();
    }

    _bindEvents() {
        this._resizeObserver = new ResizeObserver(() => this.resize());
        this._resizeObserver.observe(this.canvas.parentElement || this.canvas);

        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            this._hoverX = (e.clientX - rect.left) * this.dpr;
            this._hoverY = (e.clientY - rect.top) * this.dpr;
            this._screenX = e.clientX;
            this._screenY = e.clientY;
            this._requestFrame();
        });

        this.canvas.addEventListener('mouseleave', () => {
            this._hoverX = -1;
            this._hoverY = -1;
            this.hideTooltip();
            this._requestFrame();
        });
    }

    resize() {
        const rect = this.canvas.parentElement
            ? this.canvas.parentElement.getBoundingClientRect()
            : this.canvas.getBoundingClientRect();
        this.width = rect.width * this.dpr;
        this.height = rect.height * this.dpr;
        this.canvas.width = this.width;
        this.canvas.height = this.height;
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';

        const p = this.options.padding;
        this.plotX = p.left * this.dpr;
        this.plotY = p.top * this.dpr;
        this.plotW = this.width - (p.left + p.right) * this.dpr;
        this.plotH = this.height - (p.top + p.bottom) * this.dpr;

        this._requestFrame();
    }

    _requestFrame() {
        if (this._rafId) return;
        this._rafId = requestAnimationFrame(() => {
            this._rafId = 0;
            this.draw();
        });
    }

    startAnimation() {
        this.animStart = performance.now();
        this.animProgress = 0;
        const tick = () => {
            const elapsed = performance.now() - this.animStart;
            this.animProgress = clamp(elapsed / this.options.animDuration, 0, 1);
            this.draw();
            if (this.animProgress < 1) {
                requestAnimationFrame(tick);
            }
        };
        requestAnimationFrame(tick);
    }

    clear() {
        this.ctx.clearRect(0, 0, this.width, this.height);
    }

    drawGrid(xMin, xMax, yMin, yMax, xLabel, yLabel, logScale) {
        const ctx = this.ctx;
        const dpr = this.dpr;
        ctx.save();

        // Y gridlines
        const yRange = yMax - yMin;
        if (yRange > 0) {
            const yStep = niceStep(yRange, 5);
            const yStart = Math.ceil(yMin / yStep) * yStep;
            ctx.font = (10 * dpr) + 'px SF Mono, monospace';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            for (let v = yStart; v <= yMax; v += yStep) {
                const y = this.plotY + this.plotH - ((v - yMin) / yRange) * this.plotH;
                ctx.strokeStyle = this.options.gridColor;
                ctx.lineWidth = dpr;
                ctx.beginPath();
                ctx.moveTo(this.plotX, y);
                ctx.lineTo(this.plotX + this.plotW, y);
                ctx.stroke();

                ctx.fillStyle = this.options.textColor;
                let label;
                if (logScale) {
                    label = Math.exp(v).toFixed(v < 1 ? 2 : 1);
                } else {
                    label = formatNum(v, v < 1 ? 3 : v < 10 ? 2 : v < 100 ? 1 : 0);
                }
                ctx.fillText(label, this.plotX - 6 * dpr, y);
            }
        }

        // X gridlines
        const xRange = xMax - xMin;
        if (xRange > 0) {
            const xStep = niceStep(xRange, 6);
            const xStart = Math.ceil(xMin / xStep) * xStep;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            for (let v = xStart; v <= xMax; v += xStep) {
                const x = this.plotX + ((v - xMin) / xRange) * this.plotW;
                ctx.strokeStyle = this.options.gridColor;
                ctx.lineWidth = dpr;
                ctx.beginPath();
                ctx.moveTo(x, this.plotY);
                ctx.lineTo(x, this.plotY + this.plotH);
                ctx.stroke();

                ctx.fillStyle = this.options.textColor;
                ctx.fillText(formatNum(v, 0), x, this.plotY + this.plotH + 6 * dpr);
            }
        }

        // Axis labels
        if (xLabel) {
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillStyle = this.options.textColor;
            ctx.fillText(xLabel, this.plotX + this.plotW / 2, this.plotY + this.plotH + 22 * dpr);
        }

        ctx.restore();
    }

    showTooltip(html, screenX, screenY) {
        if (!this._tooltip) {
            this._tooltip = document.createElement('div');
            this._tooltip.className = 'chart-tooltip';
            document.body.appendChild(this._tooltip);
        }
        this._tooltip.innerHTML = html;
        this._tooltip.classList.add('visible');

        // Position near cursor.
        const tw = this._tooltip.offsetWidth;
        const th = this._tooltip.offsetHeight;
        let tx = screenX + 14;
        let ty = screenY - th / 2;
        if (tx + tw > window.innerWidth - 10) tx = screenX - tw - 14;
        if (ty < 10) ty = 10;
        if (ty + th > window.innerHeight - 10) ty = window.innerHeight - th - 10;
        this._tooltip.style.left = tx + 'px';
        this._tooltip.style.top = ty + 'px';
    }

    hideTooltip() {
        if (this._tooltip) {
            this._tooltip.classList.remove('visible');
        }
    }

    destroy() {
        if (this._resizeObserver) {
            this._resizeObserver.disconnect();
        }
        if (this._tooltip) {
            this._tooltip.remove();
            this._tooltip = null;
        }
        if (this._rafId) {
            cancelAnimationFrame(this._rafId);
        }
    }

    draw() {
        // Override in subclass.
    }
}


// ─── Line Chart ────────────────────────────────────────────────────

class LineChart extends BaseChart {
    constructor(canvas, options = {}) {
        super(canvas, Object.assign({
            lineColor: '#00d4ff',
            lineWidth: 2,
            fillGradient: true,
            fillOpacity: 0.15,
            showDots: false,
            dotRadius: 3,
            smoothing: 0.2,
            logScale: false,
            yMin: null,
            yMax: null,
        }, options));

        this.series = []; // [{data: [{x, y}], color, label}]
        this.activeSeries = new Set();
    }

    setData(seriesArray) {
        // seriesArray: [{data: [{x, y}], color, label}]
        this.series = seriesArray;
        this.activeSeries = new Set(seriesArray.map((_, i) => i));
        this.startAnimation();
    }

    appendPoint(seriesIndex, point) {
        if (seriesIndex < this.series.length) {
            this.series[seriesIndex].data.push(point);
            this._requestFrame();
        }
    }

    toggleSeries(index) {
        if (this.activeSeries.has(index)) {
            this.activeSeries.delete(index);
        } else {
            this.activeSeries.add(index);
        }
        this._requestFrame();
    }

    _computeBounds() {
        let xMin = Infinity, xMax = -Infinity;
        let yMin = Infinity, yMax = -Infinity;

        for (const idx of this.activeSeries) {
            const s = this.series[idx];
            if (!s || !s.data.length) continue;
            for (const p of s.data) {
                if (p.x < xMin) xMin = p.x;
                if (p.x > xMax) xMax = p.x;
                let yVal = this.options.logScale ? Math.log(Math.max(p.y, 1e-10)) : p.y;
                if (yVal < yMin) yMin = yVal;
                if (yVal > yMax) yMax = yVal;
            }
        }

        if (!isFinite(xMin)) { xMin = 0; xMax = 1; yMin = 0; yMax = 1; }

        // Apply overrides.
        if (this.options.yMin !== null) yMin = this.options.logScale ? Math.log(Math.max(this.options.yMin, 1e-10)) : this.options.yMin;
        if (this.options.yMax !== null) yMax = this.options.logScale ? Math.log(Math.max(this.options.yMax, 1e-10)) : this.options.yMax;

        // Add 5% padding on Y.
        const yPad = (yMax - yMin) * 0.05 || 0.1;
        yMin -= yPad;
        yMax += yPad;

        return { xMin, xMax, yMin, yMax };
    }

    _toCanvasXY(px, py, bounds) {
        const xFrac = (px - bounds.xMin) / (bounds.xMax - bounds.xMin || 1);
        const yFrac = (py - bounds.yMin) / (bounds.yMax - bounds.yMin || 1);
        return {
            x: this.plotX + xFrac * this.plotW,
            y: this.plotY + this.plotH - yFrac * this.plotH,
        };
    }

    draw() {
        this.clear();
        if (!this.series.length) return;

        const ctx = this.ctx;
        const dpr = this.dpr;
        const bounds = this._computeBounds();
        const progress = easeOut(this.animProgress);

        this.drawGrid(bounds.xMin, bounds.xMax, bounds.yMin, bounds.yMax, 'step', null, this.options.logScale);

        // Draw each active series.
        let nearestSeries = -1, nearestIdx = -1, nearestDist = Infinity;

        for (const si of this.activeSeries) {
            const s = this.series[si];
            if (!s || s.data.length < 2) continue;

            const color = s.color || this.options.lineColor;
            const points = s.data.map(p => {
                const yVal = this.options.logScale ? Math.log(Math.max(p.y, 1e-10)) : p.y;
                return this._toCanvasXY(p.x, yVal, bounds);
            });

            // Clip to plot area.
            ctx.save();
            ctx.beginPath();
            ctx.rect(this.plotX, this.plotY, this.plotW, this.plotH);
            ctx.clip();

            // Animated reveal: only draw up to progress fraction.
            const drawCount = Math.max(2, Math.floor(points.length * progress));
            const pts = points.slice(0, drawCount);

            // Draw fill gradient.
            if (this.options.fillGradient && pts.length >= 2) {
                const grad = ctx.createLinearGradient(0, this.plotY, 0, this.plotY + this.plotH);
                const c = this._parseColor(color);
                grad.addColorStop(0, `rgba(${c.r},${c.g},${c.b},${this.options.fillOpacity})`);
                grad.addColorStop(1, `rgba(${c.r},${c.g},${c.b},0)`);

                ctx.beginPath();
                ctx.moveTo(pts[0].x, this.plotY + this.plotH);
                ctx.lineTo(pts[0].x, pts[0].y);
                this._drawSmoothPath(ctx, pts, false);
                ctx.lineTo(pts[pts.length - 1].x, this.plotY + this.plotH);
                ctx.closePath();
                ctx.fillStyle = grad;
                ctx.fill();
            }

            // Draw line.
            ctx.beginPath();
            ctx.moveTo(pts[0].x, pts[0].y);
            this._drawSmoothPath(ctx, pts, false);
            ctx.strokeStyle = color;
            ctx.lineWidth = this.options.lineWidth * dpr;
            ctx.lineJoin = 'round';
            ctx.lineCap = 'round';
            ctx.stroke();

            // Find nearest point to hover.
            if (this._hoverX >= this.plotX && this._hoverX <= this.plotX + this.plotW) {
                for (let i = 0; i < pts.length; i++) {
                    const d = Math.abs(pts[i].x - this._hoverX);
                    if (d < nearestDist) {
                        nearestDist = d;
                        nearestSeries = si;
                        nearestIdx = i;
                    }
                }
            }

            ctx.restore();
        }

        // Draw hover indicator.
        if (nearestSeries >= 0 && nearestIdx >= 0 && nearestDist < 30 * dpr) {
            const s = this.series[nearestSeries];
            const p = s.data[nearestIdx];
            const yVal = this.options.logScale ? Math.log(Math.max(p.y, 1e-10)) : p.y;
            const cp = this._toCanvasXY(p.x, yVal, bounds);

            // Vertical line.
            ctx.save();
            ctx.beginPath();
            ctx.rect(this.plotX, this.plotY, this.plotW, this.plotH);
            ctx.clip();
            ctx.strokeStyle = 'rgba(255,255,255,0.15)';
            ctx.lineWidth = dpr;
            ctx.setLineDash([4 * dpr, 4 * dpr]);
            ctx.beginPath();
            ctx.moveTo(cp.x, this.plotY);
            ctx.lineTo(cp.x, this.plotY + this.plotH);
            ctx.stroke();
            ctx.setLineDash([]);

            // Dot.
            ctx.beginPath();
            ctx.arc(cp.x, cp.y, 5 * dpr, 0, Math.PI * 2);
            ctx.fillStyle = s.color || this.options.lineColor;
            ctx.fill();
            ctx.strokeStyle = '#0a0a0f';
            ctx.lineWidth = 2 * dpr;
            ctx.stroke();
            ctx.restore();

            // Tooltip.
            let tooltipHTML = `<div class="tooltip-row"><span class="tooltip-label">Step</span><span class="tooltip-value">${p.x}</span></div>`;
            for (const si2 of this.activeSeries) {
                const s2 = this.series[si2];
                if (s2 && nearestIdx < s2.data.length) {
                    const v = s2.data[nearestIdx].y;
                    tooltipHTML += `<div class="tooltip-row"><span class="tooltip-label" style="color:${s2.color}">${s2.label || 'value'}</span><span class="tooltip-value">${formatNum(v, 4)}</span></div>`;
                }
            }
            this.showTooltip(tooltipHTML, this._screenX, this._screenY);
        } else {
            this.hideTooltip();
        }
    }

    _drawSmoothPath(ctx, points, isFirst) {
        const tension = this.options.smoothing;
        for (let i = 1; i < points.length; i++) {
            const p0 = points[Math.max(0, i - 2)];
            const p1 = points[i - 1];
            const p2 = points[i];
            const p3 = points[Math.min(points.length - 1, i + 1)];

            const cp1x = p1.x + (p2.x - p0.x) * tension;
            const cp1y = p1.y + (p2.y - p0.y) * tension;
            const cp2x = p2.x - (p3.x - p1.x) * tension;
            const cp2y = p2.y - (p3.y - p1.y) * tension;

            ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y);
        }
    }

    _parseColor(cssColor) {
        // Handle hex and named colors.
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = 1;
        const c = canvas.getContext('2d');
        c.fillStyle = cssColor;
        c.fillRect(0, 0, 1, 1);
        const d = c.getImageData(0, 0, 1, 1).data;
        return { r: d[0], g: d[1], b: d[2] };
    }
}


// ─── Area Chart (extends LineChart with default area fill) ────────

class AreaChart extends LineChart {
    constructor(canvas, options = {}) {
        super(canvas, Object.assign({
            fillGradient: true,
            fillOpacity: 0.25,
        }, options));
    }
}


// ─── Bar Chart ─────────────────────────────────────────────────────

class BarChart extends BaseChart {
    constructor(canvas, options = {}) {
        super(canvas, Object.assign({
            horizontal: true,
            barGap: 6,
            barRadius: 4,
            colors: ['#00d4ff', '#ffbe0b', '#ff006e', '#39ff14', '#b388ff'],
            showValues: true,
            animDuration: 600,
        }, options));

        this.data = []; // [{label, value, color?}]
        this.prevData = [];
    }

    setData(data) {
        this.prevData = this.data.slice();
        this.data = data;
        this.startAnimation();
    }

    draw() {
        this.clear();
        if (!this.data.length) return;

        const ctx = this.ctx;
        const dpr = this.dpr;
        const progress = easeOut(this.animProgress);

        if (this.options.horizontal) {
            this._drawHorizontal(ctx, dpr, progress);
        } else {
            this._drawVertical(ctx, dpr, progress);
        }
    }

    _drawHorizontal(ctx, dpr, progress) {
        const maxVal = Math.max(...this.data.map(d => d.value), 0.001);
        const barH = Math.min(
            (this.plotH - this.options.barGap * dpr * (this.data.length - 1)) / this.data.length,
            36 * dpr
        );
        const totalHeight = this.data.length * barH + (this.data.length - 1) * this.options.barGap * dpr;
        const startY = this.plotY + (this.plotH - totalHeight) / 2;

        ctx.font = (10 * dpr) + 'px SF Mono, monospace';
        ctx.textBaseline = 'middle';

        for (let i = 0; i < this.data.length; i++) {
            const d = this.data[i];
            const y = startY + i * (barH + this.options.barGap * dpr);
            const color = d.color || this.options.colors[i % this.options.colors.length];

            // Animate from previous value.
            let prevVal = 0;
            if (this.prevData[i]) prevVal = this.prevData[i].value;
            const curVal = lerp(prevVal, d.value, progress);
            const w = (curVal / maxVal) * this.plotW * 0.75;

            // Label on the left.
            ctx.fillStyle = this.options.textColor;
            ctx.textAlign = 'right';
            ctx.fillText(d.label, this.plotX - 8 * dpr, y + barH / 2);

            // Bar with rounded ends.
            const r = Math.min(this.options.barRadius * dpr, barH / 2);
            const grad = ctx.createLinearGradient(this.plotX, 0, this.plotX + w, 0);
            const c = this._parseColor(color);
            grad.addColorStop(0, `rgba(${c.r},${c.g},${c.b},0.8)`);
            grad.addColorStop(1, `rgba(${c.r},${c.g},${c.b},0.4)`);

            ctx.beginPath();
            this._roundedRect(ctx, this.plotX, y, Math.max(w, r * 2), barH, r);
            ctx.fillStyle = grad;
            ctx.fill();

            // Value label.
            if (this.options.showValues) {
                ctx.fillStyle = 'rgba(255,255,255,0.7)';
                ctx.textAlign = 'left';
                ctx.fillText(formatMs(curVal), this.plotX + w + 6 * dpr, y + barH / 2);
            }
        }
    }

    _drawVertical(ctx, dpr, progress) {
        const maxVal = Math.max(...this.data.map(d => d.value), 0.001);
        const barW = Math.min(
            (this.plotW - this.options.barGap * dpr * (this.data.length - 1)) / this.data.length,
            48 * dpr
        );
        const totalWidth = this.data.length * barW + (this.data.length - 1) * this.options.barGap * dpr;
        const startX = this.plotX + (this.plotW - totalWidth) / 2;

        ctx.font = (10 * dpr) + 'px SF Mono, monospace';

        for (let i = 0; i < this.data.length; i++) {
            const d = this.data[i];
            const x = startX + i * (barW + this.options.barGap * dpr);
            const color = d.color || this.options.colors[i % this.options.colors.length];

            let prevVal = 0;
            if (this.prevData[i]) prevVal = this.prevData[i].value;
            const curVal = lerp(prevVal, d.value, progress);
            const h = (curVal / maxVal) * this.plotH * 0.85;

            const r = Math.min(this.options.barRadius * dpr, barW / 2);
            const grad = ctx.createLinearGradient(0, this.plotY + this.plotH, 0, this.plotY + this.plotH - h);
            const c = this._parseColor(color);
            grad.addColorStop(0, `rgba(${c.r},${c.g},${c.b},0.3)`);
            grad.addColorStop(1, `rgba(${c.r},${c.g},${c.b},0.8)`);

            ctx.beginPath();
            this._roundedRect(ctx, x, this.plotY + this.plotH - h, barW, h, r);
            ctx.fillStyle = grad;
            ctx.fill();

            // Label below.
            ctx.fillStyle = this.options.textColor;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText(d.label, x + barW / 2, this.plotY + this.plotH + 6 * dpr);
        }
    }

    _roundedRect(ctx, x, y, w, h, r) {
        if (w < 0) w = 0;
        if (h < 0) h = 0;
        r = Math.min(r, w / 2, h / 2);
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
    }

    _parseColor(cssColor) {
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = 1;
        const c = canvas.getContext('2d');
        c.fillStyle = cssColor;
        c.fillRect(0, 0, 1, 1);
        const d = c.getImageData(0, 0, 1, 1).data;
        return { r: d[0], g: d[1], b: d[2] };
    }
}


// ─── Waterfall Chart ───────────────────────────────────────────────

class WaterfallChart extends BaseChart {
    constructor(canvas, options = {}) {
        super(canvas, Object.assign({
            barHeight: 22,
            barGap: 4,
            barRadius: 4,
            labelWidth: 130,
            colorFast: '#39ff14',
            colorMid: '#ffbe0b',
            colorSlow: '#ff006e',
            animDuration: 600,
            padding: { top: 10, right: 20, bottom: 10, left: 140 },
        }, options));

        this.data = []; // [{label, value}] sorted by value desc
        this.prevData = [];
    }

    setData(data) {
        this.prevData = this.data.slice();
        // Sort by value descending.
        this.data = data.slice().sort((a, b) => b.value - a.value);
        this.startAnimation();
    }

    draw() {
        this.clear();
        if (!this.data.length) return;

        const ctx = this.ctx;
        const dpr = this.dpr;
        const progress = easeOut(this.animProgress);

        const maxVal = Math.max(...this.data.map(d => d.value), 0.001);
        const barH = Math.min(this.options.barHeight * dpr, (this.plotH - this.options.barGap * dpr * (this.data.length - 1)) / this.data.length);

        ctx.font = (10 * dpr) + 'px SF Mono, monospace';
        ctx.textBaseline = 'middle';

        for (let i = 0; i < this.data.length; i++) {
            const d = this.data[i];
            const y = this.plotY + i * (barH + this.options.barGap * dpr);

            let prevVal = 0;
            if (this.prevData.length > 0) {
                const prev = this.prevData.find(p => p.label === d.label);
                if (prev) prevVal = prev.value;
            }
            const curVal = lerp(prevVal, d.value, progress);
            const w = Math.max((curVal / maxVal) * this.plotW, 2 * dpr);

            // Color gradient based on relative magnitude.
            const frac = curVal / maxVal;
            const color = this._gradientColor(frac);

            const r = Math.min(this.options.barRadius * dpr, barH / 2);
            const c = this._parseColor(color);
            const grad = ctx.createLinearGradient(this.plotX, 0, this.plotX + w, 0);
            grad.addColorStop(0, `rgba(${c.r},${c.g},${c.b},0.7)`);
            grad.addColorStop(1, `rgba(${c.r},${c.g},${c.b},0.3)`);

            ctx.beginPath();
            this._roundedRect(ctx, this.plotX, y, w, barH, r);
            ctx.fillStyle = grad;
            ctx.fill();

            // Label.
            ctx.fillStyle = 'rgba(255,255,255,0.5)';
            ctx.textAlign = 'right';
            ctx.fillText(d.label, this.plotX - 8 * dpr, y + barH / 2);

            // Value.
            ctx.fillStyle = 'rgba(255,255,255,0.6)';
            ctx.textAlign = 'left';
            ctx.fillText(formatMs(curVal), this.plotX + w + 6 * dpr, y + barH / 2);
        }
    }

    _gradientColor(frac) {
        // 0 = fast (green), 0.5 = mid (amber), 1 = slow (red/pink)
        if (frac < 0.5) {
            return this._lerpColor(this.options.colorFast, this.options.colorMid, frac * 2);
        }
        return this._lerpColor(this.options.colorMid, this.options.colorSlow, (frac - 0.5) * 2);
    }

    _lerpColor(c1, c2, t) {
        const a = this._parseColor(c1);
        const b = this._parseColor(c2);
        const r = Math.round(lerp(a.r, b.r, t));
        const g = Math.round(lerp(a.g, b.g, t));
        const bl = Math.round(lerp(a.b, b.b, t));
        return `rgb(${r},${g},${bl})`;
    }

    _roundedRect(ctx, x, y, w, h, r) {
        if (w < 0) w = 0;
        if (h < 0) h = 0;
        r = Math.min(r, w / 2, h / 2);
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
    }

    _parseColor(cssColor) {
        // Handle rgb(r,g,b) and hex.
        if (cssColor.startsWith('rgb(')) {
            const m = cssColor.match(/(\d+)/g);
            return { r: +m[0], g: +m[1], b: +m[2] };
        }
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = 1;
        const c = canvas.getContext('2d');
        c.fillStyle = cssColor;
        c.fillRect(0, 0, 1, 1);
        const d = c.getImageData(0, 0, 1, 1).data;
        return { r: d[0], g: d[1], b: d[2] };
    }
}


// ─── Sparkline ─────────────────────────────────────────────────────

class Sparkline {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.dpr = window.devicePixelRatio || 1;
        this.options = Object.assign({
            color: '#00d4ff',
            fillOpacity: 0.3,
            lineWidth: 1.5,
            smoothing: 0.15,
        }, options);
        this.data = [];

        this._resizeObserver = new ResizeObserver(() => this._resize());
        this._resizeObserver.observe(this.canvas.parentElement || this.canvas);
        this._resize();
    }

    _resize() {
        const rect = this.canvas.parentElement
            ? this.canvas.parentElement.getBoundingClientRect()
            : this.canvas.getBoundingClientRect();
        this.width = rect.width * this.dpr;
        this.height = rect.height * this.dpr;
        this.canvas.width = this.width;
        this.canvas.height = this.height;
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';
        this.draw();
    }

    setData(values) {
        this.data = values;
        this.draw();
    }

    push(value) {
        this.data.push(value);
        if (this.data.length > 200) this.data.shift();
        this.draw();
    }

    draw() {
        const ctx = this.ctx;
        const w = this.width;
        const h = this.height;
        ctx.clearRect(0, 0, w, h);

        if (this.data.length < 2) return;

        let min = Infinity, max = -Infinity;
        for (const v of this.data) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        const range = max - min || 1;
        const pad = h * 0.1;

        const points = this.data.map((v, i) => ({
            x: (i / (this.data.length - 1)) * w,
            y: pad + (1 - (v - min) / range) * (h - 2 * pad),
        }));

        // Fill gradient.
        const c = this._parseColor(this.options.color);
        const grad = ctx.createLinearGradient(0, 0, 0, h);
        grad.addColorStop(0, `rgba(${c.r},${c.g},${c.b},${this.options.fillOpacity})`);
        grad.addColorStop(1, `rgba(${c.r},${c.g},${c.b},0)`);

        ctx.beginPath();
        ctx.moveTo(points[0].x, h);
        ctx.lineTo(points[0].x, points[0].y);
        this._smoothPath(ctx, points);
        ctx.lineTo(points[points.length - 1].x, h);
        ctx.closePath();
        ctx.fillStyle = grad;
        ctx.fill();

        // Line.
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        this._smoothPath(ctx, points);
        ctx.strokeStyle = this.options.color;
        ctx.lineWidth = this.options.lineWidth * this.dpr;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.stroke();
    }

    _smoothPath(ctx, points) {
        const t = this.options.smoothing;
        for (let i = 1; i < points.length; i++) {
            const p0 = points[Math.max(0, i - 2)];
            const p1 = points[i - 1];
            const p2 = points[i];
            const p3 = points[Math.min(points.length - 1, i + 1)];
            ctx.bezierCurveTo(
                p1.x + (p2.x - p0.x) * t, p1.y + (p2.y - p0.y) * t,
                p2.x - (p3.x - p1.x) * t, p2.y - (p3.y - p1.y) * t,
                p2.x, p2.y
            );
        }
    }

    _parseColor(cssColor) {
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = 1;
        const c = canvas.getContext('2d');
        c.fillStyle = cssColor;
        c.fillRect(0, 0, 1, 1);
        const d = c.getImageData(0, 0, 1, 1).data;
        return { r: d[0], g: d[1], b: d[2] };
    }

    destroy() {
        if (this._resizeObserver) this._resizeObserver.disconnect();
    }
}


// ─── Exports ───────────────────────────────────────────────────────

window.AnperfCharts = {
    LineChart,
    AreaChart,
    BarChart,
    WaterfallChart,
    Sparkline,
    formatNum,
    formatMs,
};
