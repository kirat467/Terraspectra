// Everything about turning a risk map (a grid of numbers from 0 to 1) into colours, labels and pictures.

// colour stops: [risk, [r, g, b, alpha]]. Below 0.15 the map stays see-through so healthy land is not tinted.
const STOPS = [
  [0.15, [255, 236, 139, 0]],
  [0.30, [250, 204, 60, 150]],
  [0.50, [245, 158, 40, 205]],
  [0.70, [232, 98, 40, 230]],
  [0.85, [214, 40, 40, 245]],
  [1.00, [150, 10, 35, 255]],
];

export function riskColor(r) {
  if (r == null || Number.isNaN(r)) return [0, 0, 0, 0];
  if (r <= STOPS[0][0]) return STOPS[0][1];
  for (let i = 1; i < STOPS.length; i++) {
    if (r <= STOPS[i][0]) {
      const [r0, c0] = STOPS[i - 1];
      const [r1, c1] = STOPS[i];
      const t = (r - r0) / (r1 - r0);
      return c0.map((v, k) => Math.round(v + (c1[k] - v) * t));
    }
  }
  return STOPS[STOPS.length - 1][1];
}

export const riskLabel = (r) => (r < 0.3 ? "Low" : r < 0.6 ? "Moderate" : r < 0.8 ? "High" : "Severe");

// solid colour for chips and swatches (the map colours are semi-transparent)
export function riskSolid(r) {
  if (r < 0.15) return "#56655c";
  const [R, G, B] = riskColor(Math.max(r, 0.3));
  return `rgb(${R}, ${G}, ${B})`;
}

// brighter colour for text on the dark panels
export const riskText = (r) => (r < 0.3 ? "#9fb0a6" : r < 0.6 ? "#f5c04a" : r < 0.8 ? "#f58f4f" : "#ff6b5e");

export function legendGradient() {
  const stops = STOPS.slice(1).map(([r, [R, G, B]]) => `rgb(${R}, ${G}, ${B}) ${Math.round(r * 100)}%`);
  return `linear-gradient(90deg, #56655c 0%, ${stops.join(", ")})`;
}

// draw the grid as a small picture, one pixel per cell, so the map can stretch it over the farm
export function riskImage(grid) {
  const rows = grid.length, cols = grid[0].length;
  const canvas = document.createElement("canvas");
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(cols, rows);
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const [R, G, B, A] = riskColor(grid[y][x]);
      const i = (y * cols + x) * 4;
      img.data[i] = R; img.data[i + 1] = G; img.data[i + 2] = B; img.data[i + 3] = A;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL("image/png");
}

// which grid cell is under a map position, and what is its risk (null on tracks and outside the farm)
export function cellAt(risk, lon, lat) {
  if (!risk) return null;
  const [west, south, east, north] = risk.bounds;
  const col = Math.floor(((lon - west) / (east - west)) * risk.cols);
  const row = Math.floor(((north - lat) / (north - south)) * risk.rows);
  if (row < 0 || col < 0 || row >= risk.rows || col >= risk.cols) return null;
  const value = risk.risk[row][col];
  return value == null ? null : { row, col, value };
}
