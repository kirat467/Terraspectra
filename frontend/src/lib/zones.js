// Priority zones: connected patches of the risk map at or above a threshold that cover at least a minimum area.
// Pure functions (no React, no browser), so they can be tested with `node scripts/test_zones.mjs`.

export const SQ_M_PER_ACRE = 4046.8564;
export const acresPerCell = (pixelSizeM) => (pixelSizeM * pixelSizeM) / SQ_M_PER_ACRE;

// 4-neighbour connected groups of cells with risk >= threshold. Cells with null (tracks) never count.
export function findZones(grid, threshold, minAcres, cellAcres) {
  const rows = grid.length, cols = grid[0].length;
  const seen = new Uint8Array(rows * cols);
  const mask = new Uint8Array(rows * cols);          // 1 where a cell belongs to a zone that is big enough
  const zones = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = grid[r][c];
      if (v == null || v < threshold || seen[r * cols + c]) continue;
      const stack = [[r, c]], cells = [];
      seen[r * cols + c] = 1;
      while (stack.length) {
        const [y, x] = stack.pop();
        cells.push([y, x]);
        for (const [dy, dx] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const ny = y + dy, nx = x + dx;
          if (ny < 0 || nx < 0 || ny >= rows || nx >= cols || seen[ny * cols + nx]) continue;
          const w = grid[ny][nx];
          if (w == null || w < threshold) continue;
          seen[ny * cols + nx] = 1;
          stack.push([ny, nx]);
        }
      }
      const acres = cells.length * cellAcres;
      if (acres < minAcres) continue;
      let sumR = 0, sumC = 0, peak = 0;
      for (const [y, x] of cells) { sumR += y; sumC += x; peak = Math.max(peak, grid[y][x]); mask[y * cols + x] = 1; }
      zones.push({ cells: cells.length, acres, peak, row: sumR / cells.length + 0.5, col: sumC / cells.length + 0.5 });
    }
  }
  zones.sort((a, b) => b.acres - a.acres);
  zones.forEach((z, i) => { z.id = `P${i + 1}`; });
  return { zones, mask, rows, cols, totalAcres: zones.reduce((s, z) => s + z.acres, 0) };
}

// grid position (fractional row and column) -> [lon, lat] using the map bounds [west, south, east, north]
export function cellToLonLat(row, col, bounds, rows, cols) {
  const [west, south, east, north] = bounds;
  return [west + (col / cols) * (east - west), north - (row / rows) * (north - south)];
}

// which field polygon contains a point (ray casting); returns the feature or null
export function fieldAt(lon, lat, features) {
  for (const f of features) {
    const ring = f.geometry.coordinates[0];
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i], [xj, yj] = ring[j];
      if ((yi > lat) !== (yj > lat) && lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) inside = !inside;
    }
    if (inside) return f;
  }
  return null;
}

// outline of the zone cells: the cells of the mask that touch a cell outside the mask
export function outlineCells(mask, rows, cols) {
  const out = new Uint8Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (!mask[r * cols + c]) continue;
      const edge = r === 0 || c === 0 || r === rows - 1 || c === cols - 1 ||
        !mask[(r - 1) * cols + c] || !mask[(r + 1) * cols + c] || !mask[r * cols + c - 1] || !mask[r * cols + c + 1];
      if (edge) out[r * cols + c] = 1;
    }
  }
  return out;
}

// draw the outline as a picture, one pixel per cell (browser only)
export function outlineImage(mask, rows, cols) {
  const edge = outlineCells(mask, rows, cols);
  const canvas = document.createElement("canvas");
  canvas.width = cols; canvas.height = rows;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(cols, rows);
  for (let i = 0; i < edge.length; i++) {
    if (!edge[i]) continue;
    img.data[i * 4] = 255; img.data[i * 4 + 1] = 255; img.data[i * 4 + 2] = 255; img.data[i * 4 + 3] = 235;
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL("image/png");
}
