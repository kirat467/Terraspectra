// Tests for src/lib/zones.js. Run from the frontend folder:  node scripts/test_zones.mjs
import assert from "node:assert/strict";
import test from "node:test";
import { acresPerCell, cellToLonLat, fieldAt, findZones, outlineCells } from "../src/lib/zones.js";

const grid = (rows, cols, fill = 0.1) => Array.from({ length: rows }, () => Array(cols).fill(fill));
const paint = (g, r0, r1, c0, c1, v) => { for (let r = r0; r < r1; r++) for (let c = c0; c < c1; c++) g[r][c] = v; };
const cell = acresPerCell(15);           // 0.0556 acres per 15 m cell

test("acres per cell", () => assert.ok(Math.abs(cell - 0.05560) < 1e-4));

test("finds two separate zones and sorts by size", () => {
  const g = grid(40, 40);
  paint(g, 2, 12, 2, 12, 0.9);      // 100 cells
  paint(g, 25, 31, 25, 31, 0.8);    // 36 cells
  const { zones, totalAcres } = findZones(g, 0.6, 0, cell);
  assert.equal(zones.length, 2);
  assert.deepEqual(zones.map((z) => z.cells), [100, 36]);
  assert.deepEqual(zones.map((z) => z.id), ["P1", "P2"]);
  assert.ok(Math.abs(totalAcres - 136 * cell) < 1e-9);
  assert.equal(zones[0].peak, 0.9);
  assert.ok(Math.abs(zones[0].row - 7) < 1e-9 && Math.abs(zones[0].col - 7) < 1e-9);
});

test("minimum area drops small zones and keeps them out of the mask", () => {
  const g = grid(40, 40);
  paint(g, 2, 12, 2, 12, 0.9);      // 100 cells = 5.56 acres
  paint(g, 25, 28, 25, 28, 0.9);    // 9 cells = 0.5 acres
  const { zones, mask } = findZones(g, 0.6, 5, cell);
  assert.equal(zones.length, 1);
  assert.equal(mask.reduce((a, b) => a + b, 0), 100);
  assert.equal(mask[26 * 40 + 26], 0);
});

test("threshold changes what counts, tracks (null) never count", () => {
  const g = grid(10, 10, null);
  paint(g, 0, 5, 0, 5, 0.5);
  assert.equal(findZones(g, 0.6, 0, cell).zones.length, 0);
  assert.equal(findZones(g, 0.4, 0, cell).zones[0].cells, 25);
  g[2][2] = null;                                    // a hole splits nothing but is not counted
  assert.equal(findZones(g, 0.4, 0, cell).zones[0].cells, 24);
});

test("cells touching only diagonally are separate zones", () => {
  const g = grid(6, 6);
  g[1][1] = 0.9; g[2][2] = 0.9;
  assert.equal(findZones(g, 0.6, 0, cell).zones.length, 2);
});

test("outline keeps the border of a block and drops the inside", () => {
  const rows = 12, cols = 12, mask = new Uint8Array(rows * cols);
  for (let r = 3; r < 8; r++) for (let c = 3; c < 8; c++) mask[r * cols + c] = 1;
  const edge = outlineCells(mask, rows, cols);
  assert.equal(edge.reduce((a, b) => a + b, 0), 16);       // 5x5 block: 25 cells - 9 inner cells
  assert.equal(edge[5 * cols + 5], 0);
  assert.equal(edge[3 * cols + 3], 1);
});

test("cellToLonLat maps the corners", () => {
  const b = [76, 12, 77, 13];
  assert.deepEqual(cellToLonLat(0, 0, b, 100, 100), [76, 13]);
  assert.deepEqual(cellToLonLat(100, 100, b, 100, 100), [77, 12]);
  assert.deepEqual(cellToLonLat(50, 50, b, 100, 100), [76.5, 12.5]);
});

test("fieldAt finds the field that contains a point", () => {
  const sq = (x0, y0, x1, y1, name) => ({ properties: { name }, geometry: { coordinates: [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]] } });
  const fields = [sq(0, 0, 1, 1, "A"), sq(2, 0, 3, 1, "B")];
  assert.equal(fieldAt(2.5, 0.5, fields).properties.name, "B");
  assert.equal(fieldAt(1.5, 0.5, fields), null);
});
