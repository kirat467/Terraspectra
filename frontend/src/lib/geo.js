// Small map helpers: where to point the camera, and how long a scale bar is.

// metres covered by one screen pixel at a zoom level (deck.gl uses 512 px tiles)
export function metresPerPixel(zoom, latitude) {
  return (78271.517 * Math.cos((latitude * Math.PI) / 180)) / Math.pow(2, zoom);
}

// camera that shows the whole farm with a little breathing room
export function fitFarm(meta, width, height) {
  const [west, south, east, north] = meta.bounds;
  const lat = (south + north) / 2;
  const widthM = (east - west) * 111320 * Math.cos((lat * Math.PI) / 180);
  const heightM = (north - south) * 111320;
  const usable = Math.max(200, Math.min(width, height) - 48);
  const zoom = Math.log2((78271.517 * Math.cos((lat * Math.PI) / 180) * usable) / Math.max(widthM, heightM));
  return { longitude: (west + east) / 2, latitude: lat, zoom, pitch: 0, bearing: 0 };
}

// camera that frames one field (zoomed so the field fills about 70% of the map)
export function fitField(feature, width, height) {
  const ring = feature.geometry.coordinates[0];
  const lons = ring.map((p) => p[0]);
  const lats = ring.map((p) => p[1]);
  const west = Math.min(...lons), east = Math.max(...lons), south = Math.min(...lats), north = Math.max(...lats);
  const lat = (south + north) / 2;
  const widthM = (east - west) * 111320 * Math.cos((lat * Math.PI) / 180);
  const heightM = (north - south) * 111320;
  const usable = Math.max(200, Math.min(width, height) * 0.7);
  const zoom = Math.log2((78271.517 * Math.cos((lat * Math.PI) / 180) * usable) / Math.max(widthM, heightM));
  return { longitude: (west + east) / 2, latitude: lat, zoom, pitch: 0, bearing: 0 };
}

// pick a round scale-bar length that is between 60 and 140 px wide
export function scaleBar(zoom, latitude) {
  const mpp = metresPerPixel(zoom, latitude);
  const steps = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
  const metres = steps.find((m) => m / mpp >= 60) ?? steps[steps.length - 1];
  return { metres, px: metres / mpp, label: metres >= 1000 ? `${metres / 1000} km` : `${metres} m` };
}

export const fmtAcres = (n) => n.toLocaleString("en-IN", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
export const fmtDate = (iso) =>
  new Date(iso + "T00:00:00").toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });

export function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
