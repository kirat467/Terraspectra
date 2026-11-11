import { scaleBar } from "../lib/geo.js";

export default function MapControls({ viewState, cursor, onZoom, onReset }) {
  const bar = scaleBar(viewState.zoom, viewState.latitude);
  return (
    <>
      <div className="map-buttons">
        <button type="button" onClick={() => onZoom(1)} aria-label="Zoom in">+</button>
        <button type="button" onClick={() => onZoom(-1)} aria-label="Zoom out">−</button>
        <button type="button" onClick={onReset} aria-label="Show whole farm" title="Show whole farm">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"><path d="M2 6V2h4M10 2h4v4M14 10v4h-4M6 14H2v-4" /></svg>
        </button>
      </div>
      <div className="map-readout">
        <div className="scalebar" style={{ width: bar.px }}><span>{bar.label}</span></div>
        <div className="coords num">{cursor ? `${cursor[1].toFixed(5)}° N  ${cursor[0].toFixed(5)}° E` : "Move the pointer over the map"}</div>
      </div>
      <div className="sim-badge">Simulated data</div>
    </>
  );
}
