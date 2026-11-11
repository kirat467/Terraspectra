import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { MapView, FlyToInterpolator, LinearInterpolator } from "@deck.gl/core";
import { BitmapLayer, GeoJsonLayer, TextLayer } from "@deck.gl/layers";
import Sidebar from "./components/Sidebar.jsx";
import MapControls from "./components/MapControls.jsx";
import { fitFarm, fitField, fmtAcres, fmtDate } from "./lib/geo.js";

const DATA = `${import.meta.env.BASE_URL}data/`;
const VIEWS = [new MapView({ id: "map", repeat: false })];
const LIMITS = { minZoom: 11, maxZoom: 19 };
const CONTROLLER = { dragRotate: false, touchRotate: false, keyboard: false };

export default function App() {
  const [meta, setMeta] = useState(null);
  const [fieldData, setFieldData] = useState(null);
  const [error, setError] = useState(null);
  const [viewState, setViewState] = useState(null);
  const [selected, setSelected] = useState(null);
  const [hover, setHover] = useState(null);          // { name, x, y }
  const [cursor, setCursor] = useState(null);
  const [layers, setLayers] = useState({ imagery: true, boundaries: true, labels: true, opacity: 1 });
  const mapBox = useRef(null);
  const size = useRef({ width: 900, height: 600 });
  const ready = useRef(false);

  // load the farm files
  useEffect(() => {
    Promise.all([
      fetch(`${DATA}farm_meta.json`).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }),
      fetch(`${DATA}farm_fields.geojson`).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }),
    ]).then(([m, f]) => { setMeta(m); setFieldData(f); }).catch((e) => setError(String(e.message || e)));
  }, []);

  // keep track of the map size, and frame the whole farm the first time we know it
  useEffect(() => {
    if (!mapBox.current) return;
    const ro = new ResizeObserver(([entry]) => {
      size.current = { width: entry.contentRect.width, height: entry.contentRect.height };
      if (meta && !ready.current && size.current.width > 0) {
        ready.current = true;
        setViewState({ ...fitFarm(meta, size.current.width, size.current.height), ...LIMITS });
      }
    });
    ro.observe(mapBox.current);
    return () => ro.disconnect();
  }, [meta]);

  const fly = useCallback((target) => {
    setViewState((vs) => ({ ...vs, ...target, ...LIMITS, transitionDuration: 900, transitionInterpolator: new FlyToInterpolator({ speed: 1.6 }) }));
  }, []);

  const showWholeFarm = useCallback(() => {
    setSelected(null);
    fly(fitFarm(meta, size.current.width, size.current.height));
  }, [meta, fly]);

  const zoomBy = useCallback((d) => {
    setViewState((vs) => ({ ...vs, zoom: Math.min(LIMITS.maxZoom, Math.max(LIMITS.minZoom, vs.zoom + d)), transitionDuration: 250, transitionInterpolator: new LinearInterpolator(["zoom"]) }));
  }, []);

  const features = fieldData?.features ?? [];
  const select = useCallback((name) => {
    setSelected(name);
    const f = features.find((x) => x.properties.name === name);
    if (f) fly(fitField(f, size.current.width, size.current.height));
  }, [features, fly]);

  const detailed = viewState ? viewState.zoom >= 15.3 : false;
  const deckLayers = useMemo(() => {
    if (!meta || !fieldData) return [];
    return [
      new BitmapLayer({ id: "imagery", image: `${DATA}farm_rgb.png`, bounds: meta.bounds, opacity: layers.opacity, visible: layers.imagery }),
      new GeoJsonLayer({
        id: "fields", data: fieldData, pickable: true, stroked: true, filled: true, visible: layers.boundaries,
        lineWidthUnits: "pixels", lineWidthMinPixels: 1,
        getFillColor: (f) => (f.properties.name === selected ? [155, 210, 90, 46] : f.properties.name === hover?.name ? [255, 255, 255, 30] : [255, 255, 255, 2]),
        getLineColor: (f) => (f.properties.name === selected ? [155, 210, 90, 255] : f.properties.name === hover?.name ? [255, 255, 255, 255] : [255, 255, 255, 150]),
        getLineWidth: (f) => (f.properties.name === selected ? 3 : 1.5),
        updateTriggers: { getFillColor: [selected, hover?.name], getLineColor: [selected, hover?.name], getLineWidth: [selected] },
      }),
      new TextLayer({
        id: "labels", data: features, visible: layers.labels, pickable: false,
        getPosition: (f) => f.properties.center,
        getText: (f) => (detailed ? `${f.properties.name.replace("Field ", "")}\n${f.properties.crop}` : f.properties.name.replace("Field ", "")),
        getSize: 13, sizeUnits: "pixels", fontWeight: 700, fontFamily: '"Segoe UI", system-ui, -apple-system, Roboto, sans-serif', getColor: [245, 250, 245, 255],
        background: true, getBackgroundColor: [14, 19, 17, 200], backgroundPadding: [7, 4], getBorderColor: [255, 255, 255, 60], getBorderWidth: 1,
        updateTriggers: { getText: [detailed] },
      }),
    ];
  }, [meta, fieldData, features, layers, selected, hover?.name, detailed]);

  if (error) {
    return (
      <div className="state">
        <h1>TerraSpectra</h1>
        <p>The farm files could not be loaded ({error}).</p>
        <p className="muted">Run <code>python frontend/scripts/make_mock_scene.py</code> from the repository root, then reload.</p>
      </div>
    );
  }
  if (!meta) return <div className="state"><h1>TerraSpectra</h1><p className="muted">Loading farm data…</p></div>;

  const hoverFeature = hover ? features.find((f) => f.properties.name === hover.name) : null;
  const latestScan = meta.scan_dates[meta.scan_dates.length - 1];

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="mark" aria-hidden="true"><svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M2 13 7 7l3.5 3.5L16 3" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" /></svg></span>
          <div><b>TerraSpectra</b><small>Hyperspectral crop disease forecasting</small></div>
        </div>
        <div className="top-right">
          <span className="chip">Latest scan <b className="num">{fmtDate(latestScan)}</b></span>
          <span className="chip warn">Simulated data</span>
        </div>
      </header>

      <Sidebar meta={meta} fields={features} selected={selected} hovered={hover?.name} onSelect={select} layers={layers} setLayers={setLayers} />

      <main className="map" ref={mapBox}>
        {viewState && (
          <DeckGL
            views={VIEWS}
            viewState={viewState}
            onViewStateChange={({ viewState: vs }) => setViewState({ ...vs, ...LIMITS })}
            controller={CONTROLLER}
            layers={deckLayers}
            getCursor={({ isDragging, isHovering }) => (isDragging ? "grabbing" : isHovering ? "pointer" : "grab")}
            onHover={(info) => {
              setHover(info.object ? { name: info.object.properties.name, x: info.x, y: info.y } : null);
              if (info.coordinate) setCursor(info.coordinate);
            }}
            onClick={(info) => select(info.object && info.object.properties.name !== selected ? info.object.properties.name : null)}
          />
        )}
        {viewState && <MapControls viewState={viewState} cursor={cursor} onZoom={zoomBy} onReset={showWholeFarm} />}
        {hoverFeature && (
          <div className="tooltip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
            <b>{hoverFeature.properties.name}</b>
            <span>{hoverFeature.properties.crop} · {fmtAcres(hoverFeature.properties.acres)} acres</span>
            <span className="muted">Risk score: not loaded yet</span>
          </div>
        )}
      </main>
    </div>
  );
}
