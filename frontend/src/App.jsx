import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { MapView, FlyToInterpolator, LinearInterpolator } from "@deck.gl/core";
import { BitmapLayer, GeoJsonLayer, TextLayer } from "@deck.gl/layers";
import Sidebar from "./components/Sidebar.jsx";
import MapControls from "./components/MapControls.jsx";
import RiskLegend from "./components/RiskLegend.jsx";
import { fitFarm, fitField, fmtAcres, fmtDate } from "./lib/geo.js";
import { cellAt, riskImage, riskLabel } from "./lib/risk.js";

const DATA = `${import.meta.env.BASE_URL}data/`;
const VIEWS = [new MapView({ id: "map", repeat: false })];
const LIMITS = { minZoom: 11, maxZoom: 19 };
const CONTROLLER = { dragRotate: false, touchRotate: false, keyboard: false };
const getJson = (url) => fetch(url).then((r) => { if (!r.ok) throw new Error(`${url} (${r.status})`); return r.json(); });

export default function App() {
  const [meta, setMeta] = useState(null);
  const [fieldData, setFieldData] = useState(null);
  const [risk, setRisk] = useState(null);
  const [error, setError] = useState(null);
  const [viewState, setViewState] = useState(null);
  const [selected, setSelected] = useState(null);
  const [hover, setHover] = useState(null);          // { name, x, y }
  const [cursor, setCursor] = useState(null);
  const [layers, setLayers] = useState({ risk: true, riskOpacity: 0.85, imagery: true, opacity: 1, boundaries: true, labels: true });
  const mapBox = useRef(null);
  const size = useRef({ width: 900, height: 600 });
  const ready = useRef(false);

  // load the farm files and the latest risk map
  useEffect(() => {
    Promise.all([getJson(`${DATA}farm_meta.json`), getJson(`${DATA}farm_fields.geojson`), getJson(`${DATA}risk/risk_index.json`)])
      .then(async ([m, f, idx]) => {
        const latest = await getJson(`${DATA}risk/risk_${idx.latest}.json`);
        setMeta(m); setFieldData(f); setRisk(latest);
      })
      .catch((e) => setError(String(e.message || e)));
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

  const flyToZone = useCallback((z) => {
    setSelected(z.field);
    fly({ longitude: z.center[0], latitude: z.center[1], zoom: 16.8, pitch: 0, bearing: 0 });
  }, [fly]);

  const image = useMemo(() => (risk ? riskImage(risk.risk) : null), [risk]);
  const cell = cursor ? cellAt(risk, cursor[0], cursor[1]) : null;
  const detailed = viewState ? viewState.zoom >= 15.3 : false;

  const deckLayers = useMemo(() => {
    if (!meta || !fieldData) return [];
    const list = [
      new BitmapLayer({ id: "imagery", image: `${DATA}farm_rgb.png`, bounds: meta.bounds, opacity: layers.opacity, visible: layers.imagery }),
    ];
    if (risk && image) {
      list.push(new BitmapLayer({
        id: "risk", image, bounds: risk.bounds, opacity: layers.riskOpacity, visible: layers.risk,
        textureParameters: { minFilter: "nearest", magFilter: "nearest" },      // keep the cells square
      }));
    }
    list.push(
      new GeoJsonLayer({
        id: "fields", data: fieldData, pickable: true, stroked: true, filled: true, visible: layers.boundaries,
        lineWidthUnits: "pixels", lineWidthMinPixels: 1,
        getFillColor: (f) => (f.properties.name === selected ? [255, 255, 255, 22] : f.properties.name === hover?.name ? [255, 255, 255, 30] : [255, 255, 255, 2]),
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
    );
    return list;
  }, [meta, fieldData, features, risk, image, layers, selected, hover?.name, detailed]);

  if (error) {
    return (
      <div className="state">
        <h1>TerraSpectra</h1>
        <p>The farm files could not be loaded ({error}).</p>
        <p className="muted">Run <code>python frontend/scripts/make_mock_scene.py</code> and then <code>python frontend/scripts/make_mock_risk.py</code> from the repository root, then reload.</p>
      </div>
    );
  }
  if (!meta || !risk) return <div className="state"><h1>TerraSpectra</h1><p className="muted">Loading farm data…</p></div>;

  const hoverFeature = hover ? features.find((f) => f.properties.name === hover.name) : null;
  const hoverStats = hoverFeature ? risk.fields.find((f) => f.name === hoverFeature.properties.name) : null;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="mark" aria-hidden="true"><svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M2 13 7 7l3.5 3.5L16 3" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" /></svg></span>
          <div><b>TerraSpectra</b><small>Hyperspectral crop disease forecasting</small></div>
        </div>
        <div className="top-right">
          <span className="chip">Scan <b className="num">{fmtDate(risk.date)}</b></span>
          <span className="chip">Forecast <b className="num">next {risk.horizon_days} days</b></span>
          <span className="chip warn">Simulated data</span>
        </div>
      </header>

      <Sidebar meta={meta} fields={features} risk={risk} selected={selected} hovered={hover?.name} onSelect={select} onZone={flyToZone} layers={layers} setLayers={setLayers} />

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
        {viewState && layers.risk && <RiskLegend horizon={risk.horizon_days} />}
        {viewState && <MapControls viewState={viewState} cursor={cursor} cursorRisk={hover ? cell?.value : null} onZoom={zoomBy} onReset={showWholeFarm} />}
        {hoverFeature && (
          <div className="tooltip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
            <b>{hoverFeature.properties.name}</b>
            <span>{hoverFeature.properties.crop} · {fmtAcres(hoverFeature.properties.acres)} acres</span>
            {cell && <span>Risk here <b className="num">{cell.value.toFixed(2)}</b> ({riskLabel(cell.value)})</span>}
            {hoverStats && <span className="muted">Field peak {hoverStats.peak_risk.toFixed(2)} · {fmtAcres(hoverStats.acres_high)} ac high risk</span>}
          </div>
        )}
      </main>
    </div>
  );
}
