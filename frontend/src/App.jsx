import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { MapView, FlyToInterpolator, LinearInterpolator } from "@deck.gl/core";
import { BitmapLayer, GeoJsonLayer, TextLayer } from "@deck.gl/layers";
import Sidebar from "./components/Sidebar.jsx";
import MapControls from "./components/MapControls.jsx";
import RiskLegend from "./components/RiskLegend.jsx";
import Timeline from "./components/Timeline.jsx";
import DemoTour from "./components/DemoTour.jsx";
import { TOUR } from "./lib/tour.js";
import { fitFarm, fitField, fmtAcres, fmtDate } from "./lib/geo.js";
import { cellAt, riskImage, riskLabel } from "./lib/risk.js";
import { acresPerCell, cellToLonLat, fieldAt, findZones, outlineImage } from "./lib/zones.js";

// ?data=real shows the real-scene folder (public/data_real) made by scripts/real_scene_to_dashboard.py
const DATASET = new URLSearchParams(window.location.search).get("data") === "real" ? "data_real" : "data";
const DATA = `${import.meta.env.BASE_URL}${DATASET}/`;
const VIEWS = [new MapView({ id: "map", repeat: false })];
const DEFAULT_LIMITS = { minZoom: 11, maxZoom: 19 };
const CONTROLLER = { dragRotate: false, touchRotate: false, keyboard: false };
const getJson = (url) => fetch(url).then((r) => { if (!r.ok) throw new Error(`${url} (${r.status})`); return r.json(); });

export default function App() {
  const [meta, setMeta] = useState(null);
  const [fieldData, setFieldData] = useState(null);
  const [scans, setScans] = useState(null);       // one risk map per scan date
  const [index, setIndex] = useState(0);          // which scan is showing
  const [history, setHistory] = useState(null);   // risk_index.json: acres over time and spectral signals
  const [error, setError] = useState(null);
  const [viewState, setViewState] = useState(null);
  const [selected, setSelected] = useState(null);
  const [hover, setHover] = useState(null);          // { name, x, y }
  const [cursor, setCursor] = useState(null);
  const [zoneCfg, setZoneCfg] = useState({ threshold: 0.6, minAcres: 5, outline: true });   // priority zones: risk level and smallest area
  const [layers, setLayers] = useState({ risk: true, riskOpacity: 0.85, imagery: true, opacity: 1, boundaries: true, labels: true });
  const [tour, setTour] = useState(null);           // index of the demo step, null when the demo is off
  const mapBox = useRef(null);
  const size = useRef({ width: 900, height: 600 });
  const ready = useRef(false);
  const limits = useRef(DEFAULT_LIMITS);          // minimum zoom adapts to the size of the scene (a 30 km scene needs a wider view than a farm)

  // load the farm files and the latest risk map
  useEffect(() => {
    Promise.all([getJson(`${DATA}farm_meta.json`), getJson(`${DATA}farm_fields.geojson`), getJson(`${DATA}risk/risk_index.json`)])
      .then(async ([m, f, idx]) => {
        const maps = await Promise.all(idx.dates.map((d) => getJson(`${DATA}risk/risk_${d}.json`)));
        setMeta(m); setFieldData(f); setHistory(idx); setScans(maps); setIndex(maps.length - 1);
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
        const fit = fitFarm(meta, size.current.width, size.current.height);
        limits.current = { minZoom: Math.min(DEFAULT_LIMITS.minZoom, fit.zoom - 1), maxZoom: DEFAULT_LIMITS.maxZoom };
        setViewState({ ...fit, ...limits.current });
      }
    });
    ro.observe(mapBox.current);
    return () => ro.disconnect();
  }, [meta]);

  const fly = useCallback((target) => {
    setViewState((vs) => ({ ...vs, ...target, ...limits.current, transitionDuration: 900, transitionInterpolator: new FlyToInterpolator({ speed: 1.6 }) }));
  }, []);

  const showWholeFarm = useCallback(() => {
    setSelected(null);
    fly(fitFarm(meta, size.current.width, size.current.height));
  }, [meta, fly]);

  const zoomBy = useCallback((d) => {
    setViewState((vs) => ({ ...vs, zoom: Math.min(limits.current.maxZoom, Math.max(limits.current.minZoom, vs.zoom + d)), transitionDuration: 250, transitionInterpolator: new LinearInterpolator(["zoom"]) }));
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


  const risk = scans ? scans[index] : null;
  const images = useMemo(() => (scans ? scans.map((m) => riskImage(m.risk)) : null), [scans]);   // drawn once, so stepping through dates is instant
  const image = images ? images[index] : null;

  // priority zones from the current risk map and the sliders
  const zoneInfo = useMemo(() => (risk && meta ? findZones(risk.risk, zoneCfg.threshold, zoneCfg.minAcres, acresPerCell(meta.pixel_size_m)) : null),
    [risk, meta, zoneCfg.threshold, zoneCfg.minAcres]);
  const zoneList = useMemo(() => {
    if (!zoneInfo) return [];
    return zoneInfo.zones.map((z) => {
      const center = cellToLonLat(z.row, z.col, risk.bounds, risk.rows, risk.cols);
      const f = fieldAt(center[0], center[1], features);
      return { ...z, center, field: f?.properties.name ?? null, crop: f?.properties.crop ?? "" };
    });
  }, [zoneInfo, risk, features]);
  const zoneImage = useMemo(() => (zoneInfo && zoneInfo.zones.length ? outlineImage(zoneInfo.mask, zoneInfo.rows, zoneInfo.cols) : null), [zoneInfo]);
  const tourCtx = useMemo(() => ({ meta, scans, zones: zoneList, setIndex, setLayers, reset: showWholeFarm, flyToZone }), [meta, scans, zoneList, showWholeFarm, flyToZone]);
  const canTour = !!scans && scans.length > 1 && features.length > 0 && risk?.kind !== "stress_index";
  const goStep = useCallback((i) => { setTour(i); TOUR[i].run(tourCtx); }, [tourCtx]);
  const exitTour = useCallback(() => { setTour(null); setLayers((l) => ({ ...l, risk: true })); }, []);
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
    if (risk && zoneImage) {
      list.push(new BitmapLayer({
        id: "zones", image: zoneImage, bounds: risk.bounds, visible: zoneCfg.outline,
        textureParameters: { minFilter: "nearest", magFilter: "nearest" },
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
  }, [meta, fieldData, features, risk, image, zoneImage, zoneCfg.outline, layers, selected, hover?.name, detailed]);

  if (error) {
    return (
      <div className="state">
        <h1>TerraSpectra</h1>
        <p>The farm files could not be loaded ({error}).</p>
        <p className="muted">Run <code>python frontend/scripts/make_mock_scene.py</code> and then <code>python frontend/scripts/make_mock_risk.py</code> from the repository root, then reload.</p>
      </div>
    );
  }
  if (!meta || !risk || !history) return <div className="state"><h1>TerraSpectra</h1><p className="muted">Loading farm data…</p></div>;

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
          {canTour && tour === null && <button className="demo-btn" onClick={() => goStep(0)}>&#9654; Demo</button>}
          <span className="chip">Scan <b className="num">{fmtDate(risk.date)}</b></span>
          {risk.kind !== "stress_index" && <span className="chip">Forecast <b className="num">next {risk.horizon_days} days</b></span>}
          <span className={"chip" + (meta.simulated === false ? "" : " warn")}>{meta.simulated === false ? (meta.source ?? "Real scene") : "Simulated data"}</span>
        </div>
      </header>

      <Sidebar meta={meta} fields={features} risk={risk} prevRisk={index > 0 ? scans[index - 1] : null} history={history} index={index} onIndex={setIndex} selected={selected} hovered={hover?.name} onSelect={select} onZone={flyToZone} zoneCfg={zoneCfg} setZoneCfg={setZoneCfg} zoneList={zoneList} layers={layers} setLayers={setLayers} />

      <main className="map" ref={mapBox}>
        {viewState && (
          <DeckGL
            views={VIEWS}
            viewState={viewState}
            onViewStateChange={({ viewState: vs }) => setViewState({ ...vs, ...limits.current })}
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
        {viewState && layers.risk && <RiskLegend horizon={risk.horizon_days} title={risk.legend_title} />}
        {viewState && <MapControls simulated={meta.simulated !== false} source={meta.source} viewState={viewState} cursor={cursor} cursorRisk={hover ? cell?.value : null} onZoom={zoomBy} onReset={showWholeFarm} />}
        {tour !== null && canTour && <DemoTour steps={TOUR} step={tour} text={TOUR[tour].text(tourCtx)} onStep={goStep} onExit={exitTour} />}
        {hoverFeature && (
          <div className="tooltip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
            <b>{hoverFeature.properties.name}</b>
            <span>{hoverFeature.properties.crop} · {fmtAcres(hoverFeature.properties.acres)} acres</span>
            {cell && <span>Risk here <b className="num">{cell.value.toFixed(2)}</b> ({riskLabel(cell.value)})</span>}
            {hoverStats && <span className="muted">Field peak {hoverStats.peak_risk.toFixed(2)} · {fmtAcres(hoverStats.acres_high)} ac high risk</span>}
          </div>
        )}
      </main>

      <Timeline dates={history.dates} index={index} onIndex={setIndex} />
    </div>
  );
}
