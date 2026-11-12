import { fmtAcres, fmtDate } from "../lib/geo.js";
import { riskLabel, riskSolid, riskText } from "../lib/risk.js";

export default function Sidebar({ meta, fields, risk, selected, hovered, onSelect, onZone, layers, setLayers }) {
  const stats = Object.fromEntries((risk?.fields ?? []).map((f) => [f.name, f]));
  const sorted = [...fields].sort((a, b) => (stats[b.properties.name]?.peak_risk ?? 0) - (stats[a.properties.name]?.peak_risk ?? 0));
  const active = fields.find((f) => f.properties.name === selected);
  const activeStats = active ? stats[active.properties.name] : null;
  const s = risk?.summary;

  return (
    <aside className="sidebar" aria-label="Farm overview">
      {risk && (
        <section className="panel risk-panel">
          <h3>Predicted outbreak risk</h3>
          <p className="muted small">Chance that fungal blight shows up in the next {risk.horizon_days} days, from the scan of {fmtDate(risk.date)}.</p>
          <dl className="stats">
            <div><dt>High-risk area, acres</dt><dd className="num bad">{fmtAcres(s.acres_high)}</dd></div>
            <div><dt>Share of farm</dt><dd className="num">{((s.acres_high / meta.farm_acres) * 100).toFixed(1)}<span>%</span></dd></div>
            <div><dt>Peak risk</dt><dd className="num" style={{ color: riskText(s.peak_risk) }}>{s.peak_risk.toFixed(2)}</dd></div>
            <div><dt>Moderate, acres</dt><dd className="num">{fmtAcres(s.acres_moderate)}</dd></div>
          </dl>
          <h4>Risk zones</h4>
          {risk.zones.length === 0 && <p className="muted small">No zone above the high-risk level yet.</p>}
          <ul className="zones">
            {risk.zones.map((z) => (
              <li key={z.id}>
                <button type="button" className="zone-row" onClick={() => onZone(z)}>
                  <span className="zone-id" style={{ background: riskSolid(z.peak_risk) }}>{z.id}</span>
                  <span className="fname">{z.field}<small>{z.crop}, peak {z.peak_risk.toFixed(2)}</small></span>
                  <span className="num">{fmtAcres(z.acres)} ac</span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="panel">
        <h2>{meta.name}</h2>
        <p className="muted small">Hyperspectral scan of a 1,000-acre farm: {meta.bands} bands, {meta.pixel_size_m} m pixels. All imagery on this page is simulated.</p>
      </section>

      <section className="panel">
        <h3>Layers</h3>
        <label className="toggle"><input type="checkbox" checked={layers.risk} onChange={(e) => setLayers({ ...layers, risk: e.target.checked })} /><span>Disease risk heatmap</span></label>
        <label className="slider" style={{ opacity: layers.risk ? 1 : 0.45 }}>
          <span>Heatmap opacity</span>
          <input type="range" min="20" max="100" value={Math.round(layers.riskOpacity * 100)} disabled={!layers.risk} onChange={(e) => setLayers({ ...layers, riskOpacity: e.target.value / 100 })} aria-label="Heatmap opacity" />
        </label>
        <label className="toggle"><input type="checkbox" checked={layers.imagery} onChange={(e) => setLayers({ ...layers, imagery: e.target.checked })} /><span>Satellite imagery</span></label>
        <label className="slider" style={{ opacity: layers.imagery ? 1 : 0.45 }}>
          <span>Imagery opacity</span>
          <input type="range" min="20" max="100" value={Math.round(layers.opacity * 100)} disabled={!layers.imagery} onChange={(e) => setLayers({ ...layers, opacity: e.target.value / 100 })} aria-label="Imagery opacity" />
        </label>
        <label className="toggle"><input type="checkbox" checked={layers.boundaries} onChange={(e) => setLayers({ ...layers, boundaries: e.target.checked })} /><span>Field boundaries</span></label>
        <label className="toggle"><input type="checkbox" checked={layers.labels} onChange={(e) => setLayers({ ...layers, labels: e.target.checked })} /><span>Field labels</span></label>
      </section>

      <section className="panel">
        <h3>Fields, highest risk first</h3>
        <ul className="fields" role="listbox" aria-label="Fields">
          {sorted.map((f) => {
            const p = f.properties, st = stats[p.name];
            return (
              <li key={p.name}>
                <button type="button" role="option" aria-selected={selected === p.name}
                  className={"field-row" + (selected === p.name ? " selected" : "") + (hovered === p.name ? " hover" : "")}
                  onClick={() => onSelect(selected === p.name ? null : p.name)}>
                  <i style={{ background: meta.crops[p.crop] }} />
                  <span className="fname">{p.name}<small>{p.crop}, {fmtAcres(p.acres)} ac</small></span>
                  {st && <span className="risk-chip num" style={{ background: riskSolid(st.peak_risk) }} title={`Peak risk ${st.peak_risk.toFixed(2)}`}>{st.peak_risk.toFixed(2)}</span>}
                </button>
              </li>
            );
          })}
        </ul>
        {active && activeStats && (
          <div className="detail">
            <b>{active.properties.name}, {active.properties.crop}</b>
            <span>{fmtAcres(active.properties.acres)} acres</span>
            <span>Peak risk <b className="num">{activeStats.peak_risk.toFixed(2)}</b> ({riskLabel(activeStats.peak_risk)}), average <b className="num">{activeStats.mean_risk.toFixed(2)}</b></span>
            <span>{fmtAcres(activeStats.acres_high)} acres at high risk, {fmtAcres(activeStats.acres_moderate)} at moderate risk</span>
          </div>
        )}
      </section>

      <section className="panel">
        <h3>Scan dates</h3>
        <ul className="dates">
          {meta.scan_dates.map((d) => <li key={d} className={"num" + (risk && d === risk.date ? " current" : "")}>{fmtDate(d)}</li>)}
        </ul>
        <p className="muted small">One simulated scan per week. A timeline slider will step through them.</p>
      </section>
    </aside>
  );
}
