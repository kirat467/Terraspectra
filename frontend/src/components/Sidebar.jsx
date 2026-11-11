import { fmtAcres, fmtDate } from "../lib/geo.js";

export default function Sidebar({ meta, fields, selected, hovered, onSelect, layers, setLayers }) {
  const active = fields.find((f) => f.properties.name === selected);
  const share = active ? (active.properties.acres / meta.farm_acres) * 100 : null;

  return (
    <aside className="sidebar" aria-label="Farm overview">
      <section className="panel">
        <h2>{meta.name}</h2>
        <p className="muted">Hyperspectral scan of a 1,000-acre farm. All imagery on this page is simulated.</p>
        <dl className="stats">
          <div><dt>Farm area, acres</dt><dd className="num">{fmtAcres(meta.farm_acres)}</dd></div>
          <div><dt>Fields</dt><dd className="num">{fields.length}</dd></div>
          <div><dt>Resolution</dt><dd className="num">{meta.pixel_size_m}<span> m</span></dd></div>
          <div><dt>Spectral bands</dt><dd className="num">{meta.bands}</dd></div>
        </dl>
      </section>

      <section className="panel">
        <h3>Layers</h3>
        <label className="toggle"><input type="checkbox" checked={layers.imagery} onChange={(e) => setLayers({ ...layers, imagery: e.target.checked })} /><span>Satellite imagery</span></label>
        <label className="slider" style={{ opacity: layers.imagery ? 1 : 0.45 }}>
          <span>Imagery opacity</span>
          <input type="range" min="20" max="100" value={Math.round(layers.opacity * 100)} disabled={!layers.imagery} onChange={(e) => setLayers({ ...layers, opacity: e.target.value / 100 })} aria-label="Imagery opacity" />
        </label>
        <label className="toggle"><input type="checkbox" checked={layers.boundaries} onChange={(e) => setLayers({ ...layers, boundaries: e.target.checked })} /><span>Field boundaries</span></label>
        <label className="toggle"><input type="checkbox" checked={layers.labels} onChange={(e) => setLayers({ ...layers, labels: e.target.checked })} /><span>Field labels</span></label>
        <label className="toggle off"><input type="checkbox" disabled /><span>Disease risk heatmap <em>(added next)</em></span></label>
      </section>

      <section className="panel">
        <h3>Fields</h3>
        <ul className="fields" role="listbox" aria-label="Fields">
          {fields.map((f) => {
            const p = f.properties;
            return (
              <li key={p.name}>
                <button type="button" role="option" aria-selected={selected === p.name}
                  className={"field-row" + (selected === p.name ? " selected" : "") + (hovered === p.name ? " hover" : "")}
                  onClick={() => onSelect(selected === p.name ? null : p.name)}>
                  <i style={{ background: meta.crops[p.crop] }} />
                  <span className="fname">{p.name}<small>{p.crop}</small></span>
                  <span className="num">{fmtAcres(p.acres)} ac</span>
                </button>
              </li>
            );
          })}
        </ul>
        {active && (
          <div className="detail">
            <b>{active.properties.name}</b>
            <span>{active.properties.crop}, {fmtAcres(active.properties.acres)} acres ({share.toFixed(1)}% of the farm)</span>
            <span className="muted">Risk score: not loaded yet</span>
          </div>
        )}
      </section>

      <section className="panel">
        <h3>Scan dates</h3>
        <ul className="dates">
          {meta.scan_dates.map((d) => <li key={d} className="num">{fmtDate(d)}</li>)}
        </ul>
        <p className="muted small">One simulated scan per week. A timeline slider will step through them.</p>
      </section>
    </aside>
  );
}
