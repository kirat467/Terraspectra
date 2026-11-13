import { useState } from "react";
import { fmtDate } from "../lib/geo.js";

const W = 308, H = 150, M = { l: 40, r: 22, t: 10, b: 24 };
const OUT = "#f07c52", OK = "#9bd25a";

// How the simulated spectrum of the outbreak core drifts away from healthy crop, scan by scan.
export default function SignalChart({ signals, dates, index }) {
  const keys = Object.keys(signals);
  const [key, setKey] = useState(keys[0]);
  const sig = signals[key];
  const n = dates.length;
  const all = [...sig.healthy, ...sig.outbreak];
  const lo0 = Math.min(...all), hi0 = Math.max(...all), pad = (hi0 - lo0) * 0.18 || 0.01;
  const lo = lo0 - pad, hi = hi0 + pad;
  const x = (i) => M.l + (i / (n - 1)) * (W - M.l - M.r);
  const y = (v) => M.t + (1 - (v - lo) / (hi - lo)) * (H - M.t - M.b);
  const path = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const dec = sig.unit === "nm" ? 0 : 2;
  const ticks = [lo0, (lo0 + hi0) / 2, hi0];
  const h = sig.healthy[index], o = sig.outbreak[index];
  const change = ((o - h) / h) * 100;

  return (
    <div className="chart">
      <div className="seg" role="tablist" aria-label="Spectral signal">
        {keys.map((k) => (
          <button type="button" role="tab" key={k} aria-selected={k === key} className={k === key ? "on" : ""} onClick={() => setKey(k)}>{signals[k].label}</button>
        ))}
      </div>
      <div className="legend-row">
        <span><i style={{ background: OUT }} />Outbreak core</span>
        <span><i style={{ background: OK }} />Healthy crop</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${sig.label} over ${n} scans. On ${fmtDate(dates[index])}: outbreak core ${o}${sig.unit}, healthy crop ${h}${sig.unit}.`}>
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} stroke="#28332e" strokeWidth="1" />
            <text x={M.l - 6} y={y(t) + 4} textAnchor="end" className="ax">{t.toFixed(dec)}</text>
          </g>
        ))}
        <line x1={x(index)} x2={x(index)} y1={M.t} y2={H - M.b} stroke="#9bd25a" strokeWidth="1.5" strokeDasharray="3 3" />
        <path d={path(sig.healthy)} fill="none" stroke={OK} strokeWidth="2.2" strokeLinejoin="round" />
        <path d={path(sig.outbreak)} fill="none" stroke={OUT} strokeWidth="2.2" strokeLinejoin="round" />
        {dates.map((d, i) => (
          <g key={d}>
            <circle cx={x(i)} cy={y(sig.healthy[i])} r={i === index ? 5 : 3} fill={OK} stroke="#151b18" strokeWidth="2" />
            <circle cx={x(i)} cy={y(sig.outbreak[i])} r={i === index ? 5 : 3} fill={OUT} stroke="#151b18" strokeWidth="2" />
            <text x={x(i)} y={H - 7} textAnchor="middle" className={"ax" + (i === index ? " on" : "")}>{fmtDate(d).replace(/ \d{4}$/, "").replace(",", "")}</text>
          </g>
        ))}
      </svg>
      <p className="chart-note">
        {fmtDate(dates[index])}: core <b className="num" style={{ color: OUT }}>{o.toFixed(dec)}{sig.unit && ` ${sig.unit}`}</b> against <b className="num" style={{ color: OK }}>{h.toFixed(dec)}{sig.unit && ` ${sig.unit}`}</b> healthy
        (<span className="num">{change >= 0 ? "+" : "−"}{Math.abs(change).toFixed(1)}%</span>).
      </p>
      <p className="muted small">Disease lowers greenness and leaf water and moves the red edge toward blue. Simulated values; the model's real band indices replace them later.</p>
    </div>
  );
}
