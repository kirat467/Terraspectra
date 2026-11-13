import { fmtAcres, fmtDate } from "../lib/geo.js";

const W = 308, H = 168, M = { l: 34, r: 22, t: 22, b: 26 };
const HIGH = "#f07c52", MOD = "#f5c04a";

function niceMax(v) {
  const steps = [5, 10, 20, 25, 50, 100, 200];
  return steps.find((s) => s >= v * 1.1) ?? Math.ceil(v / 100) * 100;
}

// Acres at high and moderate risk at every scan. Click a point to jump to that scan.
export default function TrendChart({ history, index, onIndex }) {
  const n = history.length;
  const top = niceMax(Math.max(...history.map((h) => h.acres_moderate + h.acres_high), 1));
  const x = (i) => M.l + (i / (n - 1)) * (W - M.l - M.r);
  const y = (v) => M.t + (1 - v / top) * (H - M.t - M.b);
  const line = (key) => history.map((h, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(h[key]).toFixed(1)}`).join("");
  const ticks = [0, top / 2, top];
  const cur = history[index];
  const prev = index > 0 ? history[index - 1] : null;

  return (
    <div className="chart">
      <div className="legend-row">
        <span><i style={{ background: HIGH }} />High risk (0.6+)</span>
        <span><i style={{ background: MOD }} />Moderate (0.3 to 0.6)</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Acres at risk over ${n} scans. On ${fmtDate(cur.date)}: ${fmtAcres(cur.acres_high)} acres high risk, ${fmtAcres(cur.acres_moderate)} acres moderate.`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} stroke="#28332e" strokeWidth="1" />
            <text x={M.l - 6} y={y(t) + 4} textAnchor="end" className="ax">{Math.round(t)}</text>
          </g>
        ))}
        <text x={M.l - 6} y={10} textAnchor="end" className="ax unit">acres</text>
        <line x1={x(index)} x2={x(index)} y1={M.t} y2={H - M.b} stroke="#9bd25a" strokeWidth="1.5" strokeDasharray="3 3" />
        <path d={line("acres_moderate")} fill="none" stroke={MOD} strokeWidth="2.2" strokeLinejoin="round" />
        <path d={line("acres_high")} fill="none" stroke={HIGH} strokeWidth="2.2" strokeLinejoin="round" />
        {history.map((h, i) => (
          <g key={h.date}>
            <circle cx={x(i)} cy={y(h.acres_moderate)} r={i === index ? 5 : 3} fill={MOD} stroke="#151b18" strokeWidth="2" />
            <circle cx={x(i)} cy={y(h.acres_high)} r={i === index ? 5 : 3} fill={HIGH} stroke="#151b18" strokeWidth="2" />
            <text x={x(i)} y={H - 8} textAnchor="middle" className={"ax" + (i === index ? " on" : "")}>{fmtDate(h.date).replace(/ \d{4}$/, "").replace(",", "")}</text>
            <rect x={x(i) - 18} y={M.t} width="36" height={H - M.t} fill="transparent" style={{ cursor: "pointer" }} onClick={() => onIndex(i)}>
              <title>{`${fmtDate(h.date)}: ${fmtAcres(h.acres_high)} ac high, ${fmtAcres(h.acres_moderate)} ac moderate`}</title>
            </rect>
          </g>
        ))}
      </svg>
      <p className="chart-note">
        {fmtDate(cur.date)}: <b className="num" style={{ color: HIGH }}>{fmtAcres(cur.acres_high)} ac</b> high risk
        {prev && <> (<span className="num">{cur.acres_high - prev.acres_high >= 0 ? "+" : ""}{fmtAcres(cur.acres_high - prev.acres_high)}</span> since the last scan)</>}.
      </p>
    </div>
  );
}
