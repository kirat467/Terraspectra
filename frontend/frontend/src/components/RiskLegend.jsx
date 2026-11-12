import { legendGradient } from "../lib/risk.js";

export default function RiskLegend({ horizon }) {
  return (
    <div className="legend" aria-label="Risk colour scale">
      <div className="legend-title">Blight risk, next {horizon} days</div>
      <div className="legend-bar" style={{ background: legendGradient() }} />
      <div className="legend-ticks num"><span>0 Low</span><span>0.6 High</span><span>1 Severe</span></div>
    </div>
  );
}
