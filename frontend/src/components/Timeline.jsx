import { useEffect, useRef, useState } from "react";
import { fmtDate } from "../lib/geo.js";

const short = (iso) => fmtDate(iso).replace(/ \d{4}$/, "");   // "6 Oct"

// Play / pause, step buttons and a slider across the scan dates. The map and every panel follow `index`.
export default function Timeline({ dates, index, onIndex }) {
  const [playing, setPlaying] = useState(false);
  const last = dates.length - 1;
  const idx = useRef(index);
  idx.current = index;

  useEffect(() => {
    if (!playing) return undefined;
    const timer = setInterval(() => {
      if (idx.current >= last) { setPlaying(false); return; }
      onIndex(idx.current + 1);
    }, 1300);
    return () => clearInterval(timer);
  }, [playing, last, onIndex]);

  const togglePlay = () => {
    if (!playing && index >= last) onIndex(0);        // pressing play at the end starts over
    setPlaying(!playing);
  };
  const go = (i) => { setPlaying(false); onIndex(Math.min(last, Math.max(0, i))); };

  return (
    <div className="timeline" role="group" aria-label="Scan timeline">
      <button type="button" className="tl-btn" onClick={() => go(index - 1)} disabled={index === 0} aria-label="Previous scan">‹</button>
      <button type="button" className="tl-btn play" onClick={togglePlay} aria-label={playing ? "Pause" : "Play through the scans"}>
        {playing
          ? <svg width="14" height="14" viewBox="0 0 14 14"><rect x="2" y="1" width="3.6" height="12" rx="1" fill="currentColor" /><rect x="8.4" y="1" width="3.6" height="12" rx="1" fill="currentColor" /></svg>
          : <svg width="14" height="14" viewBox="0 0 14 14"><path d="M3 1.5v11l9-5.5z" fill="currentColor" /></svg>}
      </button>
      <button type="button" className="tl-btn" onClick={() => go(index + 1)} disabled={index === last} aria-label="Next scan">›</button>
      <div className="tl-track">
        <input type="range" min="0" max={last} step="1" value={index} onChange={(e) => go(Number(e.target.value))}
          aria-label="Scan date" aria-valuetext={fmtDate(dates[index])} />
        <div className="tl-ticks">
          {dates.map((d, i) => (
            <button type="button" key={d} className={"tl-tick num" + (i === index ? " on" : "")} style={{ left: `${(i / last) * 100}%` }} onClick={() => go(i)} aria-label={fmtDate(d)}>{short(d)}</button>
          ))}
        </div>
      </div>
      <span className="tl-now num"><small>Scan {index + 1} of {dates.length}</small></span>
    </div>
  );
}
