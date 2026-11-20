// The guided demo for the mid-review. Each step moves the map and the panels, then shows a caption.
// `run(ctx)` gets the app's controls; `text(ctx)` can quote numbers from the loaded data so the story never goes stale.
import { fmtAcres, fmtDate } from "./geo.js";

const first = (c) => c.scans[0];
const last = (c) => c.scans[c.scans.length - 1];

export const TOUR = [
  {
    title: "The whole farm",
    run: (c) => { c.setIndex(c.scans.length - 1); c.setLayers((l) => ({ ...l, risk: true, imagery: true, opacity: 1 })); c.reset(); },
    text: (c) => `${c.meta.name}: ${fmtAcres(c.meta.farm_acres)} acres, ${c.meta.grid[0]} × ${c.meta.grid[1]} pixels, ${c.meta.bands ?? 200} spectral bands per pixel. The colours are the predicted blight risk for the next ${last(c).horizon_days} days. Greener means safe, red means act.`,
  },
  {
    title: "What a normal camera sees",
    run: (c) => { c.reset(); c.setLayers((l) => ({ ...l, risk: false })); },
    text: () => "Switch the risk layer off and the farm looks healthy. Early blight changes how leaves reflect light long before the eye can see it, and that is what the 200 bands pick up.",
  },
  {
    title: "The first scan",
    run: (c) => { c.setLayers((l) => ({ ...l, risk: true })); c.setIndex(0); c.reset(); },
    text: (c) => `${fmtDate(first(c).date)}: peak risk only ${first(c).summary.peak_risk.toFixed(2)} and ${fmtAcres(first(c).summary.acres_high)} acres at high risk. A faint spot is starting and nothing is visible on the ground yet.`,
  },
  {
    title: "Six weeks later",
    run: (c) => { c.setIndex(c.scans.length - 1); c.reset(); },
    text: (c) => `${fmtDate(last(c).date)}: ${fmtAcres(last(c).summary.acres_high)} acres at high risk, peak ${last(c).summary.peak_risk.toFixed(2)}. Press play on the timeline to watch it spread, scan by scan.`,
  },
  {
    title: "Where to go first",
    run: (c) => { if (c.zones[0]) c.flyToZone(c.zones[0]); },
    text: (c) => {
      const z = c.zones[0];
      return z
        ? `Priority zone P1: ${fmtAcres(z.acres)} acres${z.field ? ` in ${z.field}${z.crop ? ` (${z.crop})` : ""}` : ""}, peak risk ${z.peak.toFixed(2)}. Zones come from a risk level and a smallest area (sliders in the left panel), so the farmer gets a short visit list, not a heat map to interpret.`
        : "No zone meets the current thresholds. Lower the sliders in the left panel to see smaller or milder zones.";
    },
  },
  {
    title: "Honest about the limits",
    run: (c) => { c.reset(); },
    text: () => "This farm is simulated: the spectra come from a physical leaf model, so the demo shows the pipeline end to end, not field accuracy. The forecast is a trend rule on the model's scores. Real labels from field visits are the next step before any claim about real farms.",
  },
];
