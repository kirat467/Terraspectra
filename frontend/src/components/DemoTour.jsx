import { useEffect } from "react";

// Caption card for the guided demo. Arrow keys step, Escape exits.
export default function DemoTour({ steps, step, text, onStep, onExit }) {
  useEffect(() => {
    const key = (e) => {
      if (e.key === "ArrowRight") onStep(Math.min(steps.length - 1, step + 1));
      else if (e.key === "ArrowLeft") onStep(Math.max(0, step - 1));
      else if (e.key === "Escape") onExit();
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [step, steps.length, onStep, onExit]);

  const s = steps[step];
  const lastStep = step === steps.length - 1;
  return (
    <div className="tour" role="dialog" aria-label="Guided demo">
      <div className="tour-top">
        <span className="tour-count num">{step + 1} / {steps.length}</span>
        <button className="tour-x" onClick={onExit} aria-label="Exit demo">Exit</button>
      </div>
      <h3>{s.title}</h3>
      <p>{text}</p>
      <div className="tour-nav">
        <div className="tour-dots" aria-hidden="true">{steps.map((_, i) => <i key={i} className={i === step ? "on" : i < step ? "done" : ""} />)}</div>
        <button className="tour-btn" disabled={step === 0} onClick={() => onStep(step - 1)}>Back</button>
        <button className="tour-btn primary" onClick={() => (lastStep ? onExit() : onStep(step + 1))}>{lastStep ? "Finish" : "Next"}</button>
      </div>
    </div>
  );
}
