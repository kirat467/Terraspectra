# Mid-review demo script (about 4 minutes)

Open the dashboard and press **Demo** (top right). Arrow keys step, Escape exits. Say the lines below while each step plays.

1. **The whole farm (30 s).** "This is a simulated 1,013-acre farm. Every pixel is 15 m and has 200 spectral bands. Colour is the predicted blight risk for the next 21 days."
2. **What a normal camera sees (30 s).** "With the risk layer off, the farm looks healthy. Blight changes how leaves reflect light before we can see it."
3. **The first scan (30 s).** "6 October: peak risk 0.22 and no area at high risk yet. A faint spot is starting."
4. **Six weeks later (40 s).** Press play on the timeline. "It spreads scan by scan. The sidebar shows acres at risk and the trend."
5. **Where to go first (40 s).** "Priority zones turn the map into a visit list: a risk level and a smallest area, both adjustable." Move the sliders live.
6. **Honest about the limits (30 s).** "The data is simulated. The forecast is a trend rule. Real field labels are the next step."

If asked to show real data: open `/?data=real` (stress index from the EMIT scene, not a disease probability).

## Likely questions
- **Why simulated data?** No labelled hyperspectral disease dataset covers one farm over weeks. The simulator uses a physical leaf model: the red edge moves first, NDVI later.
- **Why 200 bands?** Early stress shows in narrow bands (red edge, water absorption) that RGB and NDVI miss. The baseline in `ml/src/spectral_baseline.py` shows this.
- **How was the split done without leakage?** By spatial blocks, with PCA fitted on training blocks only (`docs/week2_data_notes.md`).
- **What does the model get wrong?** It misses the visible-stress class and is overconfident (`docs/first_training_run.md`).
- **How does the forecast work?** A damped trend on each pixel's score, 21 days ahead; not a learned forecaster yet.
- **Does it scale?** `docs/large_scene.md`: a 1024 x 1024 x 200 scene is generated with 540 MB to 1 GB peak memory depending on strip size.
