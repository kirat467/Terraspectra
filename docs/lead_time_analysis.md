# Lead-time analysis (Day 22)

Question from the plan: **how many days before symptoms become visible does each detector flag stress?** Code: `ml/src/lead_time.py`. Results: `docs/lead_time.json` and `docs/figures/lead_time.png`.

    python ml/src/lead_time.py        # needs data/generated/patches.npz and ml/models/cnn_run/

## Method
- For every pixel that reaches visible stress (label 2), lead time = days between the first scan where the detector flags it and the first scan where it is visible. Scans are a week apart, so leads come in steps of 7 days.
- Detectors: NDVI, red edge position, NDII (leaf water), each averaged over 5 x 5 pixels, and the trained 3D-CNN (disease probability = 1 minus P(healthy)).
- **Same false-alarm budget for all:** each threshold lets 1 percent of healthy pixels through in a single scan. The threshold uses healthy pixels only.
- **Scored on new data:** three newly simulated farms (192 x 192 pixels, 6 weekly scans each) from `data/make_large_scene.py` with other field layouts, crop mixes, outbreak places and noise than the farm the CNN trained on. 9,745 pixels reach visible stress. Pixels within 4 pixels of soil, tracks or water are left out for every detector.
- Why not the original test blocks: they hold one outbreak, centred on a pond, leaving too few valid pixels (79 labelled, most of them water; 2 after excluding water).

## Results
Share of pixels flagged at least 14 days before symptoms are visible, per farm:

| Detector | Farm 1 | Farm 2 | Farm 3 | Pooled shares (21+ / 14 / 7 days before) |
|---|---|---|---|---|
| NDVI | 76% | 74% | 76% | 12% / 63% / 24% |
| Red edge position | 99% | 100% | 94% | 24% / 73% / 3% |
| NDII (leaf water) | 99% | 100% | 91% | 23% / 74% / 4% |
| 3D-CNN | 100% | 100% | 95% | 28% / 70% / 2% |

- The median lead is **14 days for every detector on every farm**. No detector missed a pixel or flagged one after the symptoms were visible.
- **14 days is also the head start the simulation itself builds in** (first stress to first visible stress, median 14 days on all three farms). The detectors reach that limit; none can beat it by much.
- The CNN is slightly earlier than the others (28% flagged 21 or more days ahead against 12% for NDVI), and NDVI is clearly the slowest. **The CNN is not clearly better than the plain red edge or NDII index**: their shares sit within a few points of each other, and farm 3 is lower for the red edge, NDII and CNN.
- False alarms: 1 percent per scan means about 6 percent of healthy pixels get flagged at least once across 6 scans (5.8 to 6.0 percent measured).

## What this does not show
- **It is a simulation.** The red edge shift is built into the spectra as the first change, so spectral indices doing well here says little about real fields. Real labels are needed.
- Weekly resolution: differences smaller than 7 days cannot show up. Pixels next to each other are strongly correlated, so 9,745 pixels are far fewer than 9,745 independent samples; three farms give a rough idea of spread, not a confidence interval.
- A pixel flagged on the very first scan has a lead time that is only a lower bound; about 1 percent of pixels or fewer here.

## Found and fixed on the way
`make_large_scene.py` labelled pond pixels by the disease map around them, so a pond inside an outbreak counted as "visible stress". Water is now labelled "not a crop" (-1), with a test. **`data/make_scene.py` (the original farm) has the same flaw** for its three ponds, which also touches a few dozen training labels; the next regeneration should fix it, and the first CNN run should be repeated afterwards.
