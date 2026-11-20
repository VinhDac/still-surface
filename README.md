# Still surface

A validation note on Lucic and Tse, *Option market-making and vol arbitrage* (Risk.net, 2025): the example's vega penalty protects the close, not the half day.

The paper's abstract says factor-based risk controls make its options market-making strategy more resilient to unexpected moves of the implied volatility surface. Its numerical example (section 3.2) holds the surface still. This code rebuilds the example, then lets the surface move through the half day by amounts read from the VIX and the VIX3M.

- The example reproduces: the baseline's mean half-day value is 166.6 against about 167 read off the paper's Figure 1(b).
- With each maturity moving by its own loading on the VIX, the half day's standard deviation goes from 4.80 to 5.45 (9.34 if every maturity moved as far as the VIX).
- Ten times the terminal penalty cuts the risk carried past the close by 21%, and the in-session risk by 4%.
- A running penalty on the vega the move loads on cuts the in-session risk by 51% for 0.14 of mean value; ten times the terminal penalty costs 0.79.

The note: [the page](https://vinhdac.github.io/notes/still-surface) and [the PDF](https://vinhdac.github.io/notes/still-surface/Dac-Vinh-Nguyen-Still-Surface.pdf).

## Run it

Python 3 with numpy, scipy and matplotlib (and pdflatex for the PDF).

```
python study.py          # the study: writes results.json (about twenty minutes on ten cores)
python hjb.py            # the closed form against the exact HJB: writes hjb_results.json
python checks.py         # the study's checks; exit code 1 if one fails
python hjb_checks.py     # the HJB's checks, the same way
python mutations.py      # plants faults in the study's code, one at a time on a copy, and expects a check to catch each
python hjb_mutations.py  # the same for hjb.py
python build_report.py   # the PDF, every number read from the results
python build_page.py     # the page, the same way
```

40 checks hold, and all 69 planted faults are caught.

`data.py` reads the VIX (VIXCLS) and the VIX3M (VXVCLS) from FRED once and keeps them in `data/`, which is not in this repository.

## What is assumed

- The implied vol surface: the paper shows it only as a picture, Figure 1(a). `surface.py` fits a four-parameter form to points read off it by eye.
- The skew move is a labelled scenario, not sized from data.
- Past 93 days, how far a maturity moves with the VIX is a power law through the VIX3M's loading.
- Order flow does not react to the surface's moves, and spot and vol move independently.

The paper itself is not in this repository: its author manuscript is at [UCL Discovery](https://discovery.ucl.ac.uk/id/eprint/10207338/).
