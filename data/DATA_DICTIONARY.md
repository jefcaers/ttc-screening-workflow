# Data dictionary

All monetary values are US dollars. "NPV" and "RAV" figures are in US$ millions
unless a field note says otherwise; capital-cost ratios are dimensionless
multiples (realized ÷ sanctioned). Every JSON file here is either a **derived
result** produced by the code in this repository or a **manuscript table** of
public-sourced figures; none is a hand-entered model input consumed by the
engines (the engines carry their own inputs inline — see the README).

---

## `rf_data.json`

Master results dump written by `make_results_figs.py`. Every value is engine
output. One object with these keys:

| Key | Structure | Meaning |
|---|---|---|
| `forest` | array of 21 records (10 lithium + 11 copper) | Per project: `name`; `yr` (sanction year); `ratio` (realized ÷ sanctioned capital, e.g. 1.75); `lo`/`hi` (frozen 80% reference-class band, ratio units); `ilo`/`ihi` (inside-view 80% band); `pit` (probability-integral-transform value under the frozen posterior, 0–1); `com` ("Li"/"Cu"); `in` (bool, realized inside the reference-class band); `in_inside` (bool, inside the inside-view band); copper rows add `conf` ("high"/"med"/"low" curation confidence). |
| `pit` | object | Probability-integral-transform summary: `all` (21 values), `ks_p_all`, `mean_all`, and per-commodity `ks_p_li`/`mean_li`/`ks_p_cu`/`mean_cu`. KS = Kolmogorov–Smirnov p-value against uniform. |
| `coverage` | object | 80%-band coverage counts: `Li_rc`, `Li_in` (of 10), `Cu_rc`, `Cu_in` (of 11). |
| `copper_choices` | array of 11 | Per copper case: `name`, `yr`, `rav` (US$M), `choice` ("enter"/"stage"/"walk"), `realized` (reconstructed realized NPV, US$M), `ratio`, `conf`. |
| `rho8` | `[ρ, p]` | Spearman rank correlation of ex-ante RAV vs realized NPV, copper, 8% discount. |
| `li_loss` | object | Lithium decision loss (US$M): `workflow`, `always_enter`, `always_walk`, `random`. |
| `cu_lossgrid` | 3×4 int array | % of the always-enter→oracle gap closed. Rows = staged exposure {0.1,0.3,0.5}; cols = missed-upside weight χ {0.25,0.5,0.75,1.0}. |
| `rho_grid` | object | `rho` (3×3), `p` (3×3), `mix` (3×3 "enter/stage/walk" strings). Rows = discount {8,9,10}%, cols = λ {0.3,0.5,0.7}. |
| `corr_mean` | float | Mean off-diagonal pairwise correlation of sanction-window log prices, all 21 cases. |
| `heldout` | object | `events` (4 post-freeze policy events) and `static`/`onestep`/`multi`/`base`, each a 4-array of P(activation within 5 yr). |
| `chile_path` | 6×3 array | Regime-belief path [orthodox, interventionist, crisis] over 6 years. |
| `qb2` | object | Quebrada Blanca 2 stats (US$M): `rav`, `cvar`, `mean`. |

---

## `cohort_tables.json`

The two cohort tables of the manuscript, each an array of rows (first row is the
header).

- **`T1`** — lithium cohort, 10 projects. Columns: Project, FID year, Sanction
  CapEx (e.g. "US$229M"), Realized ratio (e.g. "x1.75"), Outcome (prose).
- **`T2`** — copper cohort, 11 projects. Same columns plus `Conf` (high/med/low
  curation confidence). CapEx in US$B.

## `appendix_tables.json`

Three appendix tables, each an object with `headers`, `rows`, `widths` (docx
column widths in twips — for document layout only), and `cap` (caption).

- **`A1`** — AACE estimate class → dispersion (`s_1`) mapping (5 rows): class,
  maturity, the 47R-11 P10/P90 accuracy band, and the log-σ used (0.09–0.45).
- **`A2`** — lithium era parameter register (6 rows): sanction year, reference
  Li₂CO₃ price (US$/t), central demand-growth outlook, primary source.
- **`A3`** — copper era parameter register (8 rows): sanction year, reference LME
  copper price (US$/t), market-balance/demand outlook, primary source.

## `policy_event_table.json`

One array of 42 rows (1 header + 41 policy events) spanning 17 countries,
2007–2026. Columns: Country, Year, Lever (Nationalization / Export restriction /
Tax-royalty / Local content), Regime (orthodox / interventionist / crisis), Named
instrument, Primary source. **Data status:** every `Primary source` cell is
currently `[CHECK]` — the legal citations are being finalized before submission.
The event coding itself (country, year, lever, instrument, regime) is complete and
is what the country-risk hazards are estimated from.

## `table5_regen.json`

One array of 10 rows (lithium projects, no header). Positional columns:
[project name, Sobol top driver, realized decisive factor, match ("yes"/"no"),
cost-audit percentile, outcome prose]. Backs the driver-attribution table.

---

## `Policy_Event_Register.md`

Narrative companion to `policy_event_table.json`, framed as supplementary
material: the 41 resource-nationalism events with their instruments and regime
context, cross-referenced to the OECD 2024 inventory of export restrictions. Carries
the same `[CHECK]` note on primary-citation finalization.

## Not included in this public release

The copper **curation register** (per-case confidence tiering and source audit)
and the **era-sourcing research memo** are internal working documents containing
candid review notes; they are held back from the public repository and can be
provided in sanitized form on request. The confidence tier of every copper figure
is preserved here in the `conf` fields of `rf_data.json` and the `Conf` column of
`cohort_tables.json` (`T2`).
