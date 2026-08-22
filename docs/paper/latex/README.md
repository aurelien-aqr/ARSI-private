# The manuscript

Publisher-neutral LaTeX. Content lives in `sections/`; everything a journal
template touches lives in three files. Build with `make` (needs `latexmk` and
`pdflatex`).

```
main.tex        \documentclass + the section order          <- template-specific
preamble.tex    packages, \keywords, \figref/\tabref/\secref <- template-specific
metadata.tex    title, authors, affiliations, abstract       <- template-specific
sections/*.tex  the paper                                    <- PORTABLE, never edit for a template
references.bib  40 entries, Crossref/DataCite BibTeX
figures/        fig2_proposers.jpg, fig4_aboda.jpg
```

The split is the point: when Pavol confirms the venue, converting means
replacing the three files at the top and leaving `sections/` untouched. Any
change you make to a section file must keep that true — if you find yourself
writing `\IEEEPARstart` or `\Author` inside a section, it belongs in
`metadata.tex` instead.

## Converting

**To MDPI** (*Journal of Imaging*, *Sensors*, *Applied Sciences*). Download the
`Definitions/` + `mdpi.cls` bundle, then:

1. `main.tex`: `\documentclass[journalname,article,submit,pdftex]{Definitions/mdpi}`.
2. `metadata.tex`: `\Title{}`, `\Author{}`, `\AuthorNames{}`, `\address{}`,
   `\corres{}`, `\abstract{}`, `\keyword{}`. The class defines `\keyword`, so
   drop our `\keywords` fallback from `preamble.tex`.
3. `preamble.tex`: delete `geometry`, `natbib`, `hyperref` — the class loads its
   own. Keep `booktabs`, `multirow`, `graphicx`, `amsmath`.
4. Bibliography: `\bibliographystyle{mdpi}`.
5. Move `sections/backmatter.tex` content into the class's dedicated commands
   (`\authorcontributions{}`, `\funding{}`, `\dataavailability{}`,
   `\conflictsofinterest{}`).
6. MDPI folds related work into the Introduction and a "Background" subsection.
   The header comment in `02_related_work.tex` says how it splits.

**To IEEE** (Access or a conference). `\documentclass{ieeeaccess}` or
`\documentclass[conference]{IEEEtran}`, then:

1. `metadata.tex`: `\author{\uppercase{Name}\authorrefmark{1}}`,
   `\IEEEmembership{}`, `\address{}`, `\tfootnote{}`. Abstract becomes
   `\begin{abstract}`; keywords become `\begin{keywords}` (Access) or
   `\begin{IEEEkeywords}` (conference).
2. `preamble.tex`: delete `geometry` and `natbib`; IEEE uses plain `\cite` with
   `\bibliographystyle{IEEEtran}`.
3. Two-column: the wide tables in `05_results.tex` need `table*` instead of
   `table`. That is the only section-file edit the conversion requires, and it
   is mechanical — Tables 2, 4 and 6 (`tab:main`, `tab:judge`, `tab:cdnet`).
4. Back matter: IEEE keeps only acknowledgements; funding goes in a page-1
   footnote.

## What is not finished

Marked `TODO` in the files themselves:

- **`metadata.tex`** — the author list. Only the first author is filled in.
- **`sections/backmatter.tex`** — author-contribution initials, the grant
  number, and whether an ethics statement is required for operational CCTV
  footage of passengers. That last one is a real question, not boilerplate.
- **`sections/03_methodology.tex`** — Figure 1, the pipeline diagram, is a
  placeholder box. A TikZ version keeps it template-independent and scales in
  two columns; an exported bitmap does not.
- **Three reading debts** listed in `../related_work.md` §11 (BEM in full,
  AnomalyRuler's perception front-end, Caetano et al.). They affect the
  Introduction and Related Work only.

## Where the numbers come from

Every table maps to a file under `docs/`, and every one of those is
reproducible from a script named in `../results.md`:

| Table | Source |
|---|---|
| 1 data composition | `benchmark/datasets/ground_truth.json` |
| 2, 3 proposal stages | `../results.md` §0a, `tools/rescore_localizer.py` |
| 4 judge sweep | `docs/vlm_benchmark/metrics.json`, `tools/judge_sweep.py` |
| 5 baselines | `docs/dino_models/anomalib_baseline.json` |
| 6 CDnet | `docs/public_benchmarks/cdnet_*.json` |
| 7 cross-camera | `docs/dino_models/cross_camera*.json` |
| Fig. 2 | `tools/figure_proposer.py` |
| Fig. 3 | `tools/aboda_qualitative.py` |

If a number ever appears here with no script behind it, that is the one a
reviewer will ask about.
