"""The one design system for the standalone notes under docs/.

Every note listed in tools/notes.py is built from this module, so a second note
cannot quietly drift into a second design. It used to be two sheets - a serif newspaper for the framing
drift measurement and a sans note for the model comparison - which read as two
unrelated documents; this is their union, in the Studio's own palette.

    from report_style import STYLE, head, foot

`head()` opens the page: <title>, the sheet, the sticky bar with the PDF
button, and the wrapper. `foot()` closes it. What sits between the two is the
note's own business - the notes carry different data and are not meant to look
identical, only to look related.

STYLE is a plain string, NOT an f-string fragment: callers interpolate it as
{STYLE}, so the braces here are single.

Class inventory, in the order a note tends to need it:

    header.masthead   eyebrow / h1 / .lede / .meta
    .keys > .key      the headline numbers, .key__v / __l / __n
    section           two-track grid: .sect-head in the rail, the rest right
    .col              a prose measure inside that right track
    ol.steps          a numbered sequence, --a1 / --a2 tint it
    ul.plain          a plain list that is not a sequence
    figure            + figcaption, .chart / .diagram / .tablewrap inside it
    table             .num .strong .muted .win .bad, tr.base, .bar--ok/warn/crit
    .cards > .card    side-by-side objects, .card--a1 / --a2
    .finding          a numbered list of conclusions, boxed
    .note             a callout
    .case             a worked example
    .cams > .cam      the per-item gallery
    pre               a fenced listing
    .pill .badge      inline status
    .legend           colour key under a figure
"""

from html import escape

STYLE = r"""<style>
:root {
  color-scheme: light dark;
  --bg:#F6F8F9; --card:#FFFFFF; --sunk:#ECF0F2;
  --ink:#161A20; --ink-2:#4A545F; --ink-3:#78838E;
  --line:#DCE2E7; --line-2:#E9EDF0;
  --accent:#1D6A9B; --accent-soft:#E4EFF7;
  --a1:#1E6BA8; --a1-soft:#E5F0F8;
  --a2:#8E3A7E; --a2-soft:#F6E9F3;
  --ok:#1F7A4D; --warn:#8A5B0F; --crit:#A63A32;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Inter,Roboto,"Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,"SF Mono","DejaVu Sans Mono",Menlo,Consolas,monospace;
  --rail:212px; --gutter:44px;
  /* the prose measure, in px not ch: `ch` depends on which font actually
     resolved, and the notes are read in a browser, printed by Chrome and
     framed in the Studio - all three have to break lines in the same place */
  --measure:680px;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg:oklch(0.15 0.008 250); --card:oklch(0.17 0.01 250); --sunk:oklch(0.13 0.008 250);
    --ink:oklch(0.93 0.006 250); --ink-2:oklch(0.7 0.012 250); --ink-3:oklch(0.55 0.012 250);
    --line:oklch(0.26 0.012 250); --line-2:oklch(0.22 0.01 250);
    --accent:oklch(0.72 0.13 225); --accent-soft:oklch(0.22 0.04 225);
    --a1:oklch(0.72 0.13 225); --a1-soft:oklch(0.24 0.06 225);
    --a2:oklch(0.8 0.13 300); --a2-soft:oklch(0.26 0.08 300);
    --ok:oklch(0.82 0.11 150); --warn:oklch(0.8 0.09 75); --crit:oklch(0.78 0.16 22);
  }
}
:root[data-theme="dark"] {
  --bg:oklch(0.15 0.008 250); --card:oklch(0.17 0.01 250); --sunk:oklch(0.13 0.008 250);
  --ink:oklch(0.93 0.006 250); --ink-2:oklch(0.7 0.012 250); --ink-3:oklch(0.55 0.012 250);
  --line:oklch(0.26 0.012 250); --line-2:oklch(0.22 0.01 250);
  --accent:oklch(0.72 0.13 225); --accent-soft:oklch(0.22 0.04 225);
  --a1:oklch(0.72 0.13 225); --a1-soft:oklch(0.24 0.06 225);
  --a2:oklch(0.8 0.13 300); --a2-soft:oklch(0.26 0.08 300);
  --ok:oklch(0.82 0.11 150); --warn:oklch(0.8 0.09 75); --crit:oklch(0.78 0.16 22);
}

* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
body {
  background:var(--bg); color:var(--ink);
  font-family:var(--sans); font-size:16.5px; line-height:1.62;
  margin:0; padding:0 0 110px;
  -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility;
}
.wrap { max-width:1180px; margin:0 auto; padding:0 30px; }

h1,h2,h3,h4 { margin:0; line-height:1.2; text-wrap:balance; font-weight:650; }
h1 { font-size:clamp(2.05rem,4.4vw,3rem); letter-spacing:-.025em; }
h2 { font-size:1.32rem; letter-spacing:-.015em; }
h3 { font-size:1.04rem; }
h4 { font-size:.93rem; }
p { margin:0; }
/* the converted notes cite full URLs; without this one runs off the sheet */
a { color:var(--accent); text-underline-offset:2px; overflow-wrap:break-word; }
em { font-style:italic; }
code { font-family:var(--mono); font-size:.85em; background:var(--sunk);
       padding:1px 5px; border-radius:4px; }

.eyebrow { font-family:var(--mono); font-size:.68rem; letter-spacing:.15em;
           text-transform:uppercase; color:var(--ink-3); }

/* --- the bar, shared by every note ------------------------------------- */
.docbar {
  position:sticky; top:0; z-index:20; background:color-mix(in srgb, var(--bg) 88%, transparent);
  backdrop-filter:saturate(1.4) blur(9px); border-bottom:1px solid var(--line);
  margin-bottom:0;
}
.docbar__in { max-width:1180px; margin:0 auto; padding:0 30px; height:54px;
              display:flex; align-items:center; gap:12px; }
/* the Studio's own badge: accent square, glyph knocked out to the page ground,
   which lands white on light and near-black on dark without a second rule */
.mark { width:26px; height:26px; border-radius:7px; background:var(--accent);
        color:var(--bg); display:grid; place-items:center; flex:0 0 auto;
        font-family:var(--mono); font-weight:800; font-size:13px; }
.docbar__app { font-size:13px; font-weight:650; letter-spacing:-.01em; }
.docbar__sep { color:var(--ink-3); font-size:13px; }
.docbar__name { font-family:var(--mono); font-size:12px; color:var(--ink-3);
                white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.docbar__sp { margin-left:auto; }
.btn { font:inherit; font-size:12.5px; font-weight:550; color:var(--ink-2);
       background:var(--card); border:1px solid var(--line); border-radius:8px;
       padding:7px 12px; cursor:pointer; display:inline-flex; align-items:center;
       gap:7px; text-decoration:none; white-space:nowrap; }
.btn:hover { color:var(--ink); border-color:var(--ink-3); }
.btn[disabled] { opacity:.6; cursor:progress; }
.btn svg { width:14px; height:14px; }
.btn--go { color:var(--bg); background:var(--accent); border-color:var(--accent); }
.btn--go:hover { filter:brightness(1.08); color:var(--bg); }
@keyframes spin { to { transform:rotate(360deg); } }
.btn.busy svg { animation:spin 1s linear infinite; }

/* --- masthead ---------------------------------------------------------- */
header.masthead { padding:74px 0 40px; display:flex; flex-direction:column; gap:17px; }
/* framed in the Studio there is no bar above the title, so it starts higher */
:root.embed header.masthead { padding-top:44px; }
header.masthead .lede { font-size:1.14rem; color:var(--ink-2); max-width:600px; }
header.masthead .lede strong, header.masthead .lede b { color:var(--ink); font-weight:650; }
.meta { display:flex; flex-wrap:wrap; gap:6px 26px; font-family:var(--mono);
        font-size:.73rem; color:var(--ink-3); padding-top:4px; }

/* --- headline numbers --------------------------------------------------- */
.keys { display:grid; grid-template-columns:repeat(auto-fit,minmax(186px,1fr));
        gap:1px; background:var(--line); border:1px solid var(--line);
        border-radius:12px; overflow:hidden; margin:8px 0 0; }
.key { background:var(--card); padding:19px 20px 21px; display:flex;
       flex-direction:column; gap:5px; }
.key__v { font-family:var(--mono); font-size:1.75rem; font-weight:600;
          letter-spacing:-.035em; font-variant-numeric:tabular-nums; line-height:1.05; }
.key__v.a1 { color:var(--a1); } .key__v.a2 { color:var(--a2); }
.key__l { font-size:.85rem; color:var(--ink-2); line-height:1.42; }
.key__n { font-family:var(--mono); font-size:.67rem; color:var(--ink-3); margin-top:auto; padding-top:4px; }

/* --- the section grid: heading in a rail, everything else beside it ----- */
section { display:grid; grid-template-columns:var(--rail) minmax(0,1fr);
          gap:0 var(--gutter); align-items:start;
          margin-top:52px; padding-top:34px; border-top:1px solid var(--line); }
.sect-head { grid-column:1; grid-row:1; display:flex; flex-direction:column; gap:7px; }
.sect-head .eyebrow { order:-1; }
section > :not(.sect-head) { grid-column:2; }
section > * + *:not(.sect-head) { margin-top:20px; }
section > .sect-head + * { margin-top:0; }   /* first block level with its heading */
.col { max-width:var(--measure); display:flex; flex-direction:column; gap:15px; }
.col p { color:var(--ink-2); }
.col p b, .col p strong { color:var(--ink); font-weight:650; }
.full { grid-column:1 / -1 !important; }
figure.full > figcaption { max-width:900px; }

/* --- sequences ---------------------------------------------------------- */
ol.steps { margin:0; padding:0; list-style:none; counter-reset:s;
           display:flex; flex-direction:column; gap:16px; max-width:var(--measure); }
ol.steps li { counter-increment:s; padding-left:40px; position:relative; color:var(--ink-2); }
ol.steps li::before {
  content:counter(s,decimal-leading-zero); position:absolute; left:0; top:2px;
  width:27px; height:27px; border-radius:50%; display:grid; place-items:center;
  font-family:var(--mono); font-size:.72rem; font-weight:600;
  background:var(--sunk); color:var(--ink-2); }
ol.steps--a1 li::before { background:var(--a1-soft); color:var(--a1); }
ol.steps--a2 li::before { background:var(--a2-soft); color:var(--a2); }
ol.steps b { color:var(--ink); font-weight:650; }
ol.steps p { font-size:.93rem; margin-top:5px; color:var(--ink-2); }

ul.plain { margin:0; padding-left:1.15em; display:flex; flex-direction:column;
           gap:11px; color:var(--ink-2); max-width:var(--measure); }
ul.plain li::marker { color:var(--ink-3); }
ul.plain b { color:var(--ink); font-weight:650; }

/* --- figures ------------------------------------------------------------ */
figure { margin:0; display:flex; flex-direction:column; gap:10px; }
figure img { width:100%; height:auto; display:block; background:var(--sunk);
             border:1px solid var(--line); border-radius:10px; }
figcaption { font-size:.82rem; line-height:1.55; color:var(--ink-3); max-width:740px; }
.chart, .diagram { background:var(--card); border:1px solid var(--line);
                   border-radius:12px; padding:18px 16px 10px; }
.chart svg, .diagram svg { width:100%; height:auto; display:block; }
text.lbl  { font-family:var(--sans); font-size:12.5px; fill:var(--ink); }
text.sub  { font-family:var(--mono); font-size:10.5px; fill:var(--ink-3); }
text.big  { font-family:var(--sans); font-size:14px; font-weight:650; fill:var(--ink); }
text.tick { font-family:var(--mono); font-size:10px; fill:var(--ink-3); }
text.pt   { font-family:var(--mono); font-size:11px; fill:var(--ink); }
text.axis { font-family:var(--mono); font-size:10.5px; fill:var(--ink-3);
            letter-spacing:.06em; text-transform:uppercase; }

/* --- tables ------------------------------------------------------------- */
.tablewrap { overflow-x:auto; border:1px solid var(--line); border-radius:12px;
             background:var(--card); }
table { border-collapse:collapse; width:100%; font-size:.85rem;
        min-width:max-content; font-variant-numeric:tabular-nums; }
caption { text-align:left; padding:13px 16px; font-family:var(--mono);
          font-size:.71rem; line-height:1.5; color:var(--ink-3);
          border-bottom:1px solid var(--line); white-space:normal; }
th,td { padding:9px 14px; text-align:left; white-space:nowrap;
        border-bottom:1px solid var(--line-2); }
thead th { font-family:var(--mono); font-size:.66rem; font-weight:500;
           letter-spacing:.08em; text-transform:uppercase; color:var(--ink-3);
           background:var(--sunk); vertical-align:bottom; }
tbody th { font-weight:650; }
tbody th small, td small { display:block; font-weight:400; font-size:.72rem;
           color:var(--ink-3); font-family:var(--mono); margin-top:2px;
           letter-spacing:0; text-transform:none; }
tbody tr:last-child td, tbody tr:last-child th { border-bottom:none; }
td.num { text-align:right; font-family:var(--mono); position:relative; }
td.strong { color:var(--ink); font-weight:650; }
td.muted { color:var(--ink-3); }
td.win, td.good { color:var(--ok); font-weight:650; }
td.bad { color:var(--crit); }
tr.base td, tr.base th { background:var(--sunk); }
b.alarm { color:var(--warn); font-weight:650; }
.arrow { color:var(--ink-3); }
.bar { position:absolute; left:0; bottom:1px; height:2px; width:calc(var(--v) * 100%); }
.bar--ok { background:var(--ok); } .bar--warn { background:var(--warn); }
.bar--crit { background:var(--crit); }

/* A listing, not a data table: the converted notes use fenced blocks for a
   paper outline, a protocol and a couple of small diagrams. */
pre { margin:0; background:var(--sunk); border:1px solid var(--line);
      border-radius:10px; padding:15px 17px; overflow-x:auto;
      font-family:var(--mono); font-size:.79rem; line-height:1.6;
      color:var(--ink-2); }
pre code { background:none; padding:0; font-size:inherit; color:inherit; }

/* Markdown tables hold sentences rather than numbers, so their cells wrap
   instead of running off the side under the mono/nowrap rules above. */
.tablewrap--text table { min-width:0; font-size:.85rem; }
.tablewrap--text th, .tablewrap--text td { white-space:normal; vertical-align:top;
      padding:11px 15px; }
.tablewrap--text thead th { font-family:var(--mono); }
.tablewrap--text tbody td:first-child, .tablewrap--text tbody th { color:var(--ink); }

/* --- inline status ------------------------------------------------------ */
.pill { font-family:var(--mono); font-size:.69rem; padding:3px 9px;
        border:1px solid currentColor; border-radius:99px; white-space:nowrap; }
.pill--ok { color:var(--ok); } .pill--warn { color:var(--warn); }
.pill--crit { color:var(--crit); }
.badge { display:inline-block; align-self:flex-start; font-family:var(--mono); font-size:.65rem;
         letter-spacing:.08em; text-transform:uppercase; padding:3px 8px;
         border-radius:99px; }
.badge--a1 { background:var(--a1-soft); color:var(--a1); }
.badge--a2 { background:var(--a2-soft); color:var(--a2); }

/* --- boxed blocks ------------------------------------------------------- */
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:16px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px;
        padding:21px 22px 23px; display:flex; flex-direction:column; gap:10px; }
.card--a1 { border-top:3px solid var(--a1); }
.card--a2 { border-top:3px solid var(--a2); }
.card--ok { border-top:3px solid var(--ok); }
.card--warn { border-top:3px solid var(--warn); }
.card--crit { border-top:3px solid var(--crit); }
.card .tagline { font-size:.91rem; color:var(--ink-2); }
.card p { font-size:.9rem; color:var(--ink-2); }
.card dl { margin:8px 0 0; display:grid; grid-template-columns:auto 1fr;
           gap:6px 14px; font-size:.85rem; }
.card dt { color:var(--ink-3); font-family:var(--mono); font-size:.7rem;
           text-transform:uppercase; letter-spacing:.06em; padding-top:2px; }
.card dd { margin:0; color:var(--ink); }

.finding { background:var(--card); border:1px solid var(--line);
           border-left:3px solid var(--accent); border-radius:4px 12px 12px 4px;
           padding:22px 26px; max-width:calc(var(--measure) + 120px); }
.finding ol { margin:0; padding-left:1.3em; display:flex; flex-direction:column;
              gap:12px; color:var(--ink-2); }
.finding li::marker { font-family:var(--mono); font-size:.85em; color:var(--ink-3); }
.finding li b { color:var(--ink); font-weight:650; }

.note { background:var(--card); border:1px solid var(--line);
        border-left:3px solid var(--warn); border-radius:4px 12px 12px 4px;
        padding:17px 22px; display:flex; flex-direction:column; gap:7px;
        max-width:calc(var(--measure) + 40px); }
.note h3, .note h4 { color:var(--ink); }
.note p { font-size:.91rem; color:var(--ink-2); }

dl.wordings { margin:0; display:grid; grid-template-columns:auto 1fr;
              gap:11px 22px; font-size:.92rem; max-width:var(--measure); }
dl.wordings dt { font-weight:650; color:var(--ink); }
dl.wordings dd { margin:0; color:var(--ink-2); }

/* --- worked examples ---------------------------------------------------- */
.case { display:flex; flex-direction:column; gap:13px; padding:24px 0 0;
        border-top:1px solid var(--line-2); }
.case p { color:var(--ink-2); max-width:var(--measure); font-size:.95rem; }
.case__cam { font-family:var(--mono); font-size:.75rem; color:var(--ink-3); margin-left:8px; }

/* --- per-item gallery --------------------------------------------------- */
.cams { display:flex; flex-direction:column; gap:0; }
.cam { padding:32px 0 34px; border-top:1px solid var(--line);
       display:flex; flex-direction:column; gap:15px; }
.cam:first-child { border-top:none; padding-top:6px; }
.cam__head { display:flex; align-items:center; justify-content:space-between;
             gap:16px; flex-wrap:wrap; }
.cam__head h3 { font-family:var(--mono); font-size:.9rem; font-weight:650; }
.cam__stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
              gap:1px; background:var(--line); border:1px solid var(--line);
              border-radius:10px; overflow:hidden; margin:0; }
.cam__stats > div { background:var(--card); padding:10px 13px; }
.cam__stats dt { font-family:var(--mono); font-size:.62rem; letter-spacing:.09em;
                 text-transform:uppercase; color:var(--ink-3); }
.cam__stats dd { margin:2px 0 0; font-family:var(--mono); font-size:1rem;
                 font-variant-numeric:tabular-nums; }

/* --- colour key --------------------------------------------------------- */
.legend { display:flex; flex-wrap:wrap; gap:6px 22px; font-family:var(--mono);
          font-size:.72rem; color:var(--ink-3); }
.legend i { display:inline-block; width:9px; height:9px; border-radius:50%;
            margin-right:7px; }
.legend .sw::before { content:"■ "; }
.legend .m { color:var(--a2); } .legend .v { color:var(--ok); }
.legend .j { color:var(--warn); }

footer { margin-top:64px; padding-top:22px; border-top:1px solid var(--line);
         font-family:var(--mono); font-size:.71rem; color:var(--ink-3);
         display:flex; flex-direction:column; gap:6px; }

/* --- one column once the rail no longer fits ---------------------------- */
@media (max-width:980px) {
  section { grid-template-columns:1fr; gap:16px; }
  .sect-head { grid-column:1; grid-row:auto; }
  section > :not(.sect-head) { grid-column:1; }
  header.masthead { padding-top:48px; }
}
@media (max-width:640px) {
  body { font-size:16px; }
  .wrap, .docbar__in { padding:0 18px; }
  ol.steps li { padding-left:34px; }
}

/* --- paper -------------------------------------------------------------- */
@page { size:A4; margin:15mm 14mm 14mm; }
@page :first { margin-top:13mm; }
@media print {
  /* the light palette, whatever the renderer's colour scheme */
  :root, :root[data-theme="dark"], :root:not([data-theme="light"]) {
    --bg:#FFFFFF; --card:#FFFFFF; --sunk:#F1F4F5;
    --ink:#111820; --ink-2:#3F4A54; --ink-3:#6D7883;
    --line:#C9D1D7; --line-2:#DFE4E8;
    --accent:#12586F; --accent-soft:#DCEAEF;
    --a1:#1A5F97; --a1-soft:#E3EEF7;
    --a2:#8A3679; --a2-soft:#F5E7F2;
    --ok:#1F6D46; --warn:#84570E; --crit:#9C352E;
  }
  html, body { background:#FFFFFF; }
  body { font-size:9.6pt; line-height:1.5; padding:0;
         -webkit-print-color-adjust:exact; print-color-adjust:exact; }
  .docbar { display:none; }
  .wrap { max-width:none; padding:0; }
  .col, ol.steps, ul.plain, .note, dl.wordings, figcaption, .case p,
  .finding, header.masthead .lede { max-width:none; }

  h1 { font-size:26pt; }
  h2 { font-size:13.5pt; }
  h3 { font-size:10.5pt; }
  h4 { font-size:9.8pt; }
  header.masthead { padding:0 0 22px; gap:12px; }
  header.masthead .lede { font-size:11pt; max-width:none; }
  .keys { grid-template-columns:repeat(3,1fr); border-radius:0; }
  /* three to a row on paper: the last one stretches over the empty cells,
     which would otherwise print as a grey block */
  .keys .key:last-child:nth-child(3n+1) { grid-column:span 3; }
  .keys .key:last-child:nth-child(3n+2) { grid-column:span 2; }
  .key__v { font-size:15pt; }

  /* the rail becomes a full-width heading; the sheet is too narrow for two */
  section { display:block; margin-top:0; padding-top:20px; }
  .sect-head { margin-bottom:13px; }
  section > * + *:not(.sect-head) { margin-top:14px; }

  /* nothing orphaned from what introduces it */
  h1, h2, h3, h4 { break-after:avoid; }
  .sect-head, .cam__head { break-after:avoid; break-inside:avoid; }
  figure, .note, .case, .key, .card, .legend, .finding,
  ol.steps li, ul.plain li, tr { break-inside:avoid; }
  figcaption { break-before:avoid; }
  p { orphans:3; widows:3; }

  /* wide tables have to fit the sheet, header repeated on every page */
  pre { break-inside:avoid; font-size:7.2pt; padding:9px 11px; border-radius:0;
        line-height:1.5; }
  .tablewrap { overflow:visible; border-radius:0; }
  table { font-size:6.8pt; min-width:0; }
  .tablewrap--text table { font-size:7.4pt; }
  .tablewrap--text th, .tablewrap--text td { padding:5px 7px; }
  caption { padding:9px; font-size:6.9pt; }
  th, td { padding:4.2px 5px; white-space:normal; }
  thead { display:table-header-group; }
  tbody th small, td small { font-size:6pt; }

  .chart, .diagram, .card, .keys, .cam__stats, .tablewrap,
  figure img { border-radius:0; }
  /* scale by height, not by box: object-fit:contain printed grey letterbox
     bands down both sides of every wide figure */
  figure img { max-height:150mm; width:auto; max-width:100%;
               align-self:flex-start; background:transparent; }
  figcaption { font-size:6.9pt; }
  .case figure img { max-height:76mm; }   /* two worked examples to a sheet */

  /* one gallery item per sheet */
  .cam { break-before:page; break-inside:avoid; padding:0; border-top:none; gap:11px; }
  .cam:first-child { break-before:auto; }
  .cam figure img { max-height:112mm; }
  .cam__stats { grid-template-columns:repeat(6,1fr); }
  .cam__stats dt { min-height:3.1em; }

  footer { break-before:page; margin-top:0; }
}
</style>"""


#: Sets the theme from ?theme=, hides the bar under ?embed=1 (the Studio frames
#: the note itself and puts these controls in its own top bar), and turns the
#: PDF link into a fetch so the button can say what it is doing - the render
#: takes a Chrome round trip, up to a minute for the note with 45 figures.
SCRIPT = r"""<script>
(function () {
  var q = new URLSearchParams(location.search);
  var theme = q.get("theme");
  if (theme === "dark" || theme === "light")
    document.documentElement.dataset.theme = theme;
  if (q.get("embed") === "1") document.documentElement.classList.add("embed");

  document.addEventListener("DOMContentLoaded", function () {
    var bar = document.querySelector(".docbar");
    if (!bar) return;
    if (q.get("embed") === "1") { bar.remove(); return; }
    var btn = bar.querySelector("[data-pdf]");
    // built on disk and opened as a file:// URL: nothing to ask for the PDF
    if (btn && location.protocol.indexOf("http") !== 0) btn.remove();
    if (!btn) return;
    btn.addEventListener("click", function () { downloadPdf(btn); });
  });

  function downloadPdf(btn) {
    if (btn.disabled) return;
    var label = btn.querySelector("[data-label]");
    var was = label.textContent;
    btn.disabled = true; btn.classList.add("busy"); label.textContent = "Rendering…";
    fetch(btn.dataset.pdf)
      .then(function (r) {
        if (!r.ok) return r.text().then(function (t) { throw new Error(t || r.status); });
        return r.blob();
      })
      .then(function (blob) {
        var url = URL.createObjectURL(blob);
        var a = document.createElement("a");
        a.href = url; a.download = btn.dataset.file;
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(function () { URL.revokeObjectURL(url); }, 60000);
        label.textContent = was;
      })
      .catch(function (e) { label.textContent = "PDF failed"; console.error(e); })
      .finally(function () { btn.disabled = false; btn.classList.remove("busy"); });
  }
})();
</script>"""

_PDF_ICON = ('<svg viewBox="0 0 18 18" fill="none" aria-hidden="true">'
             '<path d="M9 2 V11.5 M5.4 8 L9 11.6 L12.6 8" stroke="currentColor" '
             'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
             '<path d="M3 13.4 V15 H15 V13.4" stroke="currentColor" '
             'stroke-width="1.5" stroke-linecap="round"/></svg>')


def head(title: str, key: str, filename: str) -> str:
    """Open the document: title, sheet, script, the bar, and <div class="wrap">.

    `key` is the note's key in tools/notes.py, which is also its route in the
    Studio; `filename` is what the browser saves the PDF as. `title` is plain
    text - the registry holds it that way because the sidebar and the running
    foot need it unescaped - so it is escaped here.
    """
    title = escape(title)
    return f"""<title>{title} · ARSI Studio</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
{STYLE}
{SCRIPT}
<div class="docbar">
  <div class="docbar__in">
    <span class="mark">A</span>
    <span class="docbar__app">ARSI Studio</span>
    <span class="docbar__sep">/</span>
    <span class="docbar__name">{title}</span>
    <span class="docbar__sp"></span>
    <button class="btn btn--go" data-pdf="/notes/{key}.pdf" data-file="{filename}">
      {_PDF_ICON}<span data-label>Download PDF</span></button>
  </div>
</div>
<div class="wrap">"""


def foot(*lines: str) -> str:
    """Close the document with the provenance lines every note ends on."""
    rows = "\n  ".join(f"<span>{ln}</span>" for ln in lines)
    return f"""
<footer>
  {rows}
</footer>

</div>
"""
