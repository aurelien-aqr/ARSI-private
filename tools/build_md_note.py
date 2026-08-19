#!/usr/bin/env python3
"""Build a note from a Markdown document, in the shared design.

    venv/bin/python tools/build_md_note.py                # every markdown note
    venv/bin/python tools/build_md_note.py lora-plan      # one

Two of the notes were written as Markdown and are still edited as Markdown, so
they are converted rather than rewritten: the .md file stays the source of
truth, and running this again picks up whatever was last written there. The
other three notes are assembled from measurements by their own builders, which
is why they are Python and these are not.

The conversion is a parse, not a search-and-replace. markdown-it-py reads the
document into tokens and this module renders those tokens with the house
classes - a table becomes .tablewrap, a blockquote becomes a .note callout, the
lists become the ones the sheet styles - so nothing about the text is rewritten
on the way through. The one structural decision is that each `##` heading opens
a <section>, which is what puts it in the rail beside its own prose.

Output: docs/<NAME>.html, next to the Markdown it came from.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.notes import MARKDOWN_NOTES, NOTES                     # noqa: E402
from tools.report_style import foot, head                         # noqa: E402

try:
    from markdown_it import MarkdownIt
    from markdown_it.common.utils import escapeHtml
    from markdown_it.renderer import RendererHTML
except ImportError:                                               # pragma: no cover
    raise SystemExit("markdown-it-py is missing: venv/bin/pip install markdown-it-py")


#: `docs/PUBLIC_DATASETS.md` written in the prose of one note is a reference to
#: another note in the same sidebar. Rendered as inert code it is a dead end, so
#: the source filenames of notes become links to them.
BY_SOURCE = {n.source.name: n for n in NOTES.values() if n.source}


class NoteRenderer(RendererHTML):
    """markdown-it's HTML renderer, speaking the shared sheet's classes."""

    def table_open(self, tokens, idx, options, env):
        # markdown tables in these documents are prose, not number grids: the
        # cells hold sentences, so they wrap rather than scroll off the side.
        # They stay in the content track rather than breaking out to the full
        # page, so every heading keeps its place in the rail beside its text.
        return '<div class="tablewrap tablewrap--text">\n<table>'

    def table_close(self, tokens, idx, options, env):
        return "</table>\n</div>"

    def bullet_list_open(self, tokens, idx, options, env):
        return '<ul class="plain">'

    def ordered_list_open(self, tokens, idx, options, env):
        return '<ol class="steps">'

    def blockquote_open(self, tokens, idx, options, env):
        return '<div class="note">'

    def blockquote_close(self, tokens, idx, options, env):
        return "</div>"

    def link_open(self, tokens, idx, options, env):
        """External links open in a tab: the Studio frames notes in an iframe,
        and a bare link would load changedetection.net inside the app."""
        token = tokens[idx]
        href = token.attrGet("href") or ""
        if href.startswith(("http://", "https://")):
            token.attrSet("target", "_blank")
            token.attrSet("rel", "noopener")
        return self.renderToken(tokens, idx, options, env)

    def code_inline(self, tokens, idx, options, env):
        """`docs/PUBLIC_DATASETS.md` in the text is a link to that note."""
        text = tokens[idx].content
        note = BY_SOURCE.get(Path(text).name)
        code = f"<code>{escapeHtml(text)}</code>"
        if note is None or note.source.name == env.get("self"):
            return code
        # the Studio route, and _top: clicked inside the iframe, a link to
        # /notes/... would load that note *into the frame*, leaving the app's
        # top bar naming the note the reader just left
        return (f'<a href="/#note/{note.key}" target="_top" '
                f'title="{note.title}">{code}</a>')


#: Blocks that read as prose and share one measure; anything else (a table, a
#: code listing) stands on its own and may use the full width of the track.
PROSE = {"paragraph_open", "bullet_list_open", "ordered_list_open",
         "heading_open", "blockquote_open"}


def _top_level(tokens):
    """The token stream cut into top-level blocks, each as its own list."""
    blocks, depth, cur = [], 0, []
    for t in tokens:
        cur.append(t)
        depth += t.nesting
        if depth == 0:
            blocks.append(cur)
            cur = []
    return blocks


def _render(md, blocks, env):
    """Render blocks, wrapping runs of prose in a .col so lines stay readable."""
    out, run = [], []

    def flush():
        if run:
            out.append('<div class="col">'
                       + md.renderer.render(run, md.options, env) + "</div>")
            run.clear()

    for b in blocks:
        if b[0].type in PROSE:
            run.extend(b)
        else:
            flush()
            out.append(md.renderer.render(b, md.options, env))
    flush()
    return "\n".join(out)


def build(key: str) -> str:
    note = NOTES[key]
    md = MarkdownIt("commonmark", renderer_cls=NoteRenderer).enable("table")
    env = {"self": note.source.name}
    blocks = _top_level(md.parse(note.source.read_text(encoding="utf-8")))

    title, lede, intro, sections = "", "", [], []
    for b in blocks:
        head_tok = b[0] if b[0].type == "heading_open" else None
        if head_tok and head_tok.tag == "h1":
            title = md.renderInline(b[1].content, env)
        elif head_tok and head_tok.tag == "h2":
            sections.append([md.renderInline(b[1].content, env), []])
        elif sections:
            sections[-1][1].append(b)
        elif not lede and b[0].type == "paragraph_open":
            lede = md.renderInline(b[1].content, env)      # the opening paragraph
        else:
            intro.append(b)

    body = "\n".join(f"""
<section>
  <div class="sect-head"><h2>{h}</h2></div>
  {_render(md, blocks, env)}
</section>""" for h, blocks in sections)

    return f"""{head(**note.head_args)}

<header class="masthead">
  <span class="eyebrow">{note.eyebrow}</span>
  <h1>{title}</h1>
  <p class="lede">{lede}</p>
</header>
{_render(md, intro, env) if intro else ""}
{body}
{foot(note.running_head, f"Source: docs/{note.source.name}, converted by tools/build_md_note.py")}
"""


def main(keys):
    for key in keys:
        if key not in MARKDOWN_NOTES:
            raise SystemExit(f"{key!r} is not a markdown note; known: "
                             + ", ".join(MARKDOWN_NOTES))
    for key in keys:
        note = NOTES[key]
        note.html.write_text(build(key), encoding="utf-8")
        print(f"{note.html}  ({note.html.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main(sys.argv[1:] or list(MARKDOWN_NOTES))
