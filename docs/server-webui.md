# Web UI design (theme 2.0)

The rules the page `src/webui/index.html` follows to look like itself. What it does and which API it calls lives in
[server-http-apis](server-http-apis.md#chat-page-get--get-chat-get-dashboard-get-models-get-settings-get-downloads).
Guard: `tests/test_webui_theme.cjs` (run by `tests/test_webui.sh`).

## Rules

- One file, no framework, bundle, web font or CDN; a design change keeps the file growth small.
- Every colour, duration and easing is a custom property in the `:root` block. Light and dark come from the same token
  through `light-dark()`; `data-theme` (`system`, `day`, `night`, stored as `sushi-theme`) only sets `color-scheme`.
- Token names are a contract with the dashboard's charts and any branch that draws in the page: rename none; add new ones.
- Text keeps WCAG AA (4.5:1) on every surface it sits on in both themes; a focus ring or a chart line keeps 3:1.

## Palette

Neutrals are warm in light and cool in dark. Contrast is against the surface the token is used on.

| token | light | dark | use, contrast |
|---|---|---|---|
| `--bg`, `--surface`, `--surface-2`, `--surface-3` | `#f7f5f2` `#ffffff` `#f1eeea` `#e8e4de` | `#0f1115` `#16191f` `#1d2129` `#262b35` | page, cards, wells |
| `--text` | `#1d1b19` | `#eceef1` | 17.2 / 15.1 on surface |
| `--text-2` | `#5c5752` | `#a9b0bc` | 6.2 / 7.4 on surface-2 |
| `--text-3` | `#6b645e` | `#8a93a1` | 5.0 / 5.2 on surface-2 |
| `--accent` (salmon, text-safe) | `#b3452a` | `#ffa286` | 4.8 / 8.3 on surface-2, 4.9 / 7.5 on accent-soft |
| `--accent-fill` (light salmon) | `#f4896b` | `#ff9c7e` | fills only, under `--on-accent-fill` |
| `--on-accent-fill` | `#3d1408` | `#2b1209` | 6.6 / 8.6 on the fill, 5.7 / 10.2 on its hover |
| `--accent-soft` | `#fdeee8` | `#3a231c` | selected rows and conversations |
| `--focus` | = `--accent` | = `--accent` | 5.5 / 9.0 on surface |

Status colours (`--info`, `--ok`, `--warn`, `--err` and their `-soft` washes) are unchanged from 1.x.

**Salmon is a hint, not a flood.** Beside the logo's own salmon it appears only in: the primary button and Send (light salmon fill, dark
ink in both themes), the focus ring, the current tab's icon, the composer's focus edge, selected rows (wash + 3 px edge),
unsaved and new-pack dots, the chat's SSD meter and the decode series (`--c1`). Headings, links, tiles and charts'
other series stay neutral or keep their status colour.

**Charts**: series `--c1` (salmon: decode, clock, footprint) `#dc5a36` / `#ff9c7e` (3.8 / 8.6 on surface), `--c2` blue,
`--c3` green, `--c4` violet; `--chart-grid` (= surface-2), `--chart-axis` (= text-2), `--chart-ref` (= warn, dashed
ceilings and restarts). Tooltips: `--tip-bg`, `--tip-border`, `--tip-text`, `--tip-muted`; the chart tooltip `#dashTip`
and the page tooltip share them.

## Motion

| token | value | for |
|---|---|---|
| `--dur-1` | 120 ms | hover, press, colour, tooltip in/out |
| `--dur-2` | 180 ms | view cross-fade, menus and dialogs (`pop-in`: fade + 4 px, scale .98), toasts, status dots |
| `--dur-3` | 240 ms | drawers, the Installed/Available thumb, meters, theme cross-fade |
| `--ease-out` | `cubic-bezier(.2, .8, .2, 1)` | everything |

- Views cross-fade by opacity only (no transform, so layout and measurements never shift). A theme the user picks
  cross-fades the colours (`theme-fade` on `<html>` for 300 ms); the first paint and a system change do not.
- `prefers-reduced-motion: reduce` sets the three durations to 0 and zeroes every transition and animation (spinners
  stand still), so no rule can forget it.
- Dashboard charts move only to show a change: bars, meters and histogram columns ease to a new value (`--dur-3`, the
  poll updates the nodes in place); a canvas chart's time axis slides to the newest sample and its scale eases to a new
  top over 400 ms, the gridlines keeping their round values. All canvas motion goes through one helper (`eased`), which
  moves only after a poll that brought activity (a request running, waiting or finished) and jumps under reduced motion
  or in a hidden tab, so an idle or background page draws no extra frame. Tile numbers never count up: a tile is read as
  an exact value. Guard: `dashboardV2` in `tests/test_webui_dashboard.cjs`.

## Logo

- One logo: the nigiri-on-Mac-Studio art (`LOGO`, an inline 96 px webp) on its white tile, set on every `img.mark`
  (top bar, empty chat, API-key card) and as the favicon. Theme 2.0 keeps it as it was; the server's state is the
  status dot's job, not the logo's.

## Tooltips

- One component: any element with `data-tip="plain text"` shares the single `#tip` (`role="tooltip"`; the anchor gets
  `aria-describedby="tip"` while it shows). No HTML in a tip; `\n` breaks a line; `data-term="…"` adds the "Technical name" line.
- Opens on hover after 400 ms (at once when moving from one tip to the next, or on touch) and on keyboard focus at once
  (`:focus-visible`). The pointer may move onto it. Leaving both, blur, Escape (focus stays), a press, a scroll of its
  container, a resize or a shortcut close it.
- Placed above its anchor, below when the top bar is in the way, and beside it (left) inside a `data-tip-side` container
  such as the chat's settings column, whose fields stack; clamped 8 px inside the window; never over the anchor.
- Copy is for someone who is not an engineer: the visible label says what a thing does for the user in plain words
  (Generation speed, Prompt cache, tokens/s), and its `data-tip` says what it is, why it matters and a sensible value.
  Prefer the term users know from other LLM apps (prompt cache, speculative decoding, KV cache, vision) over an invented
  phrase; invent only when no common term exists.
  Where the full form would wrap and read worse (the eight-tile row), the short form (`tok/s`, Gen speed, sushibar's GEN TOK/S)
  is allowed; its tip spells it out.
  One concept, one label: a chart's name is the same in its title, its card and the Charts picker (guard: `chartAudit`),
  a request step reads Reading prompt / Generating / Cached in the Sessions chips and the compact status (which reads Waiting when requests are only queued), in sentence case.
  sushibar's menu uses the same words (GEN TOK/S, PROMPT TOK/S, `generating…`, `reading prompt…`).
  Reply is no speed, length or state word: thinking and answer are the same generated tokens, so the page and sushibar say
  generation (Generation speed, Generating, Generated); "reply" stays only where it means the answer's text (guard:
  `RETIRED_REPLY` in the chat and dashboard tests, the retired list in sushibar's wording test).
  Anything renamed from an engineering term ends its tip with `Technical name: …` through `data-term` (a settings key
  adds its `model-settings.json` key and launch flag; a label that is already the common term, such as KV cache
  precision, keeps only the key and flag): the term is plain text in a muted child of `#tip`, never in a label.
- A tip adds what the screen cannot say in place: what a metric means, its unit, where the number comes from, what a
  setting does, what a pack needs. Never needed content, an action, or a repeat of visible text.
- An anchor must be focusable: buttons, inputs and labels already are; a tile, fact or table heading takes
  `tabindex="0"`. Text anchors carry `.tip-u` (dotted underline), non-control anchors get the help cursor.
- Where tips are: Compact/Advanced and the theme toggle; every dashboard tile (Compact and Advanced; Memory and Cache
  reuse keep their live `#dashTip`); every dashboard chart's title, the Sessions heading and columns, and the Every and
  Window selects; every bar row and cached-vs-computed column of the dashboard's charts (one Tab stop per card, the
  arrow keys move along it); the chat's Last request rows, its System prompt, Max tokens, Temperature, Top-p, Top-k and
  Thinking budget fields, its context and sees-images badges and the conversation cache section; a model's facts (on
  disk, linked, in memory, max context, sees images, runs partly from SSD); every setting's label on Models and Settings
  and every Saved, Set at launch, Now, Next load and Next start badge; Rescan and Check again; the Available list's
  "Recommended for this Mac" and "Needs N GB" badges and a pack name that says bpw; the Settings cache table headings;
  the Thinking effort card's column headings, Offered boxes, a word's "same as" Sends and the DeepSeek preset chips.
- Guards: the keyboard-tooltip checks in `tests/test_webui_admin_ui.cjs` (every setting and badge) and
  `tests/test_webui_dashboard.cjs` `chartAudit` (every chart title and Sessions column); the `data-term` line in
  `tests/test_webui_theme.cjs`, and the renamed-label lists in the dashboard, chat, admin-ui and effort-table tests.
- Native `title` stays where a full value rides it (a cut-off model name, a path, a server message) and on icon buttons.

## Settings sections and the Thinking effort card

- Settings groups its cards under small uppercase section labels (`.sec-h`): Storage (SSD cache, models folder),
  Inference (Thinking effort), Server (port, update checks, about).
- The Thinking effort card: a `.seg` tab row per family (arrow keys, Home and End move between tabs), then a `.tbl` of
  the six words with Offered (checkbox), Sends (a select of the family's values; DeepSeek a 1-100 box with 50/75/100
  chips), Cap (a box; blank keeps the default, `none` removes it) and Default (radio). Default values show in grey
  (`--text-3`) as the select's first option or the box's placeholder; the copy names the shipped value "default" and
  never "built-in". A word the family has no level for reads "same as xhigh" with the tip "Qwen has no max level, so
  max uses xhigh".
- The selected row (click or focus) takes the selected-row wash and 3 px accent edge; an edited row keeps the edge and
  a small "changed" tag. A preview line under the table shows what a request for the selected word gets, and the exact
  template text. The card lists only the families with a table. A row's error stays in its word cell (`role="alert"`) and its box turns `aria-invalid`; Save stays
  disabled until every row is valid. Badges: Saved (the file holds a change) and Now.
- Under `--think` the Default column is locked and checks the word the flag serves (a save still writes the file's own
  default), and that word must stay offered, as the server requires.

## Dashboard charts

- Fourteen half-width cards, two per row: Prompt speed | Generation speed, Requests | Response time, GPU speed and load |
  Memory, Prompt cache reuse | Prompt cache storage, Draft acceptance | Prompt and generated length, Request outcomes | Why
  requests weren't batched, Prompt cache writes | Tokens verified per round (DSpark only, hidden until picked).
- One axis per chart: two measures of different scale get two charts (prompt and generation speed) or a common base (the GPU
  clock as a share of its top speed beside utilization). Parts of a whole stack, so the top edge is the total (Requests,
  Memory); a subset of a part is a line over the stack (answered together).
- Two series over the same buckets share one histogram, side by side per bucket (first and last token, prompt and
  generated length); a single share is a number, not a two-slice donut (Prompt cache reuse).
- Series colours follow the meaning across cards: prompt `--c2`, generated `--c1`; waiting is `--text-3`.
- An empty card says what it waits for in one line across its width ("No request in this window yet"), never a blank box.
- A merged card's picker name is its title without the unit; a stored choice that names a card it replaced reads as the
  merged card ([server-http-apis](server-http-apis.md#chat-page-get--get-chat-get-dashboard-get-models-get-settings-get-downloads)).

## Dashboard config mode

- The Charts button opens the picker and puts a strip on every chart: drag it, or use its ▲/▼. The arrow at either end
  of the shown order (first chart's ▲, last chart's ▼) is `aria-disabled` and drawn faded, not removed, so the focus
  stays on it; `markEnds` re-reads the ends after every move, tick and Reset.
- Escape anywhere on the page while it is open closes it and returns the focus to the Charts button, unless a key
  handler already took the Escape (an open tip closes first). Closing it, by Escape, Done or the button, ends a drag in
  progress: the marks go and the release drops nothing.
- Guard: `chartPicker` in `tests/test_webui_dashboard.cjs`.
