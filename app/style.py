"""How the page looks.

The visual language is the supermarket shelf label. In Spain a shelf tag shows
the product, then the price, and underneath in small print the price per kilo,
which is what the law requires and what actually lets you compare two things.
Our page shows the same three things in the same order, because that is the
comparison a shopper is already used to reading.

One element is loud: the similarity figure, set large with a measured bar
under it. Everything else is deliberately quiet, so that the number a person
came for is the thing their eye lands on. The only colour that is not ink or
paper marks a saving, and the shelf-tag yellow appears once, on the edge of
the product you picked, to say "this is yours".
"""

FONT = ('https://fonts.googleapis.com/css2?'
        'family=Archivo:wght@400;500;600;800&display=swap')

CSS = f"""
<style>
@import url('{FONT}');

:root {{
  --paper:  #F6F7F4;   /* cool off-white, closer to milk than to cream */
  --card:   #FFFFFF;
  --ink:    #17201C;
  --muted:  #6B7570;
  --line:   #E3E6E0;
  --save:   #2F6B46;   /* the only colour that means something: cheaper */
  --mark:   #F2C53D;   /* shelf-tag yellow, used once */
}}

html, body, .stApp, .stApp p, .stApp label, .stApp div, .stApp span,
.stApp button, .stApp input, .stApp h1, .stApp h2, .stApp h3 {{
  font-family: 'Archivo', system-ui, -apple-system, sans-serif;
}}
/* Streamlit draws its arrows and chevrons as ligatures in an icon font. The
   rule above would replace that font and print the ligature names as words,
   so the icons keep their own font. */
.stApp [data-testid="stIconMaterial"],
.stApp .material-icons, .stApp .material-icons-outlined,
.stApp span[class*="material"] {{
  font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}}
.stApp {{ background: var(--paper); }}

/* Streamlit's default top padding pushes everything below the fold. The width
   is held in at a readable measure: wider than this and the price drifts so
   far from the product name that the two stop reading as one label. */
.block-container {{ padding-top: 2.5rem; padding-bottom: 4rem; max-width: 980px; }}
header[data-testid="stHeader"] {{ background: transparent; }}

h1.masthead {{
  font-size: clamp(1.9rem, 4vw, 2.9rem);
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.05;
  color: var(--ink);
  margin: 0 0 .4rem 0;
}}
p.standfirst {{
  color: var(--muted);
  font-size: 1rem;
  max-width: 54ch;
  line-height: 1.5;
  margin: 0 0 1.8rem 0;
}}

/* The product you picked. The yellow edge is the only one on the page. */
.yours {{
  background: var(--card);
  border: 1px solid var(--line);
  border-left: 5px solid var(--mark);
  border-radius: 3px;
  padding: 1.15rem 1.35rem;
  margin: .4rem 0 2.2rem 0;
}}
.yours .name {{ font-size: 1.3rem; font-weight: 600; letter-spacing: -.015em; }}
.yours .meta {{ color: var(--muted); font-size: .85rem; margin-top: .2rem; }}
.yours .ing  {{ color: var(--muted); font-size: .8rem; line-height: 1.45;
                margin-top: .7rem; max-width: 90ch; }}

.sectionhead {{
  font-size: .95rem; font-weight: 600; color: var(--ink);
  margin: 0 0 .7rem 0; padding-bottom: .5rem;
  border-bottom: 1px solid var(--line);
}}

/* One suggestion. Grid: the score, the product, the price. Reading order on a
   phone is the same, stacked. */
.tag {{
  display: grid;
  grid-template-columns: 78px minmax(0, 1fr) 132px;
  gap: 1.1rem;
  align-items: start;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 3px;
  padding: 1rem 1.2rem;
  margin-bottom: .6rem;
}}
.score     {{ padding-top: .1rem; }}
.score .n  {{ font-size: 1.85rem; font-weight: 800; letter-spacing: -.045em;
              line-height: 1; color: var(--ink); font-variant-numeric: tabular-nums; }}
.score .n span {{ font-size: .85rem; font-weight: 500; margin-left: 1px; }}
.bar       {{ height: 4px; background: var(--line); margin-top: .5rem;
              border-radius: 2px; overflow: hidden; }}
.bar i     {{ display: block; height: 4px; background: var(--ink); }}

.tag .name {{ font-size: 1rem; font-weight: 600; line-height: 1.3;
              letter-spacing: -.01em; }}
.tag .meta {{ color: var(--muted); font-size: .82rem; margin-top: .15rem; }}
.tag .ing  {{ color: var(--muted); font-size: .78rem; line-height: 1.45;
              margin-top: .5rem; max-width: 66ch; }}

.own {{
  display: inline-block; font-size: .68rem; font-weight: 600;
  border: 1px solid var(--line); border-radius: 2px;
  padding: .08rem .35rem; margin-left: .4rem; color: var(--muted);
  vertical-align: 1px;
}}

/* Price block, right-aligned like the right-hand side of a shelf tag. */
.price      {{ text-align: right; white-space: nowrap; padding-top: .1rem; }}
.price .big {{ font-size: 1.05rem; font-weight: 600;
               font-variant-numeric: tabular-nums; }}
.price .kg  {{ font-size: .78rem; color: var(--muted); margin-top: .1rem;
               font-variant-numeric: tabular-nums; }}
.price .nil {{ font-size: .82rem; color: var(--muted); }}
.price .save,
.price .more {{ font-size: .82rem; margin-top: .45rem;
                font-variant-numeric: tabular-nums; }}
.price .save {{ color: var(--save); font-weight: 600; }}
.price .more {{ color: var(--muted); }}

@media (max-width: 640px) {{
  .tag {{ grid-template-columns: 56px 1fr; }}
  .price {{ grid-column: 2; text-align: left; margin-top: .6rem; }}
}}

/* Sidebar: quieter than the page, since nothing in it is the answer. */
section[data-testid="stSidebar"] {{
  background: var(--card); border-right: 1px solid var(--line);
}}
section[data-testid="stSidebar"] .block-container {{ padding-top: 2rem; }}

.note {{ color: var(--muted); font-size: .78rem; line-height: 1.55;
         border-top: 1px solid var(--line); padding-top: 1rem; margin-top: 2.2rem;
         max-width: 72ch; }}

@media (prefers-reduced-motion: reduce) {{
  * {{ animation: none !important; transition: none !important; }}
}}
</style>
"""


def euros(value):
    """1.79 -> '1,79 €'. Spanish shelf labels put the comma in the middle and
    the symbol at the end, and every price here is from a Spanish shop."""
    return f'{value:,.2f} €'.replace(',', '~').replace('.', ',').replace('~', '.')
