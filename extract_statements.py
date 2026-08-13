# -*- coding: utf-8 -*-
"""Extract reported financials from the published statement PDFs.

The PDFs in `public_financial_statements/` carry no text layer - every one is
"Microsoft: Print To PDF" output where the glyphs are vector paths, so
pdfplumber/camelot/tabula all return nothing. They are, however, vector rather
than scanned, so a 300dpi render is crisp and OCR is reliable.

Pipeline:

    render      pdftoppm -> one PNG per page (cached under outputs/pages/)
    orient      tesseract OSD -> rotate the landscape sheets upright
    tables      img2table (OpenCV geometry + Tesseract cells) -> a DataFrame
                per detected table
    tidy        wide tables -> one long frame of (label, column, value) rows

The orient step is load-bearing, not cosmetic: 14 of the 39 packs are portrait
pages with the content drawn sideways, and img2table's own `detect_rotation`
does not recover them (Sanlam yields 3 values a page through it, 500+ when the
page is rotated upright first).

Stage 2, `--map`, is deliberately separate: turning an issuer's own wording
("Direct expenses", "Staff expenses") into the benchmark's line items is a
judgement call, so the script writes a mapping stub for a human to fill in
rather than guessing.

Usage:
    python extract_statements.py --limit 3        # smoke test on 3 PDFs
    python extract_statements.py                  # all 39
    python extract_statements.py --map            # build the mapping stub
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "public_financial_statements"
PAGE_DIR = BASE_DIR / "outputs" / "pages"
OUT_DIR = BASE_DIR / "outputs" / "extracted"

RENDER_DPI = 300

# Filename prefix -> entity. Explicit rather than fuzzy-matched: the filenames
# carry typos ("BHB group", "Shopright") that a similarity match would get
# wrong silently.
FILE_ENTITY = {
    "Anglo American mining": ("E03", "Anglo American"),
    "AngloGold Ashanti": ("E04", "AngloGold Ashanti"),
    "Aspen Pharmacare": ("E19", "Aspen Pharmacare"),
    "BHB group mining": ("E01", "BHP Group"),
    "Bid Corporation": ("E10", "Bid Corporation"),
    "Clicks group": ("E12", "Clicks Group"),
    "Glencore mining": ("E02", "Glencore"),
    "Gold Fields mining": ("E05", "Gold Fields"),
    "MTN Group": ("E16", "MTN Group"),
    "Naspers": ("E15", "Naspers"),
    "NEPI Rockcastle": ("E13", "NEPI Rockcastle"),
    "Outsurance group": ("E07", "OUTsurance Group"),
    "Pepkor Holdings": ("E11", "Pepkor Holdings"),
    "Prosus": ("E14", "Prosus"),
    "Sanlam": ("E08", "Sanlam"),
    "Shaftesbury Capital plc": ("E20", "Shaftesbury Capital plc"),
    "Shopright Holdings": ("E09", "Shoprite Holdings"),
    "Shopright": ("E09", "Shoprite Holdings"),
    "The Bidvest Group": ("E18", "The Bidvest Group"),
    "Valterra Platinum mining": ("E06", "Valterra Platinum"),
    "Vodacom Group": ("E17", "Vodacom Group"),
}

# Reporting currency per entity, for the normalisation step. The proxy
# statements are in rands, so anything else needs converting before
# compare_to_reported() can use it.
ENTITY_CURRENCY = {
    "E01": "USD",  # BHP
    "E02": "USD",  # Glencore
    "E03": "USD",  # Anglo American
    "E04": "USD",  # AngloGold Ashanti
    "E05": "USD",  # Gold Fields
    "E06": "ZAR",  # Valterra Platinum
    "E13": "EUR",  # NEPI Rockcastle
    "E14": "USD",  # Prosus
    "E20": "GBP",  # Shaftesbury Capital
}  # everything else defaults to ZAR

UNIT_PATTERNS = [
    (re.compile(r"\bR\s?million|\bRm\b|\bR'?m\b", re.I), ("ZAR", 1e6)),
    (re.compile(r"\bR\s?000|\bR'000\b", re.I), ("ZAR", 1e3)),
    (re.compile(r"\bUS\$\s?m|\$\s?million|\bUSDm\b", re.I), ("USD", 1e6)),
    (re.compile(r"\b€\s?million|\bEURm\b", re.I), ("EUR", 1e6)),
    (re.compile(r"\b£\s?million|\bGBPm\b", re.I), ("GBP", 1e6)),
]

# A financial figure: optional bracket-negative, digits with space/comma
# thousands separators, optional decimals. "-" and "–" mean nil.
NUMERIC = re.compile(r"^\(?\s*[-–—]?\s*[\d][\d\s ,.]*\)?$")


def entity_for(path: Path) -> tuple[str, str]:
    """Map a PDF filename to (entity_id, entity_name).

    Underscores are treated as spaces so the mapping keys stay readable and
    match whether the files are named "Anglo American mining - 22&23.pdf" or
    "Anglo_American_mining_-_22&23.pdf".
    """
    stem = path.stem.replace("_", " ")
    # Longest prefix first so "Shopright Holdings" beats "Shopright".
    for prefix in sorted(FILE_ENTITY, key=len, reverse=True):
        if stem.startswith(prefix):
            return FILE_ENTITY[prefix]
    raise KeyError(f"No entity mapping for {path.name!r} - add it to FILE_ENTITY")


def parse_amount(raw: str) -> float | None:
    """Parse a reported figure, honouring bracket-negatives and nil dashes.

    Returns None for anything that is not a number, so note references and
    stray header text drop out rather than becoming zeros.
    """
    if raw is None:
        return None
    text = str(raw).strip().replace(" ", " ")
    if text in {"", "-", "–", "—"}:
        return 0.0
    if not NUMERIC.match(text):
        return None
    negative = text.startswith("(") and text.endswith(")")
    digits = re.sub(r"[^\d.]", "", text)
    if not digits or digits == ".":
        return None
    try:
        value = float(digits)
    except ValueError:
        return None
    return -value if negative else value


def detect_unit(text: str) -> tuple[str | None, float | None]:
    """Pull the currency and scale out of a table header or page caption."""
    for pattern, (currency, scale) in UNIT_PATTERNS:
        if pattern.search(text):
            return currency, scale
    return None, None


def render_pages(pdf: Path, dpi: int = RENDER_DPI) -> list[Path]:
    """Render every page to PNG, caching so reruns are free."""
    # Keyed by dpi so that changing resolution doesn't silently reuse a cache
    # rendered at the old one.
    out = PAGE_DIR / pdf.stem / f"{dpi}dpi"
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in out.glob("page-*.png") if not p.stem.endswith("-rot"))
    if existing:
        return existing
    subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-png", str(pdf), str(out / "page")],
        check=True,
        capture_output=True,
    )
    return sorted(out.glob("page-*.png"))


def orient(image: Path) -> Path:
    """Rotate a page upright if tesseract's OSD says it is sideways.

    The landscape sheets (Sanlam, several of the mining packs) are portrait A4
    pages with the content drawn at 90 degrees - the PDF page box is upright,
    so only the pixels reveal it. Rotated pages are written alongside the
    original with a `-rot` suffix.
    """
    rotated = image.with_name(image.stem + "-rot.png")
    if rotated.exists():
        return rotated

    probe = subprocess.run(
        ["tesseract", str(image), "-", "--psm", "0"],
        capture_output=True,
        text=True,
    )
    match = re.search(r"Rotate:\s*(\d+)", probe.stdout)
    angle = int(match.group(1)) if match else 0
    if angle == 0:
        return image

    from PIL import Image

    with Image.open(image) as img:
        img.rotate(-angle, expand=True).save(rotated)
    return rotated


def build_ocr(threads: int = 4):
    """Tesseract instance for img2table.

    img2table finds table geometry with OpenCV and reads the cells with this,
    which on these pages is measurably more accurate than a full-page OCR pass:
    on Vodacom's income statement it reads `Operating profit 35 791` correctly
    where a full-page pass misreads the leading digit.
    """
    from img2table.ocr import TesseractOCR

    return TesseractOCR(n_threads=threads, lang="eng")


def promote_header(frame: pd.DataFrame) -> pd.DataFrame:
    """img2table returns positional columns with the header as row 0.

    Only promote when row 0 actually looks like a header - i.e. none of its
    cells past the label parse as a figure - so a table whose first row is
    already data is left alone.
    """
    if frame.empty:
        return frame
    first = frame.iloc[0]
    if any(parse_amount(cell) is not None for cell in first[1:]):
        return frame
    promoted = frame.iloc[1:].copy()
    promoted.columns = [str(c).replace("\n", " ").strip() for c in first]
    return promoted


def tidy(frame: pd.DataFrame, **context) -> pd.DataFrame:
    """Melt one detected table into (row_label, column_header, value) rows.

    The first column of a statement table is the line-item label; every other
    column is a period. Non-numeric cells (note references, blank spacers) are
    dropped by parse_amount returning None.
    """
    if frame.empty or frame.shape[1] < 2:
        return pd.DataFrame()

    frame = frame.copy()

    # Header cells arrive blank or repeated (a statement's period columns often
    # share one spanning header), which melt rejects outright. Make them unique
    # and positional before doing anything else.
    columns, seen = [], {}
    for position, raw in enumerate(frame.columns):
        name = str(raw).strip() or f"col_{position}"
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        columns.append(name)
    frame.columns = columns

    # The notes column is small integers that would otherwise parse as figures.
    keep = [c for c in columns[1:] if not re.match(r"^notes?\b", c, re.I)]
    label_col = columns[0]
    frame = frame[[label_col] + keep]
    if not keep:
        return pd.DataFrame()

    long = frame.melt(
        id_vars=[label_col], var_name="column_header", value_name="value_raw"
    ).rename(columns={label_col: "row_label"})

    long["row_label"] = long["row_label"].astype(str).str.strip()
    long["value"] = long["value_raw"].map(parse_amount)
    long = long[long["value"].notna() & long["row_label"].ne("")]

    for key, value in context.items():
        long[key] = value
    return long


def extract_pdf(pdf: Path, ocr, dpi: int = RENDER_DPI) -> pd.DataFrame:
    """Run the full render -> tables -> tidy chain for one PDF.

    Rotation is left to img2table's own `detect_rotation` rather than the OSD
    pass, so the sideways sheets are handled in one place instead of risking a
    double rotation.
    """
    from img2table.document import Image as I2TImage

    entity_id, entity_name = entity_for(pdf)
    rows: list[pd.DataFrame] = []

    for page_no, image in enumerate(render_pages(pdf, dpi=dpi), start=1):
        # img2table's own detect_rotation does not cope with these sheets - the
        # landscape packs yielded 3 values a page through it, against 500+ from
        # the same pages pre-rotated by OSD. So rotate first, then hand it an
        # upright image and leave its detection off.
        upright = orient(image)
        try:
            tables = I2TImage(src=str(upright), detect_rotation=False).extract_tables(
                ocr=ocr,
                implicit_rows=True,
                borderless_tables=True,
                min_confidence=50,
            )
        except TypeError as exc:
            # img2table joins the OCR words of a table's title area without
            # guarding against a None value, which Tesseract does occasionally
            # return (Naspers 25&26 p.1). Contain it per page so one unreadable
            # caption does not cost the whole document.
            print(f"    page {page_no} skipped: {exc}", file=sys.stderr)
            continue

        for table_no, table in enumerate(tables):
            wide = promote_header(table.df)
            header_text = " ".join(str(c) for c in wide.columns)
            currency, scale = detect_unit(f"{table.title or ''} {header_text}")
            rows.append(
                tidy(
                    wide,
                    entity_id=entity_id,
                    entity_name=entity_name,
                    source_file=pdf.name,
                    page=page_no,
                    table_index=table_no,
                    table_title=table.title or "",
                    currency=currency or ENTITY_CURRENCY.get(entity_id, "ZAR"),
                    unit_scale=scale,
                )
            )

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_mapping_stub(extracted: pd.DataFrame, path: Path) -> Path:
    """Write every distinct row label out for a human to map to a line item.

    Left blank, a label is simply skipped downstream - so filling in only the
    ten comparable lines per entity is enough.
    """
    labels = (
        extracted.groupby(["entity_id", "entity_name", "row_label"])
        .size()
        .reset_index(name="occurrences")
        .sort_values(["entity_id", "occurrences"], ascending=[True, False])
    )
    labels["line_item"] = ""
    labels["metric_type"] = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="only process the first N PDFs")
    parser.add_argument("--pdf", action="append", help="process a named PDF (repeatable)")
    parser.add_argument("--map", action="store_true", help="write the label-mapping stub")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="render pages to PNG and stop (no OCR, no table detection)",
    )
    parser.add_argument(
        "--dpi", type=int, default=RENDER_DPI, help=f"render resolution (default {RENDER_DPI})"
    )
    parser.add_argument(
        "--orient",
        action="store_true",
        help="with --render-only, also write upright copies of sideways pages",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = OUT_DIR / "extracted_lines.csv"

    if args.map:
        if not combined_path.exists():
            sys.exit(f"{combined_path} not found - run the extraction first")
        extracted = pd.read_csv(combined_path)
        stub = build_mapping_stub(extracted, OUT_DIR / "label_mapping.csv")
        print(f"Wrote {stub} - fill in line_item/metric_type, blanks are skipped")
        return

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if args.pdf:
        wanted = {name.lower() for name in args.pdf}
        pdfs = [p for p in pdfs if p.name.lower() in wanted or p.stem.lower() in wanted]
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        sys.exit("No PDFs matched")

    if args.render_only:
        total = 0
        for index, pdf in enumerate(pdfs, start=1):
            pages = render_pages(pdf, dpi=args.dpi)
            rotated = 0
            if args.orient:
                rotated = sum(1 for p in pages if orient(p) != p)
            total += len(pages)
            suffix = f", {rotated} rotated" if args.orient else ""
            print(f"[{index}/{len(pdfs)}] {pdf.name}: {len(pages)} pages{suffix}", flush=True)
        print(f"\n{total} pages at {args.dpi}dpi -> {PAGE_DIR}")
        return

    ocr = build_ocr()
    frames = []
    for index, pdf in enumerate(pdfs, start=1):
        print(f"[{index}/{len(pdfs)}] {pdf.name}", flush=True)
        try:
            frame = extract_pdf(pdf, ocr, dpi=args.dpi)
        except Exception as exc:  # keep going; one bad pack shouldn't stop the run
            print(f"    FAILED: {exc}", file=sys.stderr)
            continue
        if frame.empty:
            print("    no tables detected", file=sys.stderr)
            continue
        frame.to_csv(OUT_DIR / f"{pdf.stem}.csv", index=False)
        frames.append(frame)
        print(f"    {len(frame)} values from {frame['page'].nunique()} pages")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(combined_path, index=False)
        print(f"\n{len(combined)} values -> {combined_path}")


if __name__ == "__main__":
    main()
