"""
Plain text tables for console reports.

Every script reports its results as small aligned tables with four
columns: the quantity, the value computed here, the value given in the
reference (a published or measured value, when there is one), and the relative difference
between the two. Keeping the formatting in one place makes the output of
all scripts look the same.
"""

from __future__ import annotations


def print_heading(text):
    """Print a section heading followed by an empty line."""
    print()
    print(text)
    print()


def format_value(value, digits=3):
    """Format a number compactly: fixed point for ordinary sizes, scientific otherwise."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    magnitude = abs(value)
    if magnitude != 0.0 and (magnitude < 1.0e-3 or magnitude >= 1.0e5):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def relative_difference(value, reference):
    """Relative difference to the reference, as text, or an empty string."""
    if not isinstance(value, (int, float)) or not isinstance(reference, (int, float)) or reference == 0:
        return ""
    return f"{100.0 * (value - reference) / abs(reference):+.1f} %"


def print_table(rows, show_reference=True):
    """
    Print rows of (quantity, value, unit) or (quantity, value, unit, reference
    value) as an aligned table. Values may be numbers or ready made strings;
    a number may also be given as (number, digits) to control the precision.
    """
    table = []
    for row in rows:
        quantity, value, unit = row[0], row[1], row[2]
        reference = row[3] if len(row) > 3 else None
        digits = 3
        if isinstance(value, tuple):
            value, digits = value
        number = value
        cells = [quantity, format_value(value, digits), unit]
        if show_reference:
            cells.append(format_value(reference, digits) if reference is not None else "")
            cells.append(relative_difference(number, reference) if reference is not None else "")
        table.append(cells)

    headers = ["Quantity", "This work", "Unit"]
    if show_reference:
        headers += ["Reference", "Difference"]
    widths = [max(len(str(r[i])) for r in table + [headers]) for i in range(len(headers))]
    right_aligned = {1, 3, 4}

    def line(cells):
        parts = []
        for i, cell in enumerate(cells):
            parts.append(str(cell).rjust(widths[i]) if i in right_aligned else str(cell).ljust(widths[i]))
        return "  " + "   ".join(parts).rstrip()

    print(line(headers))
    print()
    for cells in table:
        print(line(cells))


def print_grid(row_header, row_labels, column_labels, values, number_format="{:.2e}"):
    """
    Print a two dimensional table of numbers, for example one row per number
    of cells and one column per numerical scheme. ``values[i][j]`` belongs
    to row ``i`` and column ``j``.
    """
    header = [row_header] + list(column_labels)
    body = [[str(label)] + [number_format.format(v) for v in row] for label, row in zip(row_labels, values)]
    widths = [max(len(r[i]) for r in body + [header]) for i in range(len(header))]

    def line(cells):
        first = cells[0].ljust(widths[0])
        rest = [cell.rjust(widths[i + 1]) for i, cell in enumerate(cells[1:])]
        return "  " + "   ".join([first] + rest)

    print(line(header))
    print()
    for cells in body:
        print(line(cells))
