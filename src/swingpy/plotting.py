"""
Figures of SwingPy (they need matplotlib: pip install matplotlib).

Every figure follows one engineering journal style. The axes are open (no
top or right spine), the ticks point outwards, there is no grid, the type is
an Arial-metric sans serif, legends have no frame, panel letters stand on
the title line, and each figure is saved twice: as a 300 dpi PNG and, next to
it, as a vector PDF.

The figures carry no colour except in the position-time maps. Curves are
told apart by their grey level and their line style. Results of the model
are drawn as lines without markers; open markers (outline only, no fill) are
kept for measured or published values, where there is no line to draw.
Maps use the perceptually uniform viridis scale with labelled contours, as
recommended by Crameri, Shephard and Heron, Nat. Commun. 11 (2020) 5444.

Cycle steps are named by four-letter codes (see ``STEP_CODES``): PRES for
pressurization with feed, FEED for the feed (adsorption) step, BLOW for
blowdown, EVAC for evacuation, and so on. A figure writes the name a step
was given; when a long name does not fit, its code is written instead.
"""

from __future__ import annotations

import os
import re

import numpy as np

INK = "#000000"
"""Colour of single curves, reference lines, frames and text."""

GREYS = ["#000000", "#555555", "#8C8C8C", "#B8B8B8"]
"""Grey levels of the curves of a line plot, darkest first."""

REFERENCE = "#6B6B6B"
"""Colour of reference and tolerance lines."""

LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1.5, 1, 1.5, 1, 1.5)), (0, (8, 2)), (0, (3, 1, 1, 1)), (0, (1, 1))]
"""Line styles of the curves of a line plot, in order."""

MARKER_CYCLE = ["o", "s", "^", "D", "v", "p", "x", "+"]
"""Open markers of measured or published values, in order."""

SEQUENTIAL = "viridis"
"""Colour scale of the position-time maps."""

STEP_CODES = {
    "PRES": "pressurization",
    "FEED": "feed (adsorption)",
    "BLOW": "blowdown",
    "EVAC": "evacuation",
    "PURG": "purge",
    "LPPR": "light product pressurization",
    "EQDN": "pressure equalization, depressurization (donor)",
    "EQUP": "pressure equalization, pressurization (receiver)",
    "LREF": "light reflux",
    "HREF": "heavy reflux",
    "DESO": "desorption",
    "DEPR": "co-current depressurization",
    "RINS": "rinse",
    "RECY": "recycle",
    "SUPP": "supply of purge to another column",
    "IDLE": "idle",
}
"""The four-letter codes of cycle steps and their meaning. Numbered steps of
the same kind take the digit in the fourth place: EQD1 and EQD2 are the first
and second equalization depressurizations and EQU1 and EQU2 the matching
pressurizations."""

_NUMBERED = {"EQD": "pressure equalization, depressurization", "EQU": "pressure equalization, pressurization",
             "PUR": "purge", "BLO": "blowdown", "EVA": "evacuation"}

_LONG_TO_CODE = {"pressurization": "PRES", "adsorption": "FEED", "feed": "FEED", "blowdown": "BLOW",
                 "evacuation": "EVAC", "purge": "PURG", "rinse": "RINS",
                 "desorption": "DESO", "recycle": "RECY", "idle": "IDLE"}

STEP_LINE = dict(color="#7A7A7A", linewidth=0.7, linestyle=(0, (1.5, 2.0)))
"""Style of the dotted lines that separate the steps of a time history."""

CYCLE_LINE = dict(color="#444444", linewidth=0.8, linestyle=(0, (5, 3)))
"""Style of the dashed lines that separate the cycles of a time history."""


def step_meaning(name):
    """
    The meaning of a step code ("EQD1" -> "pressure equalization,
    depressurization 1"), or None when the name is not a known code.
    """
    name = str(name)
    if name in STEP_CODES:
        return STEP_CODES[name]
    if len(name) == 4 and name[3].isdigit() and name[:3] in _NUMBERED:
        return f"{_NUMBERED[name[:3]]} {name[3]}"
    return None


def step_code(name):
    """
    A short form of a step name for a narrow space: the name itself when it is
    already a code of four characters or fewer, the code of a known long name
    (pressurization -> PRES), and otherwise its first four letters in capitals.
    """
    name = str(name)
    if len(name) <= 4:
        return name
    return _LONG_TO_CODE.get(name.lower(), name[:4].upper())


def single_style(points=None, markers=False):
    """
    Style of a lone curve: black and solid. With ``markers`` it carries sparse
    open circles, for measured values drawn as a line.
    """
    style = dict(color=INK, linestyle="-")
    if markers:
        style.update(marker="o", markersize=4.5, markerfacecolor="none", markeredgewidth=1.0,
                     markevery=max(1, (points or 80) // 9))
    return style


def series_style(index, points=None, markers=False):
    """
    Style of the ``index``-th curve of a line plot: a grey level and a line
    style. Results of the model carry no markers. With ``markers`` the curve
    carries sparse open markers (outline only), for measured values drawn as
    a line; ``points`` is the number of points of the curve and sets how
    sparse they are.
    """
    color = GREYS[index % len(GREYS)]
    style = dict(color=color, linestyle=LINESTYLES[index % len(LINESTYLES)])
    if markers:
        every = max(1, (points or 80) // 9)
        style.update(marker=MARKER_CYCLE[index % len(MARKER_CYCLE)], markersize=5.0, markerfacecolor="none",
                     markeredgecolor=color, markeredgewidth=1.0, markevery=(index % 3 * max(1, every // 3), every))
    return style


def open_marker_style(index):
    """
    Style of the ``index``-th set of measured or published values drawn as
    points only: an open marker in the grey level of that index, no line.
    """
    color = GREYS[index % len(GREYS)]
    return dict(linestyle="none", marker=MARKER_CYCLE[index % len(MARKER_CYCLE)], markersize=6.0,
                markerfacecolor="none", markeredgecolor=color, markeredgewidth=1.1)


def formula(name):
    """A component name with the digits of its chemical formula as subscripts: CO2 -> CO$_2$."""
    return re.sub(r"(?<=[A-Za-z)])(\d+)", r"$_{\1}$", str(name))


def symbol(letter, name):
    r"""A quantity of one component in mathtext, for example symbol("y", "CO2") -> $y_{\mathrm{CO_2}}$."""
    tex = re.sub(r"(?<=[A-Za-z)])(\d+)", r"_{\1}", str(name))
    return f"${letter}_{{\\mathrm{{{tex}}}}}$"


_UNIT = re.compile(r"\[([^\]$]*)\]")


def units(text):
    """Powers inside the unit brackets of a label as superscripts: "Flow [m3/h]" -> "Flow [m$^3$/h]"."""
    return _UNIT.sub(lambda m: "[" + re.sub(r"(?<=[A-Za-z])(-?\d+)", r"$^{\1}$", m.group(1)) + "]", str(text))


def apply_style():
    """Set the matplotlib defaults shared by every figure."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 9.5,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.facecolor": "white",
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=[INK]),
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.top": False,
        "ytick.right": False,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.color": INK,
        "ytick.color": INK,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "butt",
        "image.cmap": SEQUENTIAL,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.handlelength": 2.6,
        "figure.facecolor": "white",
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "hatch.linewidth": 0.7,
        "hatch.color": INK,
    })


def label_panels(axes, letters="abcdefgh"):
    """
    Put (a), (b), ... in front of the title of each panel, on the same line,
    or alone when a panel has no title. The letters are applied again when the
    figure is saved, so a title set later keeps its letter.
    """
    for axis, letter in zip(np.ravel(axes), letters):
        axis.swingpy_panel = f"({letter})"
        _panel_title(axis)


def _panel_title(axis):
    """Write the panel letter of ``axis`` in front of its title."""
    letter = getattr(axis, "swingpy_panel", None)
    if not letter:
        return
    title = axis.get_title(loc="left")
    centre = axis.get_title(loc="center")
    if centre and not title:
        axis.set_title("", loc="center")
        title = centre
    if not title.startswith(letter):
        axis.set_title(f"{letter} {title}".rstrip(), loc="left", pad=getattr(axis, "swingpy_title_pad", 6.0))


def _typeset(figure):
    """Panel letters, and powers inside unit brackets of labels, titles and legends as superscripts."""
    for axis in figure.axes:
        _panel_title(axis)
        axis.set_xlabel(units(axis.get_xlabel()))
        axis.set_ylabel(units(axis.get_ylabel()))
        for loc in ("left", "center", "right"):
            title = axis.get_title(loc=loc)
            if title:
                axis.set_title(units(title), loc=loc, pad=getattr(axis, "swingpy_title_pad", 6.0))
    legends = [axis.get_legend() for axis in figure.axes] + list(figure.legends)
    for legend in legends:
        for text in getattr(legend, "texts", []):
            text.set_text(units(text.get_text()))


def save_figure(figure, output_path):
    """Save the figure as a PNG and, next to it, as a vector PDF, then close it."""
    import matplotlib.pyplot as plt
    _typeset(figure)
    figure.savefig(output_path)
    if output_path.lower().endswith(".png"):
        figure.savefig(output_path[:-4] + ".pdf")
    plt.close(figure)


def _profiles(model, state):
    """Pressure (bar), temperature (K), mole fractions and loadings (mol/kg) of every cell."""
    p = model.unpack(state)
    return {
        "P": p["P"] * model.P0 / 1.0e5,
        "T": p["T"] * model.T0,
        "y": model.full_composition(p["y"]),
        "q": p["x"] * model.qs0,
    }


def _cycle_series(model, results):
    """Concatenated time (s) and states of one cycle, with the time span and name of each step."""
    times, states, spans, offset = [], [], [], 0.0
    for r in results:
        if r.time[-1] <= 0.0:
            continue
        times.append(offset + r.time)
        states.append(r.states)
        spans.append((offset, offset + r.time[-1], r.step.name))
        offset += r.time[-1]
    return np.concatenate(times), np.hstack(states), spans


def _step_lines(axes, spans, cycle_ends=()):
    """Dotted lines between the steps and dashed lines between the cycles of a time history."""
    ends = [round(float(c), 9) for c in cycle_ends]
    for axis in np.atleast_1d(axes):
        for start, end, name in spans[:-1]:
            if round(float(end), 9) not in ends:
                axis.axvline(end, **STEP_LINE)
        for boundary in cycle_ends:
            axis.axvline(boundary, **CYCLE_LINE)


def _name_steps(axis, spans, total):
    """
    Write the name of each step above the axes, centred on the step: in full
    where it fits (about 0.9 % of the time axis per character), otherwise its
    code, otherwise nothing. The title of the panel is raised above the names.
    """
    for start, end, name in spans:
        for text in (str(name), step_code(name)):
            if end - start >= 0.009 * total * (len(text) + 1):
                axis.text(0.5 * (start + end), 1.01, text, transform=axis.get_xaxis_transform(),
                          ha="center", va="bottom", fontsize=8, color=INK)
                break
    axis.swingpy_title_pad = 16.0
    title = axis.get_title(loc="left")
    if title:
        axis.set_title(title, loc="left", pad=16.0)


def _mark_steps(axes, spans, label_axis=None):
    """Separate the steps of a time history by dotted lines and name them above ``label_axis``."""
    _step_lines(axes, spans)
    if label_axis is not None:
        _name_steps(label_axis, spans, spans[-1][1] - spans[0][0])


def _lollipops(axis, labels, series, names=None, value_format=None):
    """
    Horizontal lollipops: one row per label, and for every series a stem from
    zero ending in an open marker, in the grey level and marker of the series.
    Several series are offset vertically within a row. ``value_format`` turns
    a value into the text written next to its marker.
    """
    n = len(series)
    rows = np.arange(len(labels))[::-1].astype(float)
    step = 0.6 / max(n, 1)
    span = max(max(abs(float(v)) for values in series for v in values), 1e-12)
    for k, values in enumerate(series):
        y = rows + (n - 1) / 2.0 * step - k * step
        style = open_marker_style(k)
        axis.hlines(y, 0.0, values, color=style["markeredgecolor"], linewidth=1.3)
        axis.plot(values, y, label=None if names is None else names[k], **style)
        if value_format is not None:
            for v, row in zip(values, y):
                axis.text(v + (0.02 if v >= 0 else -0.02) * span, row, value_format(v), va="center",
                          ha="left" if v >= 0 else "right", fontsize=8, color=INK)
    axis.axvline(0.0, color=INK, linewidth=0.8)
    axis.set_yticks(rows)
    axis.set_yticklabels(labels)
    axis.tick_params(axis="y", length=0)
    axis.spines["left"].set_visible(False)
    axis.set_ylim(rows.min() - 0.6, rows.max() + 0.6)


def plot_profiles(model, results, output_path, component=None):
    """
    Column profiles at the end of each step of one cycle: pressure,
    temperature, mole fraction and solid loading along the column.

    Args:
        model: The ColumnModel.
        results: Step results of one cycle (for example ``CycleRun.results``).
        output_path: PNG path; a PDF is saved next to it.
        component: Component shown in the composition and loading panels,
            the first component of the gas by default.
    """
    import matplotlib.pyplot as plt
    apply_style()
    k = model.gas.index(component) if component else 0
    name = model.gas.components[k]
    z = (np.arange(model.n_cells) + 0.5) * model.cell_width
    results = [r for r in results if r.time[-1] > 0.0]
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    for i, r in enumerate(results):
        prof = _profiles(model, r.final_state)
        style = dict(series_style(i), label=r.step.name)
        axes[0, 0].plot(z, prof["P"], **style)
        axes[0, 1].plot(z, prof["T"], **style)
        axes[1, 0].plot(z, prof["y"][k], **style)
        axes[1, 1].plot(z, prof["q"][k], **style)
    labels = ["Pressure [bar]", "Temperature [K]", f"{symbol('y', name)} [-]", f"{symbol('q', name)} [mol/kg]"]
    for axis, label in zip(axes.ravel(), labels):
        axis.set_xlabel("Z [-]")
        axis.set_ylabel(label)
        axis.set_xlim(0.0, 1.0)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.04))
    label_panels(axes.ravel())
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_time_histories(model, cycles, output_path, positions=(0.0, 0.5, 1.0), component=None):
    """
    Pressure, temperature, mole fraction and loading against time over
    several consecutive cycles, at a few positions along the column. Dotted
    lines separate the steps and dashed lines the cycles, and the steps are
    named above the top panel, so the approach to (or the repetition of)
    cyclic steady state can be seen at a glance.

    Args:
        model: The ColumnModel.
        cycles: List of cycles, each a list of step results, oldest first
            (for example ``CycleRun.recent[-3:]``).
        output_path: PNG path; a PDF is saved next to it.
        positions: Dimensionless positions Z where the histories are taken
            (0 is the feed end, 1 the product end); the nearest cell is used.
        component: Component of the composition and loading rows.
    """
    import matplotlib.pyplot as plt
    apply_style()
    k = model.gas.index(component) if component else 0
    name = model.gas.components[k]
    cells = [min(int(z * model.n_cells), model.n_cells - 1) for z in positions]
    time, series, spans = [], {key: [] for key in ("P", "T", "y", "q")}, []
    offset = 0.0
    for cycle in cycles:
        for r in cycle:
            if r.time[-1] <= 0.0:
                continue
            spans.append((offset, offset + r.time[-1], r.step.name))
            for t, state in zip(r.time, r.states.T):
                prof = _profiles(model, state)
                time.append(offset + t)
                series["P"].append(prof["P"][cells])
                series["T"].append(prof["T"][cells])
                series["y"].append(prof["y"][k][cells])
                series["q"].append(prof["q"][k][cells])
            offset += r.time[-1]
    time = np.array(time)
    figure, axes = plt.subplots(4, 1, figsize=(11.0, 9.0), sharex=True)
    labels = {"P": "Pressure [bar]", "T": "Temperature [K]", "y": f"{symbol('y', name)} [-]",
              "q": f"{symbol('q', name)} [mol/kg]"}
    for axis, key in zip(axes, ("P", "T", "y", "q")):
        values = np.array(series[key])
        for i, z in enumerate(positions):
            axis.plot(time, values[:, i], label=f"Z = {z:g}", **series_style(i))
        axis.set_ylabel(labels[key])
    cycle_ends = np.cumsum([sum(r.time[-1] for r in cycle) for cycle in cycles])[:-1]
    _step_lines(axes, spans, cycle_ends)
    _name_steps(axes[0], spans, time[-1] - time[0])
    axes[-1].set_xlabel("Time over the last cycles [s]")
    axes[-1].set_xlim(time[0], time[-1])
    axes[0].legend(loc="upper right", ncol=len(positions), bbox_to_anchor=(1.0, 1.25))
    label_panels(axes)
    figure.align_ylabels(axes)
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_convergence(history, output_path, tolerance=None, profile_tolerance=None):
    """
    Purity and recovery, mass balance error, and profile change of every
    cycle (``CycleRun.history``). Optional tolerances are drawn as dashed lines.
    """
    import matplotlib.pyplot as plt
    apply_style()
    cycles = np.arange(1, len(history) + 1)
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 3.6))
    axes[0].plot(cycles, [h.purity for h in history], label="Purity", **series_style(0))
    axes[0].plot(cycles, [h.recovery for h in history], label="Recovery", **series_style(1))
    axes[0].set_ylabel("Purity, recovery [%]")
    axes[0].legend(loc="best")
    """
    The first cycles often lie far outside the final range (a column that
    starts full of feed gas can give recoveries above 100 %), so the axis is
    scaled on the second half of the run.
    """
    late = [v for h in history[len(history) // 2:] for v in (h.purity, h.recovery)]
    if late:
        spread = max(max(late) - min(late), 1.0)
        axes[0].set_ylim(min(late) - 2.0 * spread, min(max(late) + 2.0 * spread, 100.0 + spread))
    axes[1].semilogy(cycles, [max(h.mass_balance_error, 1e-6) for h in history], **single_style())
    axes[1].set_ylabel("Mass balance error [%]")
    changes = [getattr(h, "profile_change", float("nan")) for h in history]
    axes[2].semilogy(cycles, [max(c, 1e-9) for c in changes], **single_style())
    axes[2].set_ylabel("Profile change per cycle [-]")
    for axis, level in ((axes[1], tolerance), (axes[2], profile_tolerance)):
        if level:
            axis.axhline(level, color=REFERENCE, linestyle=(0, (4, 3)), linewidth=0.8)
    for axis in axes:
        axis.set_xlabel("Cycle number [-]")
        axis.set_xlim(1, max(cycles[-1], 2))
    label_panels(axes)
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_cycle_pressure(model, results, output_path):
    """
    The pressure history of one cycle at the feed end and at the product end
    of the column, with the steps separated and named: the usual cycle diagram.
    """
    import matplotlib.pyplot as plt
    apply_style()
    t, states, spans = _cycle_series(model, results)
    pressure = np.array([model.unpack(s)["P"] for s in states.T]) * model.P0 / 1.0e5
    figure, axis = plt.subplots(figsize=(9.0, 3.6))
    axis.plot(t, pressure[:, 0], label="Feed end (Z = 0)", **series_style(0))
    axis.plot(t, pressure[:, -1], label="Product end (Z = 1)", **series_style(1))
    _mark_steps(axis, spans, label_axis=axis)
    axis.set_xlabel("Time within the cycle [s]")
    axis.set_ylabel("Pressure [bar]")
    axis.set_xlim(t[0], t[-1])
    axis.legend(loc="upper right", bbox_to_anchor=(1.0, 1.22), ncol=2)
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_end_flows(model, results, output_path):
    """
    Molar flow of every component through the feed end and the product end
    over one cycle, positive out of the column, with the steps separated.
    """
    import matplotlib.pyplot as plt
    apply_style()
    c = model.column
    figure, axes = plt.subplots(2, 1, figsize=(9.0, 5.8), sharex=True)
    offset, spans = 0.0, []
    for r in results:
        if r.time[-1] <= 0.0:
            continue
        for axis, end in zip(axes, ("feed", "product")):
            h = model.end_history(r, end)
            total = (-1.0 if end == "feed" else 1.0) * h["velocity"] * c.bed_voidage * c.cross_section \
                * h["pressure"] * 1.0e5 / (8.314 * h["temperature"])
            for k, name in enumerate(model.gas.components):
                axis.plot(offset + r.time, h["mole_fractions"][k] * total, **series_style(k),
                          label=formula(name) if offset == 0.0 else None)
        spans.append((offset, offset + r.time[-1], r.step.name))
        offset += r.time[-1]
    _mark_steps(axes, spans, label_axis=axes[0])
    axes[0].set_ylabel("Feed end, out [mol/s]")
    axes[1].set_ylabel("Product end, out [mol/s]")
    axes[1].set_xlabel("Time within the cycle [s]")
    axes[1].set_xlim(0.0, offset)
    for axis in axes:
        axis.axhline(0.0, color=REFERENCE, linewidth=0.7)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", ncol=len(labels), bbox_to_anchor=(0.98, 1.0))
    label_panels(axes)
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_end_compositions(model, results, output_path):
    """
    Composition of the gas leaving each end of the column over one cycle,
    shown only while gas flows out, with the steps separated: the outlet
    analyser traces of an experiment.
    """
    import matplotlib.pyplot as plt
    from .analysis import end_flows
    apply_style()
    figure, axes = plt.subplots(2, 1, figsize=(9.0, 5.8), sharex=True)
    offset, spans = 0.0, []
    for r in results:
        if r.time[-1] <= 0.0:
            continue
        for axis, end in zip(axes, ("feed", "product")):
            h = end_flows(model, r, end)
            leaving = h["flow"] > 1e-12 * max(np.max(np.abs(h["flow"])), 1e-30)
            for k, name in enumerate(model.gas.components):
                y = np.where(leaving, h["mole_fractions"][k], np.nan)
                axis.plot(offset + r.time, y, **series_style(k), label=formula(name) if offset == 0.0 else None)
        spans.append((offset, offset + r.time[-1], r.step.name))
        offset += r.time[-1]
    _mark_steps(axes, spans, label_axis=axes[0])
    axes[0].set_ylabel("Leaving the feed end, $y$ [-]")
    axes[1].set_ylabel("Leaving the product end, $y$ [-]")
    axes[1].set_xlabel("Time within the cycle [s]")
    axes[1].set_xlim(0.0, offset)
    for axis in axes:
        axis.set_ylim(-0.02, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", ncol=len(labels), bbox_to_anchor=(0.98, 1.0))
    label_panels(axes)
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_step_balance(model, results, output_path):
    """
    Moles of each component leaving (positive) or entering (negative) the
    column through each end in every step of one cycle, as horizontal
    lollipops: one row per step, one marker per component.
    """
    import matplotlib.pyplot as plt
    apply_style()
    steps = [r for r in results if r.time[-1] > 0.0]
    names = [r.step.name for r in steps]
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 0.6 * len(steps) + 1.6), sharey=True)
    for axis, end in zip(axes, ("feed", "product")):
        series = [[-r.moles_in[end][k] for r in steps] for k in range(len(model.gas.components))]
        _lollipops(axis, names, series, names=[formula(c) for c in model.gas.components])
        axis.set_title(f"{end.capitalize()} end (out > 0, in < 0)", fontsize=10)
        axis.set_xlabel("Out of the column per cycle [mol]")
    axes[1].legend(loc="best")
    label_panels(axes)
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_energy_breakdown(results, output_path):
    """Work of the compressors and vacuum pumps in each step of one cycle, kJ, as horizontal lollipops."""
    import matplotlib.pyplot as plt
    apply_style()
    names = [r.step.name for r in results if r.time[-1] > 0.0]
    work = [sum(r.work.values()) / 1000.0 for r in results if r.time[-1] > 0.0]
    total = sum(work) or 1.0
    figure, axis = plt.subplots(figsize=(7.5, 0.55 * len(names) + 1.3))
    _lollipops(axis, names, [work], value_format=lambda v: f"{v:.2f} kJ ({100.0 * v / total:.0f} %)")
    axis.set_xlim(0.0, 1.3 * max(max(work), 1e-12))
    axis.set_xlabel("Machine work per cycle [kJ]")
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_heatmaps(model, results, output_path, component=None):
    """
    Mole fraction, temperature and loading over position and time in one
    cycle, as viridis maps with labelled contours at 10, 50 and 90 % of the
    range of each map (where the fronts are).
    """
    import matplotlib.pyplot as plt
    apply_style()
    k = model.gas.index(component) if component else 0
    name = model.gas.components[k]
    t, states, spans = _cycle_series(model, results)
    unpacked = [model.unpack(s) for s in states.T]
    fields = [(np.array([model.full_composition(u["y"])[k] for u in unpacked]).T, f"{symbol('y', name)} [-]"),
              (np.array([u["T"] for u in unpacked]).T * model.T0, "Temperature [K]"),
              (np.array([u["x"][k] for u in unpacked]).T * model.qs0, f"{symbol('q', name)} [mol/kg]")]
    figure, axes = plt.subplots(3, 1, figsize=(9.0, 8.0), sharex=True)
    zc = (np.arange(model.n_cells) + 0.5) * model.cell_width
    for axis, (values, label) in zip(axes, fields):
        image = axis.pcolormesh(t, zc, values, cmap=SEQUENTIAL, shading="auto", rasterized=True)
        bar = figure.colorbar(image, ax=axis, label=label, pad=0.01)
        bar.outline.set_linewidth(0.6)
        low, high = np.nanmin(values), np.nanmax(values)
        if high - low > 1e-12 * max(abs(high), 1.0):
            levels = low + (high - low) * np.array([0.1, 0.5, 0.9])
            contours = axis.contour(t, zc, values, levels=levels, colors="white", linewidths=0.7)
            axis.clabel(contours, fontsize=7, fmt=lambda v: f"{v:.3g}")
        axis.set_ylabel("Z [-]")
        for side in ("top", "right"):
            axis.spines[side].set_visible(True)
        for start, _, _ in spans[1:]:
            axis.axvline(start, color="white", linewidth=0.6, linestyle=(0, (3, 3)), alpha=0.8)
    _name_steps(axes[0], spans, t[-1] - t[0])
    axes[-1].set_xlabel("Time within the cycle [s]")
    label_panels(axes)
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_step_profiles(model, result, output_path, times=6, component=None):
    """
    How the column changes during one step: pressure, temperature, mole
    fraction and loading along the column at ``times`` moments, for example
    the front moving through the bed in a feed step.
    """
    import matplotlib.pyplot as plt
    apply_style()
    k = model.gas.index(component) if component else 0
    name = model.gas.components[k]
    z = (np.arange(model.n_cells) + 0.5) * model.cell_width
    picks = np.unique(np.linspace(0, result.states.shape[1] - 1, times).round().astype(int))
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    for n, index in enumerate(picks):
        prof = _profiles(model, result.states[:, index])
        style = dict(series_style(n), label=f"{result.time[index]:.0f} s")
        axes[0, 0].plot(z, prof["P"], **style)
        axes[0, 1].plot(z, prof["T"], **style)
        axes[1, 0].plot(z, prof["y"][k], **style)
        axes[1, 1].plot(z, prof["q"][k], **style)
    for axis, label in zip(axes.ravel(), ["Pressure [bar]", "Temperature [K]", f"{symbol('y', name)} [-]",
                                          f"{symbol('q', name)} [mol/kg]"]):
        axis.set_xlabel("Z [-]")
        axis.set_ylabel(label)
        axis.set_xlim(0.0, 1.0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.04),
                  title=f"Step {result.step.name}")
    label_panels(axes.ravel())
    figure.tight_layout()
    save_figure(figure, output_path)


FIGURE_NAMES = {
    "profiles": "Profiles",
    "cycle_pressure": "Cycle pressure",
    "end_flows": "End flows",
    "end_compositions": "End compositions",
    "step_balance": "Step balance",
    "heatmaps": "Maps",
    "feed_step_profiles": "Feed step profiles",
    "last_cycle": "Last cycle",
    "last_3_cycles": "Last three cycles",
    "last_10_cycles": "Last ten cycles",
    "convergence": "Convergence",
    "energy": "Energy",
}
"""File names of the figures written by ``plot_all``, after the prefix."""


def plot_all(model, cycles, history, prefix, component=None, positions=(0.0, 0.5, 1.0),
             tolerance=None, profile_tolerance=None):
    """
    The full set of figures of a cyclic simulation, saved as PNG and PDF.

    The files are named by the prefix and the figure, separated by a space,
    for example "Results/Case one cycle pressure.png" for the prefix
    "Results/Case one". A prefix that ends with a path separator is a
    folder, and the files are then named by the figure alone.

    Args:
        model: The ColumnModel.
        cycles: Step results of the last cycles, oldest first (``CycleRun.recent``);
            the last one is taken as the cycle at steady state.
        history: KPIs of every cycle (``CycleRun.history``).
        prefix: Path prefix of the files.
        component: Component shown in the composition and loading figures.
        positions: Positions Z of the time histories.
        tolerance: Mass balance tolerance of the steady state, %, drawn on the convergence figure.
        profile_tolerance: Profile change tolerance of the steady state, drawn on the convergence figure.

    Returns:
        List of the PNG files written.
    """
    last = cycles[-1]
    files = []
    folder = str(prefix).endswith(("/", os.sep))

    def out(figure):
        name = FIGURE_NAMES[figure]
        files.append(f"{prefix}{name}.png" if folder else f"{prefix} {name[0].lower()}{name[1:]}.png")
        return files[-1]

    plot_profiles(model, last, out("profiles"), component)
    plot_cycle_pressure(model, last, out("cycle_pressure"))
    plot_end_flows(model, last, out("end_flows"))
    plot_end_compositions(model, last, out("end_compositions"))
    plot_step_balance(model, last, out("step_balance"))
    plot_heatmaps(model, last, out("heatmaps"), component)
    fed = [r for r in last if r.time[-1] > 0.0 and getattr(r.step.feed_end, "stream", None) == "feed"
           and getattr(r.step.product_end, "stream", None)]
    if fed:
        plot_step_profiles(model, max(fed, key=lambda r: r.time[-1]), out("feed_step_profiles"), component=component)
    plot_time_histories(model, cycles[-1:], out("last_cycle"), positions, component)
    if len(cycles) >= 3:
        plot_time_histories(model, cycles[-3:], out("last_3_cycles"), positions, component)
    if len(cycles) >= 10:
        plot_time_histories(model, cycles[-10:], out("last_10_cycles"), positions, component)
    if history:
        plot_convergence(history, out("convergence"), tolerance, profile_tolerance)
    if any(sum(r.work.values()) > 0.0 for r in last):
        plot_energy_breakdown(last, out("energy"))
    return files


def plot_breakthrough(model, result, output_path, positions=(0.25, 0.5, 0.75), levels=(0.05, 0.5)):
    """
    Breakthrough of a feed step: (a) outlet mole fraction of each component
    divided by its feed value, with the times at which the first component
    fed reaches ``levels`` of its feed value; (b) outlet temperature and the
    temperature inside the column at ``positions``, as thermocouples would
    read it; (c) outlet molar flow.
    """
    import matplotlib.pyplot as plt
    from .analysis import breakthrough_metrics, end_flows
    apply_style()
    out, feed = end_flows(model, result, "product"), end_flows(model, result, "feed")
    t = out["time"]
    y_feed = feed["mole_fractions"][:, -1]
    figure, axes = plt.subplots(3, 1, figsize=(7.5, 8.4), sharex=True)
    first = None
    for i, name in enumerate(model.gas.components):
        if y_feed[i] <= 0.0:
            continue
        axes[0].plot(t, out["mole_fractions"][i] / y_feed[i], label=formula(name), **series_style(i))
        first = first or name
    if first is not None:
        metrics = breakthrough_metrics(model, result, first, levels)
        for (level, time), style in zip(metrics.breakthrough_times.items(), (":", "--", "-.")):
            if np.isfinite(time):
                axes[0].axvline(time, color=REFERENCE, linestyle=style, linewidth=0.9,
                                label=f"{formula(first)} at {level:g} of feed: {time:.0f} s")
    axes[0].set_ylabel(r"$y_{\mathrm{out}} / y_{\mathrm{feed}}$ [-]")
    axes[0].legend(loc="best")
    axes[1].plot(t, out["temperature"], label="Outlet", **series_style(0))
    for j, z in enumerate(positions):
        cell = min(int(z * model.n_cells), model.n_cells - 1)
        temperature = np.array([model.unpack(s)["T"][cell] for s in result.states.T]) * model.T0
        axes[1].plot(t, temperature, label=f"Z = {z:g}", **series_style(j + 1))
    axes[1].set_ylabel("Temperature [K]")
    axes[1].legend(loc="best", ncol=2)
    axes[2].plot(t, out["flow"], **single_style())
    axes[2].set_ylabel("Outlet flow [mol/s]")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_xlim(t[0], t[-1])
    label_panels(axes)
    figure.align_ylabels(axes)
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_isotherms(adsorbent, components, temperatures, max_pressure, output_path, composition=None, points=80):
    """
    Equilibrium loadings: one panel per component with its pure-component
    isotherm at each temperature, and, when ``composition`` is given, a last
    panel with the competitive loadings of a gas of that composition against
    total pressure.

    Args:
        adsorbent: An Adsorbent (or anything with an ``isotherm``).
        components: Component names, in the order of the isotherm.
        temperatures: Temperatures, K.
        max_pressure: Highest (partial) pressure, bar.
        output_path: PNG path; a PDF is saved next to it.
        composition: Optional {component: mole fraction} of a mixture.
        points: Number of pressures on each curve.
    """
    import matplotlib.pyplot as plt
    apply_style()
    isotherm = getattr(adsorbent, "isotherm", adsorbent)
    n = len(components)
    panels = n + (1 if composition else 0)
    columns = min(panels, 3)
    rows = int(np.ceil(panels / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.5 * rows), squeeze=False)
    axes = axes.ravel()
    pressure = np.linspace(0.0, max_pressure, points)
    for i, name in enumerate(components):
        for m, T in enumerate(temperatures):
            c = np.zeros((n, points))
            c[i] = pressure * 1.0e5 / (8.314 * T)
            q = isotherm.loadings(c, np.full(points, float(T)))[i]
            axes[i].plot(pressure, q, label=f"{T:.0f} K", **series_style(m))
        axes[i].set_title(f"{formula(name)}, pure", fontsize=10)
        axes[i].set_xlabel("Pressure [bar]")
        axes[i].set_ylabel("Loading [mol/kg]")
        axes[i].set_xlim(0.0, max_pressure)
    axes[0].legend(loc="best")
    if composition:
        axis = axes[n]
        y = np.array([composition.get(name, 0.0) for name in components], dtype=float)
        T = float(temperatures[0])
        c = np.outer(y, pressure * 1.0e5 / (8.314 * T))
        q = isotherm.loadings(c, np.full(points, T))
        for i, name in enumerate(components):
            if y[i] > 0.0:
                axis.plot(pressure, q[i], label=formula(name), **series_style(i))
        axis.set_title("Mixture, " + ", ".join(f"{v:g} {formula(k)}" for k, v in composition.items() if v)
                       + f", {T:.0f} K", fontsize=9)
        axis.set_xlabel("Total pressure [bar]")
        axis.set_ylabel("Loading [mol/kg]")
        axis.set_xlim(0.0, max_pressure)
        axis.legend(loc="best")
    for axis in axes[panels:]:
        axis.set_visible(False)
    label_panels(axes[:panels])
    figure.tight_layout()
    save_figure(figure, output_path)
