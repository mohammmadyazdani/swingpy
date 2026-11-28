"""
Draw the figure gallery of the documentation (docs/figures) from one example:
a breakthrough and a four-step VSA cycle run to cyclic steady state, both on
zeolite 13X (examples/specs/VSA zeolite 13X flue gas.yaml). Run from the repository root.
"""
import shutil
import tempfile
from pathlib import Path

from PIL import Image

from swingpy import Step
from swingpy.report import report_breakthrough, report_run
from swingpy.spec import load_problem

OUT = Path("docs/figures")
"""Pixels: legible in the documentation, small files."""
WIDTH = 1400

problem = load_problem("examples/specs/VSA zeolite 13X flue gas.yaml")
work = Path(tempfile.mkdtemp())

adsorption = problem.cycle.steps[1]
model = problem.model
start = model.initial_state(1.0, {"N2": 1.0}, 298.15)
breakthrough = model.simulate(Step("FEED", 4500.0, feed_end=adsorption.feed_end,
                                   product_end=adsorption.product_end), start, n_output=301)
report_breakthrough(model, breakthrough, work / "Breakthrough")

"""Plain cycles: acceleration does not help this case, whose approach to steady state is a slow drift."""
run = problem.run(max_cycles=1500, verbose=False, keep_cycles=3)
print(f"cycle run: {len(run.history)} cycles, converged {run.converged}, purity {run.performance.purity:.2f} %, "
      f"recovery {run.performance.recovery:.2f} %")
report_run(model, run, work / "Cycle", tolerance=0.5, profile_tolerance=1e-3)

OUT.mkdir(parents=True, exist_ok=True)
"""The comparison with published studies is drawn elsewhere and is kept."""
for old in list(OUT.glob("*.png")) + list(OUT.glob("*.txt")):
    if old.name != "Published comparison.png":
        old.unlink()
sources = [work / "Breakthrough" / name for name in ("Breakthrough.png", "Front profiles.png", "Isotherms.png")]
sources += [p for p in sorted((work / "Cycle").glob("*.png")) if p.name != "Isotherms.png"]
for source in sources:
    image = Image.open(source).convert("RGB")
    if image.width > WIDTH:
        image = image.resize((WIDTH, round(image.height * WIDTH / image.width)), Image.LANCZOS)
    image.save(OUT / source.name, optimize=True)
shutil.rmtree(work)
state = (f"at cyclic steady state, reached after {len(run.history)} cycles" if run.converged
         else f"after {len(run.history)} cycles, NOT at cyclic steady state")
(OUT / "Gallery.txt").write_text(f"{state}; purity {run.performance.purity:.2f} %, "
                                 f"recovery {run.performance.recovery:.2f} %\n")
print(sorted(p.name for p in OUT.glob("*.png")))
