"""Write the ready-made cases as specification files in examples/specs (run from the repository root)."""
from pathlib import Path

import sys

sys.path.insert(0, "examples")

import setups as presets  # noqa: E402
from setups import cycles  # noqa: E402
from swingpy.spec import save_problem  # noqa: E402

OUT = Path("examples/specs")
OUT.mkdir(parents=True, exist_ok=True)
HAGHPANAH_AICHE = ("Parameters of Haghpanah et al., AIChE J. 59 (2013) 4735, https://doi.org/10.1002/aic.14192\n"
                   "(column, adsorbent and gas of Haghpanah et al., Ind. Eng. Chem. Res. 52 (2013) 4249).")
N2 = dict(pressure=1.0, temperature=298.15, composition={"N2": 1.0})

for name in cycles.EXAMPLE_POINTS:
    model, cycle, _ = cycles.example_case(name)
    title = name.replace(",", "").replace("%", "percent")
    save_problem(OUT / f"Cycle synthesis {title}.yaml", model, cycle, N2,
                 header=f"Flue gas (15 % CO2) on zeolite 13X: {name}.\n" + HAGHPANAH_AICHE)
for name in presets.SMR_OPERATING_POINTS:
    model, cycle, _ = presets.smr_example_case(name)
    title = name.replace("-", "").replace("Zeolite", "zeolite")
    save_problem(OUT / f"SMR flue gas VSA {title}.yaml", model, cycle,
                 dict(pressure=presets.SMR_HIGH_PRESSURE, temperature=298.15, composition=presets.SMR_FEED),
                 header=f"Steam-methane-reformer flue gas (20 % CO2) on {name}: four-step cycle with light product\n"
                        "pressurization at its optimal operating point. Parameters of Subraveti et al., Sep. Purif.\n"
                        "Technol. 256 (2021) 117832, https://doi.org/10.1016/j.seppur.2020.117832")
print(sorted(p.name for p in OUT.glob("*.yaml")))
