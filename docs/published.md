# Comparison with published work

SwingPy is compared with the simulations and experiments of the studies below. Every comparison uses only the data each article reports, nothing is fitted, and the result is stated plainly, including where the agreement is partial or absent. The author is grateful to the authors of these studies for publishing their models and data in enough detail to be reproduced.

![Purity and recovery predicted by SwingPy against the published values](figures/Published%20comparison.png)

*Purity and recovery predicted by SwingPy against the values reported for VSA cycles of CO<sub>2</sub> capture (the article's model where it has one, the pilot plant or laboratory experiment otherwise). The grey band is ±3 points. The points outside it are the cases listed below as partial or not reproduced.*

| Study | What is compared | Outcome |
|---|---|---|
| R. Haghpanah, A. Majumder, R. Nilam, A. Rajendran, S. Farooq, I. A. Karimi, M. Amanullah, *Ind. Eng. Chem. Res.* 52 (2013) 4249, [doi:10.1021/ie302658y](https://doi.org/10.1021/ie302658y) | breakthrough of CO<sub>2</sub> on zeolite 13X, balance errors, accuracy of the discretization schemes, four-step VSA | breakthrough reproduced closely (CO<sub>2</sub> front at 535 s against about 520 s, balances closed to 10<sup>−4</sup> %); cycle trends reproduced, KPIs within about 10 % |
| R. Haghpanah, R. Nilam, A. Rajendran, S. Farooq, I. A. Karimi, *AIChE J.* 59 (2013) 4735, [doi:10.1002/aic.14192](https://doi.org/10.1002/aic.14192) | basic, light product pressurization and pressure equalization cycles | purity and recovery within 2 to 3 points at five operating points; the six-step cycle with light and heavy reflux is not reproduced |
| S. G. Subraveti, S. Roussanaly, R. Anantharaman, L. Riboldi, A. Rajendran, *Sep. Purif. Technol.* 256 (2021) 117832, [doi:10.1016/j.seppur.2020.117832](https://doi.org/10.1016/j.seppur.2020.117832) | three adsorbents: cycle, plant, column schedule and capture cost | purity and recovery within 1.5 points; capture cost of the published designs within 1.5 % (the plant, schedule and cost modules are not part of this release) |
| S. G. Subraveti, PhD thesis, University of Alberta (2021), Chapter 4 | isothermal three- and four-step cycles on IISERP MOF2 | partly: recovery within 1.7 points, purity 1.8 to 2.6 points lower |
| S. Krishnamurthy, V. R. Rao, S. Guntuka, P. Sharratt, R. Haghpanah, A. Rajendran, M. Amanullah, I. A. Karimi, S. Farooq, *AIChE J.* 60 (2014) 1830, [doi:10.1002/aic.14435](https://doi.org/10.1002/aic.14435) | pilot plant VSA on zeolite 13X | five runs within 1.2 points of purity and 0.7 of recovery; one partly; the two blowdown pressure runs not reproduced |
| T. T. T. Nguyen, G. K. H. Shimizu, A. Rajendran, *Chem. Eng. J.* 452 (2023) 139550, [doi:10.1016/j.cej.2022.139550](https://doi.org/10.1016/j.cej.2022.139550) | VSA experiments on structured CALF-20 | experiments II to V reproduced; VI to X partly (trends right, purity high) |
| S. G. Subraveti, K. Karimi, M. Gazzani, *Ind. Eng. Chem. Res.* 65 (2026) 9656, [doi:10.1021/acs.iecr.6c00407](https://doi.org/10.1021/acs.iecr.6c00407); S. G. Subraveti, K. Karimi, M. Gazzani, R. Anantharaman, *Syst. Control Trans.* 3 (2024) 215 | vector acceleration of the approach to cyclic steady state | steady states reproduced; numbers of cycles reproduced for the VSA test case |

The model is also checked against exact solutions: the uptake of a single particle (J. Crank, *The Mathematics of Diffusion*, 1975), within 0.5 %, and the breakthrough of a linear column with diffusion in the particles (D. M. Ruthven, *Principles of Adsorption and Adsorption Processes*, 1984), within 0.16 %.
