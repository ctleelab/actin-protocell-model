The repository contains CytoSim input files used to simulate actin filament-induced membrane deformation.
Results can be found here https://doi.org/10.1101/2023.05.26.542517

CytoSim can be downloaded directly from https://gitlab.com/f-nedelec/cytosim/-/blob/master/README.md\

Input files can be found inside folders named based on Figure numbers in which they appear in the manuscript.

FIG_NUMBER/INPUTFILES - carries inputfiles

Within the INPUTFILES folder, sub-folders are organized as follows. Sub-folders are organized by the simulated NPF condition (pVCA, mDia1, pVCA+mDia1). Within each condition folder, we futher subdivide into simulations with and without interfilament STERICS. Runs per condition are found here and are labelled run_REPLICATEID where REPLICATEID spans 000 to 019.

Analysis codes for extracting the coordinates of the 2-D membrane and calculating the membrane extension are included in the folder named analysis-code.