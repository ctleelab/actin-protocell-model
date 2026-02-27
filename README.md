# actin-bracing-model

## Light-guided actin polymerization drives directed motility in protocells

This repository provides tools for:
- simulating **actin-induced membrane deformation** using **Cytosim with a deformable membrane in 2-D**.
- analysis and plotting codes used to generate figues in the manuscript - [Light guided actin polymerization drives motitliy in protocells](https://doi.org/10.1101/2024.10.14.617543)

## Repository Structure
```
actin-bracing-model/
│── Figures/                        # Contains general inputs and outputs from simulations
│   ├── inputfiles/                 # Cytosim config files used to simulate each condition
│   ├── outputfiles/                # Final results used to make figures
│
│──Cytosim/                         # Modified source code used to simulate actin filaments with deformable membrane        
│   ├── LICENSE                     # GNU GPL v3 License file 
│   ├── source code files                 
│  
│── analysis-codes/                 # Contains scripts used to generate membrane deformation and figures from simulations
│   ├── LICENSE                     # CC BY 4.0 License file
│   ├── scripts
│ 
│── README.md                       # Project documentation
│── ./gitignore                     # Files to ignore
```

## Installation

### 1 Clone the Repository
```sh
git clone https://github.com/ctleelab/actin-bracing-model.git
cd actin-bracing-model
```

### 2 Install Cytosim
```sh
cd Cytosim
make clean
make dim2
```
For further assistance with installation, please refer to [text](https://gitlab.com/f-nedelec/cytosim/-/blob/master/README.md\)

## Running analysis codes using pixi
Simulations will generate `objects.cmo` files containing the coordinates of all filaments, actin binding proteins, and the membrane.

[Pixi](https://pixi.sh/) is a package management tool that helps to harmonize conda and pypi dependencies.
It also provides benefits such as preconfigured tasks and pipelining.

Instructions for installing pixi can be found here: https://pixi.sh/latest/installation/

Once pixi is installed you can 
```sh 
cd analysis-codes
``` 
and simply run:
1. `pixi run python extract-space.py --save_dir \path\to\folder\containing\all\sim\output` to extract the membrane coordinates from each replicate folder.
2. `pixi run python membrane_extension_calculation.py save_dir \path\to\folder\containing\all\sim\output` to calculate the membrane deformation for each replicate.
3. `pixi run makeplot` to plot the result. We import tools from another repository [ctleelab-mpl-utilites](https://github.com/ctleelab/ctleelab-mpl-utilities) for figure generation. This is automatically installed with `Pixi`.

The tasks, dependency specificiations, and other configurations can be found in `pixi.toml`. 
Notably, changes to this file can be done by hand or by the pixi cli tool.
The `pixi.lock` file is a human readable list of solved dependency versions.
Changes to the lockfile are handled by pixi and reflect changes to the environment.

## License
The analysis code is licensed under the **CC BY 4.0 International**. Cytosim is licensed under **GNU GPL v3**.

## Contact
If you have questions, feel free to reach out via:
- **Email:** hakenuwa@ucsd.edu or ctlee@ucsd.edu
