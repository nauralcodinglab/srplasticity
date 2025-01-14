Copyright (C) 2021 Julian Rossbroich, Daniel Trotter, John Beninger, Richard Naud

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Spike Response Plasticity

Implementation of a linear-nonlinear model of short-term plasticity as described in
[Linear-Nonlinear Cascades Capture Synaptic Dynamics](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008013).

## Tutorial
Users can access a detailed introductory tutorial for this package hosted on Google Colab through at
https://colab.research.google.com/drive/1Cp3lsnCCjTl_vRgXjeHzkb39Sx4m21uS?usp=sharing  

## Installation
The version of srplasticity package corresponding to the paper above can be installed directly using the following command: 
pip install srplasticity==0.0.1

All source files and additional files are hosted on Github at https://github.com/nauralcodinglab/srplasticity.git

## Repository Structure

- The **srplasticity** package contains all source code and implements
the SRP model and the associated parameter inference.
- The **examples** directory contains use examples and tutorials (currently in progress)
- The **scripts** directory contains the scripts to reproduce the figures in the manuscript
- The **data** directory contains the data used in the manuscript figures and the use examples.

## Paper Citation
@article{rossbroich2021linear,
  title={Linear-nonlinear cascades capture synaptic dynamics},
  author={Rossbroich, Julian and Trotter, Daniel and Beninger, John and T{\'o}th, Katalin and Naud, Richard},
  journal={PLoS computational biology},
  volume={17},
  number={3},
  pages={e1008013},
  year={2021},
  publisher={Public Library of Science San Francisco, CA USA}
}

