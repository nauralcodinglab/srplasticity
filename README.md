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
[Linear-Nonlinear Cascades Capture Synaptic Dynamics](https://www.biorxiv.org/content/early/2021/01/27/2020.06.04.133892).

## Repository Structure

- The **srplasticity** package contains all source code and implements
the SRP model and the associated parameter inference.
- The **examples** directory contains use examples and tutorials (currently in progress)
- The **scripts** directory contains the scripts to reproduce the figures in the manuscript
- The **data** directory contains the data used in the manuscript figures and the use examples.

## Preprint Citation

<pre><code>@article {Rossbroich2020.06.04.133892,
	author = {Rossbroich, Julian and Trotter, Daniel and Beninger, John and T{\'o}th, Katalin and Naud, Richard},
	title = {Linear-Nonlinear Cascades Capture Synaptic Dynamics},
	year = {2021},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2021/01/27/2020.06.04.133892},
	eprint = {https://www.biorxiv.org/content/early/2021/01/27/2020.06.04.133892.full.pdf},
	journal = {bioRxiv}
}
</code></pre>
