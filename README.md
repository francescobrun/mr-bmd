# MR-BMD (Minimum-Residual Basis Material Decomposition)
Sample data and Python code for the article: V. Di Trapani, L. Brombal, and F. Brun, [Multi-material spectral photon-counting micro-CT with minimum residual decomposition and self-supervised deep denoising](https://doi.org/10.1364/OE.471439), *Optics Express*, Vol. 30, Issue 24, pp. 42995-43011 (2022).

**Note**: MR-BMD might require a few minutes on a modern multi-core machine for 512x512 images. An implementation of conventional multi-material BMD is also included in the repository for comparison.

## Input

Code requires as input a number (e.g. 8) of X-ray CT (Computed Tomography) images at different energies:

![](/doc/Figure1.jpg)

and a decomposition matrix resulting from the physics of the CT settings and the desired materials (e.g. 4):

![](/doc/Figure2.jpg)

**Note**: different number of energy bins and materials can be considered.

## Output

Known the pixel size, the code outputs the concentration map of each material in mg/ml, such as e.g.:

![](/doc/Figure3.jpg)

**Note**: in this example five materials were considered, i.e. soft tissue (water), bone, I, Ba, and Gd. 

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE.txt) file for details.
