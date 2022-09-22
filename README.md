# MR-BMD
Sample data and code for the article V. Di Trapani, L. Brombal, and F. Brun, [Multi-material spectral photon-counting micro-CT with minimum residual decomposition and self-supervised deep denoising](https://doi.org/10.1364/OE.471439), Optics Express, 2022

Data and code will be available as soon as the article is published.

## Input

Code is designed to consider as input a number (e.g. 8) of images having different energetic content:

![](/doc/Figure1.jpg)

and a decomposition matrix resulting from the physics of input spectrum, detector energy response and material X-ray absorption:


Note: different number of energy bins and materials can be considered.

## Output

Known the pixel size, the code outputs the concentration maps of each material in mg/ml, such as e.g.:


Note: in this output example five materials were considered, i.e. soft tissue (water), bone, I, Ba, and Gd. 
