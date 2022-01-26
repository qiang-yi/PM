# Multi-Scale Point Pattern Analysis in Pyramid Model

This repository includes programs for the multi-scale point patter analysis in the Pyramid Model (PM). The configuration of the PM and analytical methods are presented in the manuscript "_Analyzing Multi-Scale Spatial Patterns in a Pyramid Modeling Framework_" under review by the journal of [Cartography and Geographic Information Science](https://www.tandfonline.com/journals/tcag20).

## File description
This repository includes both Python and Matlab code, which are stored in two separate folders. The Python code (in Jupyter Notebook) is used to create clustering and random point patterns for the experiment. The Matlab code is stored in Live Code File Format (.mlx), which creates 3D visualization based on the PM.

#### Python_code
- **Python_code/Kernel_Density_PM.ipynb**: Create random and clustered point patterns, and then generate <u>kernel density</u> maps with different bandwidths. The created kernel density maps are stored in 3D matrices, which are exported to matfiles (.mat).

- **Python_code/Quadrat_Density_PM.ipynb**: Create random and clustered point patterns, and then generate <u>quadrat density</u> maps with different quadrat sizes. The created quadrat density maps are stored in 3D matrices, which are exported to matfiles (.mat).

- **Python_code/kde_pyramid.py**: A function (kde_pyramid) that creates a 3D matrix that store kernel density maps with multiple bandwidths. The kde_pyramid function is called in Kernel_Density_PM.ipynb.

- **Python_code/PointProcess.py**: A class (PointProcess) modified from PySal.PointProcess to display parents and children points in different colors. The class is called in Kernel_Density_PM.ipynb and Quadrat_Density_PM.ipynb to create point patterns.

#### Matlab_code
- **Matlab_code/PM_KD.mlx**: Creates interactive 3D visualization for multi-scale kernel density of both <u>clustered</u> and <u>random</u> point sets. The program loads data generated from Kernel_Density_PM.ipynb.

- **Matlab_code/PM_quadrat_cls.mlx**: Creates interactive 3D visualization for multi-scale quadrat density of the <u>clustered</u> point set. The program loads data generated from Quadrat_Density_PM.ipynb.

- **Matlab_code/PM_quadrat_ran.mlx**: Creates interactive 3D visualization based on PM for multi-scale quadrat density of the <u>random</u> point set. The program loads data generated from Quadrat_Density_PM.ipynb.
