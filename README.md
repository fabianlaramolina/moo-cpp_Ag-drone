# moo-cpp_Ag-drone
# A Multi-Objective Optimization Approach to Coverage Path Planning of Agricultural Drone

**Manuscript ID:** IEEE LATAM Submission ID: 10298
**Authors:**  
- Fabian Andres Lara-Molina
- Fran Sérgio Lobato
- Maicon F. Appelt
## 📁 Included Scripts
This repository contains all scripts required to reproduce the simulation and numerical results presented in the article.
To reproduce the results, run the main_moo_cpp.py script. Upon execution, an interactive menu will appear in the terminal, allowing you to select the geometry for the coverage path planning analysis:

    Polygon 1 (Nonagon): Executes the optimization for a complex 9-sided polygon.
    Polygon 2 (Pentagon): Executes the optimization for a 5-sided polygon.
    Polygon 3 (Rectangle): Executes the optimization for a standard rectangular area.
    Polygon 4 (Circle): Executes the optimization for a circular geometry.
    Map - Case Study: Runs the real-world simulation using the georeferenced data from the area_UFTM.shp file.

Simply enter the corresponding number (1-5) to start the simulation for the desired scenario.
---

## 📂 Required Files

- `area_UFTM.shp`: Required for `main_moo_cpp.py`. Place it in the same folder as the script.
- The files: area_UFTM.dbf, area_UFTM.dbf, area_UFTM.shx and area_UFTM.shx can be used to edit the georeferenced area of Fig.6(a) in Qgis (https://qgis.org/). 

## 💻 Requirements

- Python 3.8 or later.
- The following libraries are required: numpy, matplotlib, shapely, pymoo, geopandas and contextily.

---

## ✉️ Contact

For questions or replication of results:  
fabian.molina@uftm.edu.br
