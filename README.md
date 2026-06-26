# py_tps2tnt: A Python GUI for Geometric Morphometric Data Conversion for Cladistics analysis

<div align="left">
  <img width="120" height="120" src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhbdwXB_EFg_UQ_wi24dN3EJ1MgsTapyelahD4VojYxY1EM9oOUa3Ryhh52_oK4gzG-koGDw75kIcgjuI8F5Y-fRC8auuLpTrTtg_6zImfoTZk_ZShDlOilkH8nLutZoF-1cqsP3A3G7dTlnCROGFA1Ds07fLYDnLjvjkAIRldPRE7IiI7rmbOr3v3dNaL6/w113-h113/Icon%20py_tm2tnt.png" alt="py_tps2tnt Logo">

  **Developed by: Jonathan Liria & Ana Soto-Vivas**

  [![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Journal](https://img.shields.io/badge/Journal-RPB%20(2025)-green)](https://doi.org/10.15381/rpb.v32i2.30018)
</div>

---

## Overview

`py_tps2tnt` is a Python-based graphical user interface (GUI) application designed for evolutionary biologists and systematists who need to convert geometric morphometric data from landmark configurations into native format for **TNT (Tree Analysis using New Technologies)** to perform parsimony-based cladistic analyses. 

The software streamlines dataset preparation by parsing standard `.tps` files containing 2D coordinate landmark datasets. It automates landmark parsing, manages multi-specimen averaging routines per terminal species, and processes measurement scales following the strict operational criteria and guidelines of *Catalano & Goloboff (2018)*. This eliminates the tedious and error-prone process of manual coordinate string formatting for TNT data blocks.

<div align="center">
  <img width="530" height="422" alt="image" src="https://github.com/user-attachments/assets/0f1f61b6-a629-42a6-84c5-142ff93eb276" style="border-radius: 6px; border: 1px solid #ddd; margin-top: 15px;">

  <p><em>Figure 1. Main graphical interface of py_tps2tnt for TPS file uploading and landmark parameter configurations.</em></p>
</div>

---

## Video Tutorial

For a comprehensive walkthrough on landmark data loading, specimen configuration, and practical export workflows, a video tutorial is available (optimized for Spanish-speaking users):

📺 **[Watch the py_tps2tnt Video Tutorial on YouTube](https://www.youtube.com/watch?v=ubR3w-yRhx4&t)**

---

## System Requirements & Prerequisites

Before launching `py_tps2tnt`, please ensure your local machine environment has the following components installed:

### System Environment
* **Python 3.x**

### Required Python Libraries
The application relies on standard and scientific computation frameworks:
```text
pandas         # Data matrix manipulation and alignment
tkinter        # Graphical user interface rendering window
numpy          # Multi-dimensional numerical array calculation for coordinates
math           # Geometrical transformations and scale processing
csv            # Standard parsing utilities
os             # Local system file directory path handling
re             # Regex-driven token parsing for TPS landmarks
```

---

### Repository structure

```text
py_tps2tnt/
│
├── CITATION.cff          # Machine-readable citation metadata file
├── LICENSE               # Full distribution text (MIT Open-Source terms)
├── README.md             # Repository documentation and landing guide
├── data_examples.rar     # Example data compressed file
├── py_tps2tnt Manual.pdf # Comprehensive user operations manual
└── py_tps2tnt_v4.2.py    # Main Python graphical application source code
```
---

### How to Cite

If this application saves you manual editing hours and assists you in compiling dataset blocks for your research projects, please cite our peer-reviewed work:   

Liria, J., & Soto-Vivas, A. 2025. «py_tps2tnt y py_tm2tnt: Dos programas en Python para procesamiento de datos morfométricos en análisis cladísticos con TNT». Revista Peruana de Biología, 32 (2): e30018. https://doi.org/10.15381/rpb.v32i2.30018   

---

### License

This project is licensed under the open-source MIT License - see the local repository LICENSE file for details.

