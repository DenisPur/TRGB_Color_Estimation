# TRGB - Color Estimation

This small program was written during the processing of the EDD catalog to measure the color index of the red giant branch. Now I'm making it freely available. 

This Python program allows you to clean the photometry file, crop the field of the instrument (for example, by selecting the outer regions of the galaxy in field), build a color-magnitude diagram and measure the color index at any level ($M_I$) in two ways: by searching for the maximum density or approximating the branch with a parabola. For convenience, a graphical interface written in QT5 is used. 


When using (including modification), I humbly ask you to quote the work "TBA".

# Install
To create a working environment in Conda, run:
```
conda env create -f environment.yml
```
This will create an environment called `trgb_gui`, which will contain all the necessary Python packages. Alternatively, you can install them manually: 
```
- python > 3.9
- pyqt
- fpdf2
- matplotlib
- numpy
- pandas
- pillow
- scipy
- seaborn
- jupyterlab
```

Note: its `fpdf2`, not `fpdf`!

### Run
If you installed the environment using `environment.yml` activate it and run the program: 
```
conda activate trgb_gui
python run.py
```

# Usage
### File selection and basic data entry
Run the program. This is what you should see:

<img src="exhibition_materials/02_base_mouse.png" width="500"/>

1. Select file `.csv` file with photometric data. Each row should represent a possible star. Mandatory columns: `x`, `y` (coordinates in the instrument's field of view), `mag_v`, `err_v`, `mag_i`, `err_i` (apparent magnitude and measurement error in filters I and V, respectively). 

    Columns `type`, `snr_x` (i.e. `snr_v` or `snr_i`), `sharp_x`, `round_x`, `crowd_x`, `flag_x`, if present, can be used to clean data in the next step.

    <img src="exhibition_materials/03_file_selections.png" width="800"/>

2. Clean the photometry data. Select criteria to use, change them if necessary. In this example, I got rid of the bottom of the CMD by raising the Signal/Noise threshold.

    <img src="exhibition_materials/05_clearing.png" width="800"/>

3. Crop the field of view if necessary. In this example, I have cut off the areas of the instrument's field most dense with stars, and thus selected only the outer regions of the galaxy. You can also select a rectangular area in the field by entering the coordinates manually.

    <img src="exhibition_materials/06_clipping.png" width="5800"/>

    <img src="exhibition_materials/07_clipped.png" width="800"/>

4. Enter distance (in MPc or in Mag).

    <img src="exhibition_materials/08_distance.png" width="500"/>

5. Enter extinction (V-I) and absorbtion in filter I.

    <img src="exhibition_materials/09_color_excess.png" width="500">

6. View the cleaned instrument field and color-magnitude diagram in absolute magnitudes. There will be density histogram over the scatterplot, you can change it to kernel-density plot (kde) if you need. Using kde usually takes some time.

    <img src="exhibition_materials/10_abs_cmd.png" width="800">
    
    <img src="exhibition_materials/11_abs_cmd_isodense.png" width="800">

The program provides the ability to measure color index using two methods described in the article (ref. TBA).

### Branch approximation
This method allows one to approximate the branch of red giants using a parabola. A peculiarity of this method is that one must manually specify the boundaries of the region in color index (V-I) - brightness (I) coordinates where the branch is located.

<img src="exhibition_materials/branch_1.png" width="500">

<img src="exhibition_materials/branch_2.png" width="800">

### Density analythis
This method allows one to measure the color index using the density profile of the stars at the specified M_I level. The confidence interval estimate will be obtained using the Monte Carlo method.

<img src="exhibition_materials/density_1.png" width="5800">

### Saving results
The processing result will be two files: a json file with parameters and numerical estimates, and a pdf file with a visualization of intermediate graphs.

<img src="exhibition_materials/saving_1+2.png" width="800">

<img src="exhibition_materials/result_pdf.png" width="800">

<img src="exhibition_materials/result_json.png" width="800">
