# liama-toolkit

A modular toolkit for spectral analysis, materials characterization, and
applied methods in cultural heritage research. Developed at LIAMA
(Laboratorio de Investigaciones Aplicadas a Materiales en Arte y Arqueología),
UMYMFOR-CONICET.

The `spectra_viewer` module offers utilities to plot or save spectra. When
smoothing is enabled these functions apply a Savitzky–Golay filter to obtain
the second derivative of the data.

## Quick Setup

These steps assume no prior experience with Git or Python packages.

1. **Install Python**
   - Download and install Python 3 from [python.org](https://www.python.org/downloads/).
   - During installation, enable the option “Add Python to PATH”.

2. **Get the toolkit**
   - Visit the repository page in your web browser.
   - Click the green **Code** button and choose **Download ZIP**.
   - Extract the ZIP file to a convenient folder (e.g. `C:\liama-toolkit` or
     `/home/user/liama-toolkit`).

3. **Open a terminal**
   - On Windows, open *Command Prompt*; on macOS/Linux, open *Terminal*.
   - Navigate to the extracted folder, for example:

     ```
     cd path/to/liama-toolkit
     ```

4. **Install required libraries**

     ```
     python -m pip install numpy scipy matplotlib
     ```

5. **(Optional) Run the tests**

     ```
     python -m pip install pytest
     python -m pytest
     ```

6. **Use the spectra viewer**

     ```
     python
     >>> import numpy as np
     >>> from spectra_viewer import plot_spectrum
     >>> x = np.linspace(0, 10, 50)
     >>> y = np.sin(x)
     >>> plot_spectrum(x, y, apply_smoothing=True)
     ```

This example plots the input spectrum alongside its second derivative
computed with a Savitzky–Golay filter.

