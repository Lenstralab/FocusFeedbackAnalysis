# Code for analysing experiments done using [FocusFeedbackGUI](https://github.com/Lenstralab/FocusFeedbackGUI)

## Installation
Make sure [python](https://www.python.org/) (version 3.10 or newer) is installed.

``` pip install git+https://github.com/Lenstralab/FocusFeedbackAnalysis.git ```

### Install FocusFeedbackGUI dependency
#### Option 1: from wheel
- Download a wheel from https://github.com/Lenstralab/FocusFeedbackGUI/releases, from the latest version, choose a wheel
fitting your pc.
- ``` pip install focusfeedbackgui*.whl ```

#### Option 2: from source
- Install [Rust](https://rustup.rs/).
- ``` pip install git+https://github.com/Lenstralab/FocusFeedbackGUI.git ```

### Install jupyter
#### Option 1: Jupyter Notebook
``` pip install notebook ```

#### Option 2: Jupyterlab
``` pip install jupyterlab ```

### Experiment preparation
Check out the [preparation notebook](notebooks/preparation.ipynb).

### Experiment analysis
Check out the [analysis notebook](notebooks/analysis.ipynb).