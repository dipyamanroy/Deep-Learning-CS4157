# README.md

### Assignments
#### Deep Learning `CS4157`
---
Codes from assignments of Deep Learning `CS4157`. So far, only the first assignment has been solved, and here is the [README.md](assignment-1/README.md).

### Table of Contents

1. [Prerequisites](#prerequisites)
2. [LaTeX font rendering](#latex-font-rendering)

### Prerequisites
1. Python 3.x installed on your system.
2. The following Python packages must be installed: `pandas`, `numpy`, `matplotlib`, `json` and `math`. (More to follow)

> [!TIP]
> The suggested way to install these packages is using `pip` and in a virtual environment. For more information refer to [Install packages in a virtual environment using pip and venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

3. To generate plots using `matplotlib`, this project ues LaTeX fonts. This requires a working LaTeX distribution on your system. Installation instructions are given in the final section.   


### LaTeX font rendering

To use LaTeX fonts in matplotlib, you need to have a LaTeX distribution installed on your system. Here are the installation instructions for Windows, Mac, and Linux:

#### Windows:

1. Install a LaTeX distribution: 
    - Download and install MiKTeX from https://miktex.org/download. 
    - Alternatively, you can install TeX Live from https://www.tug.org/texlive/.
2. Install a font package that includes the Computer Modern Roman font:
    - For MiKTeX, open the MiKTeX Console and install the cm-super package.
    - For TeX Live, run the command tlmgr install cm-super in the command prompt.
3. Restart your Python environment or IDE.

#### Mac (with Homebrew):

1.  Install a LaTeX distribution:
    - Run the command `brew install --cask mactex` in the terminal.
2. Install a font package that includes the Computer Modern Roman font:
    - Run the command `tlmgr install cm-super` in the terminal.
3. Restart your Python environment or IDE.

#### Linux (Ubuntu-based distributions):
1. Install a LaTeX distribution:
    - Run the command `sudo apt-get install texlive-full` in the terminal.
2. Install a font package that includes the Computer Modern Roman font:
    - Run the command `sudo apt-get install cm-super ` in the terminal.
3. Restart your Python environment or IDE.

Additional notes:
Make sure you have the `dvipng` package installed, which is required for matplotlib to render LaTeX fonts. You can install it using your package manager (e.g., `sudo apt-get install dvipng` on Ubuntu-based distributions).
