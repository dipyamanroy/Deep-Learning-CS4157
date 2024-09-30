# Assignment 1
## Deep Learning `CS4157`

Codes from the first assignment of Deep Learning `CS4157`, involving creating a neural network from scratch, to predict the Combined Cycle Power Plant Dataset's net hourly electrical energy output (EP) value.

### Table of Contents

1. [Prerequisites](#prerequisites)
2. [Executing the Sine Predictor](#executing-the-sine-predictor)
3. [Executing the CCPP Predictor](#executing-the-ccpp-predictor)
4. [LaTeX font rendering](#latex-font-rendering)

### Prerequisites
1. Python 3.x installed on your system.
2. The following Python packages must be installed: `pandas`, `numpy`, `matplotlib`, `json` and `math`. 

> [!TIP]
> The suggested way to install these packages is using `pip` and in a virtual environment. For more information refer to [Install packages in a virtual environment using pip and venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

3. To generate plots using `matplotlib`, this project ues LaTeX fonts. This requires a working LaTeX distribution on your system. Installation instructions are given in the final section.   


### Executing the Sine Predictor
1. Navigate to the `sine/` directory.
2. Refer to the config file, `config.json`, and make changes as necessary.
    
    ```sh
    "mini_batch_size": 64,          # Size of Minibatch
    "learning_rate": 0.01,          # Learning rate (eta)
    "num_iterations": 1000,         # Number of epochs
    "activation": "tanh",           # Activation function (tanh, relu or sigmoid)
    "cost_func": "mse",             # Cost function (mse, mape or log)
    "regularisation": "L2",         # Regularisation (L2 or none)
    "lambda_reg": 0.1,              # L2 regularisation parameter (lambda)
    "optimizer": "adam",            # Optimizer (momentum, adam or none)
    "beta": 0.9,                    # Momentum parameter
    "beta1": 0.9,                   # Adam parameter (beta1)
    "beta2": 0.999,                 # Adam parameter (beta2)
    "epsilon": 1e-8,                # Adam parameter (epsilon)
    "layers_dims": [1, 20, 20, 1],  # Neural network layers
    "early_stopping_patience": 0    # Early stopping patience (If cost does not improve after the number of layers provided, training stops)
    ```

> [!WARNING]  
> Comments in the above snippet have been provided for illustration purposes. JSON does not support comments. 

3. Run `driver.py`. This will provide a cost convergence plot (`cost_convergence.png`), R2 scatterplot (`r2_scatter_plot.png`) and a comparison between the actual sine curve  and the predicted curve (`sine_comparison.png`).


### Executing the CCPP Predictor
1. Navigate to the `ccpp/` directory.
2. Refer to the config file, `config.json`. Only one additional change to the config file of the Sine predictor has been made, which is the addition of Dropout.
    
    ```sh
    "dropout_keep_prob": 1.0,           #  Dropout keep probability (layers to keep i.e. 0.9 implies 90% of layers are preserved from one epoch to another)
    ```
> [!WARNING]  
> Comments in the above snippet have been provided for illustration purposes. JSON does not support comments. 

3. Run `driver.py`. This will provide a cost convergence plot (`cost_convergence.png`) and an R2 scatterplot (`r2_scatter_plot.png`). Additionally, the terminal output will provide the Training and Validation errors (MSE) after each epoch and Validation and Testing MAPE after the full run.
> [!IMPORTANT]  
> The [Combined Cycle Power Plant dataset](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant) has already been provided as a CSV file in the same directory, there is no need to install it again.


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
