# # Φ<sub>*Flow*</sub> Architecture


## Context

![Context](documentation/figures/Context.png)

|    Actor / System    |    Description                                                                                  |
|----------------------|-------------------------------------------------------------------------------------------------|
|    ML Researcher     |    Scientist interested in training   ML models, publishing results                             |
|    User              |    Person who wants to run built-in   simulations and store / analyse the results               |
|    NumPy             |    Non-differentiable Python   computing library                                                |
|    TensorFlow        |    Machine-learning framework   supporting GPU computations and reverse-mode differentiation    |

## Building Blocks

![Building Blocks](documentation/figures/Building_Blocks.png)

|    Actor / System |    Description                                                                                        |
|-------------------|-------------------------------------------------------------------------------------------------------|
|    Model          |    Allows setting up simulations and   GUI                                                            |
|    TF Model       |    Trains neural networks, creates   logs, visualizes results with UI                                 |
|    Data           |    Writes and loads data from disc                                                                    |
|    UI             |    Hosts web server to display data   of Model                                                        |
|    Physics        |    Defines simulation classes,   implements built-in simulations like Navier-Stokes,   Schrödinger    |

## Module dependencies

![Module Diagram](documentation/figures/Module_Diagram.png)
