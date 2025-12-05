# Metropolis Adjusted Microcanonical Hamiltonian Monte Carlo


Qualifying paper #2 with Trevor Campbell:
[Metropolis Adjusted Microcanonical Hamiltonian Monte Carlo](https://arxiv.org/pdf/2503.01707) by Jakob Robnik, Reuben Cohn-Gordon, and Uroš Seljak.


The project directory is organized as:  
* [/code](https://github.com/1saacRankin/QP2-microcanonical-HMC/tree/main/code) for the code to run experiments for the small project.
* [/report](https://github.com/1saacRankin/QP2-microcanonical-HMC/tree/main/report) for the files to produce the report.


The project compares hyperparameter tuning schemes for the Metropolis-adjusted microcanonical sampler.

We compare tuning the step size with trajectory length fixed, followed by tuning trajectory length with step size fixed, with tuning both simultaneously using Bayesian optimization.






To compile the report: 

Install the libraries in requirements.txt

Navigate to code/bayesopt_vs_autotune.py

Run: python bayesopt_vs_autotune.py > tuning_results.txt

Navigate to report/report.tex and compile


