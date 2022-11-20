# On accelerating reaction optimization using computational Gibbs energy barriers: A numerical consideration utilizing a computational dataset
![Fig1](https://user-images.githubusercontent.com/64031703/202848690-eb02a8c7-e409-4d3d-9cdd-da134516f4e2.png)

## Overview
Source code to reproduce [the work's preprint](https://chemrxiv.org/engage/chemrxiv/article-details/635e6609ac45c788bb9e44d7).

## Environment setup
(0) Create anaconda environment
```
conda create --name bo python=3.7.5
```

(1) Install  third-party libraries
```
conda install openpyxl numpy pandas matplotlib scikit-learn
```
(2) Install PyTorch and GPyTorch
```
conda install -c pytorch pytorch=1.3.1
conda install -c gpytorch gpytorch=1.0.0
```

## Usage
Main steps to reproduce:

  (0) Open `benchmark.py` and set the arguments of the benchmark function.  
  ```
  def benchmark(ini_seed=0, loop_num=200, max_trial=100, batch_size=5, finish_thresh=-3.4):
  ```
  (1) Open `run.py` and set the arguments of the ReactionOpt class.
  ```
  bo = ReactionOpt(input_name='read', desc_data=descritor,
                   init_method='rand', maxtrial=100,
                   finish_thresh=-3.4, target='Barrier',
                   opt_type='minimize', ard=False, gpu=False,
                   batch_size=5, batch_magnification=1, second_opt=True,
                   target_scaling='standard', deviation=['normal', 0, 5])
  ```
  (2) Run `benchmark.py`.  The program will run in the foreground, and the performance benchmarking process will be output to stdout one after another. At the end, a summary of the performance (list of success rates) is displayed.  
  If you want to run the program in the background, use the `nohup` command, for example.  
  ```
  nohup python benchmark.py > benchmark.log &
  ```
  It is also possible to run Bayesian optimization only once by running `run.py`.
