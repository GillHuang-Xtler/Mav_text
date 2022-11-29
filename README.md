# Poisoners_FL
Poisoners for label flipping and backdooring. It is now ready for label flipping with contribution measurements.

# To run this file

0. py3.7 + pip install -r requirements.pip 
1. generate_data_distribution.py
2. generate_default_models.py
3. label_flipping_attack.py

- The arguments are in 'arguments.py' + label_flipping_attack.py.
- The traning round (epoch in this repository) should be set to 200.

-MNIST:Batch size: 4; LR: 0.001; Momentum: 0.5; Sched-uler step size: 50; Scheduler gamma: 0.1;α: 0.15,β:0.0015;
•Fashion-MNIST:Batch size: 4; LR: 0.001; Momentum:0.9; Scheduler step size: 10; Scheduler gamma: 0.1;α:0.15,β:0.0015;
•Cifar-10:Batch  size:  10;  LR:  0.01;  Momentum:  0.5;Scheduler step size: 50; Scheduler gamma: 0.5;α: 0.11,β:0.0015;
•STL-10:Batch size: 10; LR: 0.01; Momentum: 0.5; Sched-uler step size: 50; Scheduler gamma: 0.5;α: 0.11,β:0.001;


5201 5202 5203: fashion-maverick-1-fedprox-mu0.01
5204 5205 5206: cifar-maverick-1-fedprox-mu0.01
8191-mnist-1-sv0.01-sgd-200
8190-mnist-1-sv0.015-sgd-200
8192-mnist-1-sv0.015-avg-200
8311-stl-1-sv0.015-sgd-200-9016-9017
8312-stl-1-sv0.015-avg-200-9015
9011-fashion-1-sv0.015-avg-200
9012-fashion-1-sv0.015-sgd-200
9013-cifar-1-sv0.015-sgd-200
9014-cifar-1-sv0.015-avg-200
9031-fashion-1-new-sgd-200-beta0.009-9032-9033
9041-fashion-1-new-avg-200-beta0.009-9042-9043
9044-fashion-1-new-avg-200-beta0.008-9045-9046
9061-fashion-1-new-sgd-200-beta0.008-9062-9063
9071-fashion-1-new-avg-200-beta0.01-9072-9073
9074-fashion-1-new-sgd-200-beta0.01



