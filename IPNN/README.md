# Adversarial-Sample
This is an implementation of Intriguing Properties of Neural Networks in the form of pytorch

## Model.py
  This file is used to train a convolutional neuron network to recognition MNIST dataset.
  The model is saved in model.pt


## first_finding.py
  This file is used to demonstrate the first conclusion claimed in the paper

> it is the space,rather than the individual units,that constrains the semantic information in the high layers of neuron networks

  **result**
![avatar](/result/1.PNG)
![avatar](/result/2.PNG)
![avatar](/result/3.PNG)
![avatar](/result/4.PNG)

more details about code is in the file

## second_finding.py
  This file is used to generate adversarial sample in the way claimed in the paper
    
  Minimize c|r| + loss(x+r,l) where c is a punishment factor, r is the perturbation, x is the original picture, l is the target class.

  Details are also in the file

  **result**

![avatar](/result/before_attack.PNG)
![avatar](/result/after_attack.PNG)