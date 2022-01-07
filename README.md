# Enabling Binary Neural Network Training on the Edge

In this repository we introduce a low-cost binary neural network training strategy exhibiting sizable memory footprint and energy reductions while inducing little to no accuracy loss vs Courbariaux & Bengio's standard binary neural network (BNN) training approach, thereby increasing the viability of deep neural network training on the edge.

These resource decreases are primarily enabled by keeping activations exclusively in binary format throughout both forward and backward propagation, while the standard BNN training method stores activations only in forward propagation, not backward propagation.
Please checkout our [paper](https://arxiv.org/abs/2102.04270) for more details.

## Repo organization
We first emulated our BNN traning method using TensorFlow on GPUs, then implemented it in C++ targetting Raspberry Pi 3B+'s ARM CPU.
We also include our implementations of the standard BNN training method on Raspberry Pi written in both C++ and Python with TensorFlow, which we used as baselines for performance comparison.
Memory profilers were also included with our Raspberry Pi prototypes to show our method's memory savings.

* __rpi_prototype/cpp_bnn__: C++ implementations of both standard and our BNN training methods targetting Raspberry Pi (with valgrind-based memory profiler).
* __rpi_prototype/python_bnn__: Python (TF) implementation of the standard method targetting Raspberry Pi (with PyPI memory profiler).
* __training_emulation_on_gpu__: Emulation of our method using TF on GPUs.

## Results

### Results Obtained from GPU Emulation

<table>
  <tr>
    <td>Dataset</td>
    <td>Model</td>
    <td>Batch size</td>
    <td>Top-1 accuracy (%)</td>
    <td>Modelled memory footprint savings (x&#8595;)</td>
  </tr>
  <tr>
    <td>MNIST</td>
    <td>Multilayer Perceptron</td>
    <td>100</td>
    <td>96.90</td>
    <td>2.78</td>
  </tr>
  <tr>
    <td>CIFAR-10</td>
    <td>CNV</td>
    <td>100</td>
    <td>83.08</td>
    <td>4.17</td>
  </tr>
  <tr>
    <td>SVHN</td>
    <td>CNV</td>
    <td>100</td>
    <td>94.28</td>
    <td>4.17</td>
  </tr>
  <tr>
    <td>CIFAR-10</td>
    <td>BinaryNet</td>
    <td>100</td>
    <td>89.90</td>
    <td>3.71</td>
  </tr>
  <tr>
    <td>SVHN</td>
    <td>BinaryNet</td>
    <td>100</td>
    <td>95.93</td>
    <td>3.71</td>
  </tr>
  <tr>
    <td>ImageNet</td>
    <td>ResNetE-18</td>
    <td>4096</td>
    <td>57.04</td>
    <td>3.78</td>
  </tr>
  <tr>
    <td>ImageNet</td>
    <td>Bi-Real-18</td>
    <td>4096</td>
    <td>54.45</td>
    <td>3.78</td>
  </tr>
</table>

### Results Measured from Training BNNs on Raspberry Pi 3B+

<table>
  <tr>
    <td>Dataset</td>
    <td>Model</td>
    <td>Batch size</td>
    <td>Training time per batch (s)</td>
    <td>Memory footprint (MiB)</td>
    <td>Energy per batch (J)</td>
  </tr>
  <tr>
    <td>MNIST</td>
    <td>Multilayer Perceptron</td>
    <td>200</td>
    <td>0.98</td>
    <td>5.00 (52x&#8595; vs. Keras)</td>
    <td>1.42</td>
  </tr>
  <tr>
    <td>CIFAR-10</td>
    <td>BinaryNet</td>
    <td>40</td>
    <td>121.20</td>
    <td>154.37 (Keras doesn't fit)</td>
    <td>136.53</td>
  </tr>
</table>

## Citation

If you make use of this code, please acknowledge us by citing our [paper](https://dl.acm.org/doi/abs/10.1145/3469116.3470015):
    
    @inproceedings{BNN_ON_EDGE,
        author = {Wang, Erwei and Davis, James J. and Moro, Daniele and Zielinski, Piotr and Lim, Jia Jie and Coelho, Claudionor and Chatterjee, Satrajit and Cheung, Peter Y. K. and Constantinides, George A.},
        title = {Enabling Binary Neural Network Training on the Edge},
        year = {2021},
        doi = {10.1145/3469116.3470015},
        booktitle = {Proceedings of the 5th International Workshop on Embedded and Mobile Deep Learning}
        }
