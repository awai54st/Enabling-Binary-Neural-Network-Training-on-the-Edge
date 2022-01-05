# Enabling Binary Neural Network Training on the Edge

In this repository we introduce a low-cost binary neural network training strategy exhibiting sizable memory footprint and energy reductions while inducing little to no accuracy loss vs Courbariaux & Bengio's standard binary neural network (BNN) trainnig approach.

These resource decreases are primarily enabled by keeping activations exclusively in binary format throughout both forward and backward propagation, while the standard BNN training method stores activations only in forward propagation, not backward propagation.
Please checkout our [paper](https://arxiv.org/abs/2102.04270) for more details.

## Repo organization
We first emulated our BNN traning method using TensorFlow on GPUs, then implemented it in C++ targetting Raspberry Pi 3B+'s ARM CPU.
We also include our implementations of the standard BNN training method on Raspberry Pi written in both C++ and Python with TensorFlow, which we used as baselines for performance comparison.
Memory profilers were also included with our Raspberry Pi prototypes to show our method's memory savings.

* __rpi_prototype/cpp_bnn__: C++ implementations of both standard and our BNN training methods targetting Raspberry Pi (with valgrind-based memory profiler).
* __rpi_prototype/python_bnn__: Python (TF) implementation of the standard method targetting Raspberry Pi (with PyPI memory profiler).
* __training_emulation_on_gpu__: Emulation of our method using TF on GPUs.

## Citation

If you make use of this code, please acknowledge us by citing our [paper](https://arxiv.org/abs/2102.04270):

    @article{BNN_ON_EDGE,
        author    = {Wang, Erwei and Davis, James J. and Moro, Daniele and Zielinski, Piotr and Coelho, Claudionor and Chatterjee, Satrajit and Cheung, Peter Y. K. and Constantinides, George A.},
        title     = {Enabling Binary Neural Network Training on the Edge},
        journal   = {CoRR},
        year      = {2021},
        eprinttype = {arXiv},
        eprint    = {2102.04270}
    }
