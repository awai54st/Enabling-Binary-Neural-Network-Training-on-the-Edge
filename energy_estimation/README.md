Our energy estimator is adapted from QKeras' energy estimator, qenergy.
To run our energy estimator, please download QKeras, then under qkeras/qkeras/qtools/qenergy/ replace qenergy.py with our qenergy.py. 
Then, add example_get_energy_small.py and example_get_energy_resnetE.py to qkeras/qkeras/qtools/examples/ 
Finally, under qkeras/qkeras/qtools/examples/, run either python example_get_energy_small.py for small-scale datasets, or python example_get_energy_resnetE.py for ResNetE-18.
Energy estimates will show in the following format.

total_cost:  *
total_rw_cost:  *
total_op_cost:  *

To change settings, please see #config in qenergy.py.