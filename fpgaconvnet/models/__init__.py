"""
Functional model of hardware modules, layers, partitions and networks. Also contains performance and resource models.

In order to do the CNN to hardware mapping, a model of the hardware is needed. There are four levels of abstraction for the final hardware: 

- `modules`
- `layers` 
- `partition` 
- `network` 

At each level of abstraction, there is an associated performance and resource estimate so that the constraints for the optimiser can be obtained.

"""
