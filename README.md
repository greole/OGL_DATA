# OGL_DATA

This repository collects benchmarking data for the [OGL](https://github.com/greole/OGL) library. Benchmark results are generated with the [OBR](https://github.com/greole/OBR) scripts.

# Structure

The benchmark data is structured the following way.

1. case (separate folder)
2. OGL version or tag
3. Ginkgo version or tag
3. the device on which the data was generated
4. a set of csv files

# Using the provided scripts
The simplest method to import the provided data_processing scripts is by adding the corresponding path to the sys.path
~~~
import sys
sys.path.append("<PATH_TO_SCRIPTS>")

import ogl_data_processing as odp
~~~
