# RS-BNN: A Deep Learning Framework for the Optimal Beamforming Design of Rate-Splitting Multiple Access

This is a code package related to the following [paper](https://ieeexplore.ieee.org/document/10586792):

Y. Wang, Y. Mao and S. Ji, "RS-BNN: A Deep Learning Framework for the Optimal Beamforming Design of Rate-Splitting Multiple Access," in *IEEE Transactions on Vehicular Technology*, vol. 73, no. 11, pp. 17830-17835, Nov. 2024, doi: 10.1109/TVT.2024.3423002. 

# Content of Code Package

Here is a detailed description of the package:

*   The code in all packages are implemented in Python environment.
*   The following code is used to reproduce the data presented in **Fig. 4** for the case where the number of antennas `$N_t$` and the number of users `$K$` are both set to 2. Other data points can be reproduced by modifying the corresponding parameters.In the `data` folder, we provide two datasets: `RSMA_test.mat` and `RSMA22_train.mat`.The file `RS-BNN.py` contains implementations of both the black-box CNN method and the proposed RS-BNN method. The file `Data_gen.py` is responsible for generating the training and test datasets, and it also includes implementations of the FP-HFPI algorithm and the WMMSE algorithm for comparison.

# Abstract of the Article

Rate splitting multiple access (RSMA) relies on beamforming design for attaining spectral efficiency and energy efficiency gains over traditional multiple access schemes. While conventional optimization approaches such as weighted minimum mean square error (WMMSE) achieve suboptimal solutions for RSMA beamforming optimization, they are computationally demanding. A novel approach based on fractional programming (FP) has unveiled the optimal beamforming structure (OBS) for RSMA. This method, combined with a hyperplane fixed point iteration (HFPI) approach, named FP-HFPI, provides suboptimal beamforming solutions with identical sum rate performance but much lower computational complexity compared to WMMSE. Inspired by such an approach, in this work, a novel deep unfolding framework based on FP-HFPI, named rate-splitting-beamforming neural network (RS-BNN), is proposed to unfold the FP-HFPI algorithm. Numerical results indicate that the proposed RS-BNN attains a level of performance closely matching that of WMMSE and FP-HFPI, while dramatically reducing the computational complexity.

# License and Referencing

This code package is licensed under the GPLv2 license. If you in any way use this code for research that results in publications, please cite our original article listed above.

# Acknowledgements

This work has been supported in part by the National Nature Science Foundation of China under Grant 62201347; and in part by Shanghai Sailing Program under Grant 22YF1428400.
