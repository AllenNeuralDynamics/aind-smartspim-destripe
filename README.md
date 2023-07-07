# aind-smartspim-destripe

Source code to remove streaks from lightsheet images acquired with the SmartSPIM microscope. Currently, we are using a log spatial Fast Fourier Transform to remove the streaks. This works for us since our brains have cells with high intensity values that when we apply a dual-band filtering, artifacts are generated.

![raw data](https://github.com/AllenNeuralDynamics/aind-smartspim-destripe/blob/main/metadata/imgs/raw.png?raw=true)

## Dual-band
![dual band filtering](https://github.com/AllenNeuralDynamics/aind-smartspim-destripe/blob/main/metadata/imgs/filtered_dual_band.png?raw=true)

## Log Space FFT
![log space filtering](https://github.com/AllenNeuralDynamics/aind-smartspim-destripe/blob/main/metadata/imgs/filtered_log_space.png?raw=true)