## N2O Prediction Extrapolation and Interpolation

This is the official documentation of the code repository of the paper: Patrick Killeen, Ci Lin, Futong Li, Iluju Kiringa, and Tet Yeap "Nitrous Oxide Emission Prediction Using IoT Soil and Weather Sensor Data".

## Requirements

This is my experiment environment.
-	64-bit Linux (Ubuntu 20.04) desktop with a NVIDIA GeForce RTX 3060 graphics card
-	Python 3.8.19
-	TensorFlow version 2.12.0 
-	CUDA version 11.8
-	cuDNN version 8.6
-	sklearn version 1.3.2
-	numpy version 1.24.3
-	pandas version 2.0.3


## Example of Running an LSTM Experiment

The below Linux shell commands will run a full example single-year interpolation for every model, and a multi-year extrapolation example using LSTM
```
mkdir output

mkdir output/single-year-interpolation
python experimenter.py --inFile input/configs/single-year-interpolation/2022/C2/inC2Config180.csv  --outDirectory output/single-year-interpolation --trainTestFileSpecific False

mkdir output/multi-year-extrapolation
python experimenter.py --inFile  input/configs/multi-year-extrapolation/2021-train-2022-test/without-SSNC/blockCVConfigLSTM_S.csv  --outDirectory output/multi-year-extrapolation --trainTestFileSpecific True
```

## Documentation

The API and documentation of the project can be found in documentation/n2o-predict-documentation.pdf.

### File Descriptions
We describe the files of interest of the project in this section.

- documentation/n2o-predict-documentation.pdf: the documentation that explains the API of the scripts

The scripts required to run the N2O prediction:
- common.py: common functionality shared by all models
- dataset.py: all the logic for reading and pre-processing the datasets
- experimenter.py: main, the core of running all the single-dataset and multi-data cross-validation experiments
- model.py: all the learning algorithm logic is in here
- myio.py: all the file input and output logic is in here

Other scripts are below:
- plsr.py: the feature selection logic that uses partial least squares regression
- resampling.py: the time-series analysis logic for helping with data pre-processing.
Directories:
- input/ #holds all input files
	- configs/ #holds the configuration files used in the study
		- multi-year-extrapolation/ #multi-year extrapolation configuration files
		- single-year-interpolation/ #single-year extrapolation configuration files
			- 2021-train-2022-test/ #configuration files for training models using 2021 data and testing models using 2022 data
			- 2022-train-2021-test/ #configuration files for training models using 2022 data and testing models using 2021 data
				- without-SSNC/ #configuration files involving datasets without the soil-synthetic-nitrate-content feature
				- with-SSNC/ #configuration files involving datasets with the soil-synthetic-nitrate-content feature
	- datasets/ #holds the dataset files	
		- single-year-interpolation/ #datasets used in the single-year interpolation experiments
			- <year>/ #contains datasets of 2021, 2022, and 2023 growing seasons
				- C<chamber ID>/ #datasets are further categorized by the automated gas chamber used to gather the emissions found in the dataset
					- <temporal resolution>min.csv #two datasets for each chamber of varying temporal resolution (30 minutes and 180 minutes)
		- multi-year-extrapolation/ # datasets used in the multi-year extrapolation experiments
			- 2021-C4-180min.csv # 2021 chamber 4 180 min resolution dataset
			- 2022-C1-180min.csv # 2022 chamber 1 180 min resolution dataset
		- selected-sensors/ #stores the files that indicate what feature is selected for each type of experiment  using same files structure as input/datasets
- output: the directory that will be created when the project is run using the example scripts runExperiments.sh and runFeatureSelection.sh
- raw-data/ #holds the raw dataset files used to pre-process the data and created the final dataset files in the datasets/ folder
	- licor-n2o-data/ #N2O emission data gathered from the LI-COR automated gas chamber
		- 2021.csv #N2O emission data from 2021 for each chamber
		- 2022.csv #N2O emission data from 2022 for each chamber
		- 2023.csv #N2O emission data from 2023 for each chamber
	- pessl-soil-weather-data/ # the Pessl sensor node data containing soil and weather readings from 4 sensors nodes near the LI-COR chambers
		- <year>/ #directories of data categorized by year
			- sensor-node-<sensor node ID>.csv #sensor readings from four Pessl sensor nodes (sensor node 00209FC8, 01209E4F, 01209E52, and 01209E58)
		- predictor-node-soil-data/ #contains the data from our Raspberry Pi-based soil data gathering device
			- 2023.csv #soil data gathered from 2023 (the only season our device was deployed)