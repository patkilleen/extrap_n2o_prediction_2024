#create directory for output files
mkdir output

#run deep learning and machien learning model single-dataset (single-year interpolation) experiments for 2022 chamber 2 at 180 min temporal resolution
mkdir output/single-year-interpolation
python experimenter.py --inFile input/configs/single-year-interpolation/2022/C2/inC2Config180.csv  --outDirectory output/single-year-interpolation --trainTestFileSpecific False

#train dataset provided and test dataset provided (multi-year extrapolation), training data = 2021, test data = 2022, without soil synthetic nitrogen content feature (SSNC), block cross-validation, and   model = LSTM with 10 tensors
mkdir output/multi-year-extrapolation
python experimenter.py --inFile  input/configs/multi-year-extrapolation/2021-train-2022-test/without-SSNC/blockCVConfigLSTM_S.csv  --outDirectory output/multi-year-extrapolation --trainTestFileSpecific True
