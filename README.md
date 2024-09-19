# Deeplearning-South-China-Sea-regional-temperature
regional temperature prediction base on ConvLSTM
# Data information
- collected from ECMWF era5 dataset
- hourly temperature on pressure level 850kpa from 2009 to 2023 in South China Sea area.
# model architecture
- 3 layer Convlstm.
- accept shape (3, 69, 53, 1) as input which mean time step is 3.
- output is (69,53,1). so use previous 3 hours' data to predict the next hour.
  
