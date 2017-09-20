How to run Parallel training:

command:

mpirun -n 4 python lstmUtils.py --doBatch n --Version V100 --nEpoch 5 --Mode M  --Var JF --nEvents 20000 --doTrainC y --Model LSTM --padding pre --nMaxTrack 15 --nLSTMNodes 5 --nLSTMClass 4 --nLayers 2

where:

-n _  #allows you to specify the number of cores to allocate
