#!/bin/sh
# Single LSTM
for projectname in "ant-ivy" "bigtop" "bval" "camel" "cayenne" "cordova-android" "creadur-rat" "crunch" "deltaspike" "gora" "groovy" "guacamole-client" "incubator-hivemall";
do
python main_concat_GPU1.py --batch_size 32 --test_batch_size 32 --epochs 20 --lr 0.01 --seq_size 32 --embedding_size 64 --lstm_size 64 --gradients_norm 5 --input_csv_metric_file /home/eunjiwon/Git/Collect-Data-with-BugPatchCollector/Output/DP/label_DP/${projectname}_developer.csv --output_csv_metric_file ./data/test/${projectname}_Concat_Single_LSTM_Metric.csv --train_file /home/eunjiwon/Git/Collect-Data-with-BugPatchCollector/TrainData/${projectname}_AllCommitsAddedLines.txt --test_file /home/eunjiwon/Git/Collect-Data-with-BugPatchCollector/TrainData/Commit/${projectname}/
done




