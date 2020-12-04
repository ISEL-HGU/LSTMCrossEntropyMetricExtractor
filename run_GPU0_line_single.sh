#!/bin/sh
# Single LSTM
for projectname in "ace" "ant-ivy" "bigtop" "bval" "camel" "cayenne" "cordova-android" "creadur-rat" "crunch" "deltaspike" "gora" "groovy" "guacamole-client" "incubator-dolphinscheduler" "incubator-hivemall";
do
python main_line_GPU0.py --batch_size 32 --test_batch_size 32 --epochs 20 --lr 0.01 --seq_size 32 --embedding_size 64 --lstm_size 64 --gradients_norm 5 --input_csv_metric_file [SurprisedBasedMetricsExpDataExtractor_PATH]/Output/DP/label_DP/${projectname}_developer.csv --output_csv_metric_file ./data/test/${projectname}_Line_Single_LSTM_Metric.csv --train_file [SurprisedBasedMetricsExpDataExtractor_PATH]/TrainData/${projectname}_AllCommitsAddedLines.txt --test_file [SurprisedBasedMetricsExpDataExtractor_PATH]/TrainData/Commit/${projectname}/
done
