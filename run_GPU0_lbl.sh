#!/bin/sh
for projectname in "bval" "incubator-hivemall";
do
python main_line_by_line_GPU0.py --batch_size 32 --test_batch_size 16 --epochs 20 --lr 0.01 --seq_size 32 --embedding_size 64 --lstm_size 64 --gradients_norm 5 --input_csv_metric_file /home/eunjiwon/Git/Collect-Data-with-BugPatchCollector/Output/DP/${projectname}-reference/${projectname}_developer.csv --output_csv_metric_file ./data/test/${projectname}_Add_LSTM_metric_lbl.csv --train_file /home/eunjiwon/Git/Collect-Data-with-BugPatchCollector/TrainData/${projectname}_AllCommitsAddedLines.txt --test_file /home/eunjiwon/Git/Collect-Data-with-BugPatchCollector/TrainData/Commit/${projectname}/
done






