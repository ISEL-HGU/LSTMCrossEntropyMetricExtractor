#!/bin/sh
for projectname in "metamodel" "nutch" "camel" "eagle";
do
python main_GPU0.py --batch_size 32 --test_batch_size 16 --epochs 20 --lr 0.01 --seq_size 32 --embedding_size 64 --lstm_size 64 --gradients_norm 5 --input_csv_metric_file ./data/test/${projectname}_20170101_to_20190630/labeled.csv --output_csv_metric_file ./data/test/${projectname}_20170101_to_20190630/Add_LSTM_metric.csv --train_file ./data/train/${projectname}_beginning_to_20161231.txt --test_file ./data/test/${projectname}_20170101_to_20190630/CommitFiles/
done


    
