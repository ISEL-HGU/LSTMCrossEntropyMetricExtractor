#!/bin/sh
# Master LSTM
python main_line_GPU0_master.py --batch_size 32 --test_batch_size 32 --epochs 20 --lr 0.01 --seq_size 32 --embedding_size 64 --lstm_size 64 --gradients_norm 5 --train_file /home/eunjiwon/Git/Collect-Data-with-BugPatchCollector/TrainData/Master_AllCommitsAddedLines.txt




