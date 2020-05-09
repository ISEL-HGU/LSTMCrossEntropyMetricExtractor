import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import Counter
import os
# from argparse import Namespace
import argparse
from itertools import chain
import csv
import os.path

def tokenize_data(filename):
  with open(filename, "r") as file:
    all_lines = file.readlines()
  list_split_WS = [] 
  list_split_dot = []
  return_list = []
  for line in all_lines:
    list_split_WS.append(line.split())
  list_split_WS = list(chain.from_iterable(list_split_WS))
  for token in list_split_WS:
    list_split_dot.append(token.split('.'))
  list_split_dot = list(chain.from_iterable(list_split_dot))
  for token in list_split_dot:
    if token == '' :
      pass
      # print("tokenizing data... ignore empty string")
    else :
      return_list.append(token)
  return return_list

def get_data_from_file(file, batch_size, seq_size):
  text = tokenize_data(file)
  # Create two dictionaries, one to convert words into integers indices, 
  # and the other one to convert integer indices back to word tokens
  word_counts = Counter(text)
  sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
  int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
  vocab_to_int = {w: k for k, w in int_to_vocab.items()}
  n_vocab = len(int_to_vocab)
  print('Vocabulary size', n_vocab)
  
  # Covert word tokens into integer indices. 
  # These will be the input to the network
  # We will train a mini-batch each iteration 
  # so we split the data into batches evenly. 
  # Chopping out the last uneven batch
  int_text = [vocab_to_int[w] for w in text]
  num_batches = int(len(int_text) / (seq_size * batch_size))
  if num_batches == 0:
    in_text = int_text
    num_zero_padding = (seq_size * batch_size) - len(in_text)
    print("size of zero padding: ", num_zero_padding)
    in_text = np.pad(in_text, (0, num_zero_padding), 'constant', constant_values=0)
    print("after adding zero padding, the size: ", len(in_text))
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:] # in_text의 두번째 부터 out_text의 처음으로 복사
    out_text[-1] = in_text[0] # in_text의 처음을 out_text의 마지막으로 복사
    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))
  else:
    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_text = int_text[:num_batches * batch_size * seq_size]
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:] # in_text의 두번째 부터 out_text의 처음으로 복사
    out_text[-1] = in_text[0] # in_text의 처음을 out_text의 마지막으로 복사
    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))

  # print(in_text[:10][:10]) # top and left of matrix
  # print(out_text[:10][:10]) # top and left of matrix
  return int_to_vocab, vocab_to_int, n_vocab, in_text, out_text
  
def get_batches(in_text, out_text, batch_size, seq_size):
  # print(np.prod(in_text.shape))
  if int(np.prod(in_text.shape) / (seq_size * batch_size)) == 0:
    yield in_text[:,:], out_text[:,:]
  else:
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for i in range(0, num_batches * seq_size, seq_size):
      yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]

class RNNModule(nn.Module):
  def __init__(self, n_vocab, seq_size, embedding_size, lstm_size):
    super(RNNModule, self).__init__()
    self.seq_size = seq_size
    self.lstm_size = lstm_size
    self.embedding = nn.Embedding(n_vocab, embedding_size)
    self.lstm = nn.LSTM(embedding_size, lstm_size, batch_first=True)
    self.dense = nn.Linear(lstm_size, n_vocab)

  # Take an input sequence and the previous states (hidden states) and produce the output together with states of the currents timestamp
  def forward(self, x, prev_state):
    embed = self.embedding(x)
    output, state = self.lstm(embed, prev_state)
    logits = self.dense(output)
    return logits, state # why return state variable?
  
  # Define one more method to help us set states to zero because we need to reset states at the beginning of every epoch.
  def zero_state(self, batch_size):
    return (torch.zeros(1, batch_size, self.lstm_size), # hidden state (the short-term memory)
            torch.zeros(1, batch_size, self.lstm_size)) # cell state (the long-term memory)

def get_loss_and_train_op(net, args):
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
  return criterion, optimizer
  # gradient clipping doesn't apply here!

def train(in_text, out_text, args, net, device, criterion, optimizer, e):
  print("training...")
  batches = get_batches(in_text, out_text, args.batch_size, args.seq_size)
  state_h, state_c = net.module.zero_state(64)
  # Transfer data to GPU
  state_h = state_h.to(device)
  state_c = state_c.to(device)

  iteration = 0
  for x, y in batches: # x is in_text and y is out_text
    iteration += 1
    # Tell it we are in training mode
    net.train()
    # Reset all gradients
    optimizer.zero_grad()
    # Transfer data to GPU
    x = torch.tensor(x).to(device)
    y = torch.tensor(y).to(device)
    logits, (state_h, state_c) = net(x, (state_h, state_c))
    loss = criterion(logits.transpose(1, 2), y) # why we transpose the logits?
    # Avoid autograd which is given by Pytorch to keep track of the tensor's flow to perform back-propagation.
    state_h = state_h.detach()
    state_c = state_c.detach()
    loss_value = loss.item() # this loss is cross-entropy which is thing I want!!! 
    # Perform back-propagation
    loss.backward()
    # Gradient clipping
    _ = torch.nn.utils.clip_grad_norm_(net.parameters(), args.gradients_norm)
    # Update the network's parameters
    optimizer.step() # the number of parameters update is batch-size * epoch
  # Print the loss value and have the model generate some text for us during training
  print("train_file: ", args.train_file)
  print('Epoch: {}/{}'.format(e, args.epochs), 'Loss (C.E): {}'.format(loss_value)) # here, we just print the size of epoch 

def test(test_in_text, test_out_text, args, net, device, criterion):
  print("test...")
  net.eval() # Tell it we are in evaluation mode
  batches = get_batches(test_in_text, test_out_text, args.test_batch_size, args.seq_size)
  state_h, state_c = net.module.zero_state(64)
  # Transfer data to GPU
  state_h = state_h.to(device)
  state_c = state_c.to(device)
  loss_average = []
  for x, y in batches: # x is test_in_text and y is test_out_text
    # Transfer data to GPU
    x = torch.tensor(x).to(device)
    y = torch.tensor(y).to(device)
    logits, (state_h, state_c) = net(x, (state_h, state_c))
    loss = criterion(logits.transpose(1, 2), y) # why we transpose the logits?
    loss_value = loss.item()
    loss_average.append(loss_value)
    state_h = state_h.detach()
    state_c = state_c.detach()
   
  # this loss is cross-entropy which is thing I want!!! 
  print("test set loss value (C.E.): ", sum(loss_average) / len(loss_average))
  return loss_value

def main():
  # Training settings
  parser = argparse.ArgumentParser(description='LSTM Cross-Entropy Metric')
  parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                      help='input batch size for training (default: 16)')
  parser.add_argument('--test_batch_size', type=int, default=16, metavar='N',
                      help='input batch size for testing (default: 16)')
  parser.add_argument('--epochs', type=int, default=100, metavar='N',
                      help='number of epochs to train (default: 100)')
  parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                      help='learning rate (default: 0.01)')
  parser.add_argument('--train_file', type=str, required=True, default=None,
                      help='path of training set')
  # parser.add_argument('--test_file', type=str, required=True, default=None,
                      # help='path of test set')
  parser.add_argument('--seq_size', type=int, default=32,
                      help='sequence size')
  parser.add_argument('--embedding_size', type=int, default=64,
                      help='embedding size')
  parser.add_argument('--lstm_size', type=int, default=64,
                      help='lstm size')
  parser.add_argument('--gradients_norm', type=int, default=5,
                      help='norm to clip gradients')
  # parser.add_argument('--input_csv_metric_file', type=str, required=True, default=None,
  #                     help='path of input metric csv file')
  # parser.add_argument('--output_csv_metric_file', type=str, required=True, default=None,
  #                     help='path of output metric csv file')
  # parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
  #                     help='SGD momentum (default: 0.5)')
  # parser.add_argument('--no-cuda', action='store_true', default=False,
  #                     help='disables CUDA training')
  # parser.add_argument('--seed', type=int, default=1, metavar='S',
  #                     help='random seed (default: 1)')
  # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
  #                     help='how many batches to wait before logging training status')
  
  args = parser.parse_args()
  int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_file(
    args.train_file, args.batch_size, args.seq_size)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  net = RNNModule(n_vocab, args.seq_size, args.embedding_size, args.lstm_size)
  print ('Available devices ', torch.cuda.device_count())
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # for 19 server
  if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net,dim=1)
  net = net.to(device)
  criterion, optimizer = get_loss_and_train_op(net, args)


  # Training
  for e in range(1, args.epochs + 1):
    train(in_text, out_text, args, net, device, criterion, optimizer, e)
  
  # test_int_to_vocab, test_vocab_to_int, test_n_vocab, test_in_text, test_out_text = get_data_from_file(
  #     args.test_file, 1, args.seq_size)
  # # Test
  # loss_value = test(test_in_text, test_out_text, net, device, criterion)
  # project_list = ["ace", "ant-ivy", "bigtop", "bval", "camel", "cayenne", "cordova-android", "creadur-rat", "crunch", "deltaspike", "gora", "groovy", "guacamole-client", "incubator-dolphinscheduler", "incubator-hivemall"]
  # project_list = ["cayenne", "cordova-android", "creadur-rat", "crunch", "deltaspike", "gora", "groovy", "guacamole-client", "incubator-dolphinscheduler", "incubator-hivemall", "camel"] # gora에서 메모리 터짐...
  # project_list = ["groovy", "guacamole-client", "incubator-dolphinscheduler", "incubator-hivemall", "camel", "gora"] 
  project_list = ["ant-ivy", "bigtop", "bval", "camel", "cayenne", "creadur-rat", "deltaspike", "gora", "guacamole-client", "incubator-hivemall"]
  for project_name in project_list:
    # Save the metric to the output csv file
    with open("/home/eunjiwon/Git/Collect-Data-with-BugPatchCollector/Output/DP/label_DP/" + project_name + "_developer.csv",'r') as csv_input:
      with open("/home/eunjiwon/Git/SoftwareDefectPredictionMetricUsingDeepLearning/data/test/" + project_name + "_Concat_Master_LSTM_Metric.csv", 'w+') as csv_output:
        writer = csv.writer(csv_output, lineterminator='\n')
        reader = csv.reader(csv_input)
        all = []
        row = next(reader)
        # print(row[0])
        row.append('LSTM C.E.')
        all.append(row)
        header_flag = 0
        for row in reader:
          if header_flag == 0: 
            header_flag = 1
            continue # skip header row
          commit_hash_key = row[20].split('-')[0]
          testcommit_name = "/home/eunjiwon/Git/Collect-Data-with-BugPatchCollector/TrainData/Commit/" + project_name + "/" + commit_hash_key + ".txt"
          if os.path.exists(testcommit_name):
            print(testcommit_name)
            test_int_to_vocab, test_vocab_to_int, test_n_vocab, test_in_text, test_out_text = get_data_from_file(
                testcommit_name, args.test_batch_size, args.seq_size)
            if test_n_vocab == 0: continue
            # Test and get LSTM C.E. metric
            loss_value = test(test_in_text, test_out_text, args, net, device, criterion)
            print(testcommit_name, loss_value)
            row.append(loss_value)
            all.append(row)
          else:
            print("Warning - ", testcommit_name, " does not exist :( ")
        writer.writerows(all)
    print("Finish - ", testcommit_name) 
    
if __name__ == '__main__':
  #print(torch.rand(1, device="cuda"))
  #torch.cuda.device(1)
  main()
  


