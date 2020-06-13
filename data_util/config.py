import os

root_dir = os.path.expanduser("~")

# train_data_path =  os.path.join(root_dir, "/Users/rowancassius/Desktop/pointer_summarizer-master/training_ptr_gen")
# eval_data_path =   os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/val.bin")
# decode_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/test.bin")
# vocab_path =       os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/vocab")
# log_root =         os.path.join(root_dir, "ptr_nw/log")


train_data_path = "/Users/rowancassius/Desktop/pointer_summarizer-master/training_ptr_gen/data_file.txt"
eval_data_path = "/Users/rowancassius/Desktop/pointer_summarizer-master/training_ptr_gen/data_file.txt"
vocab_path = "/Users/rowancassius/Desktop/pointer_summarizer-master/training_ptr_gen/enron_vocab.txt"
log_root = "/Users/rowancassius/Desktop/pointer_summarizer-master/training_ptr_gen"
decode_data_path = "/Users/rowancassius/Desktop/pointer_summarizer-master/training_ptr_gen/data_file.txt"


# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size= 32
max_enc_steps=25
max_dec_steps=12
beam_size=4
min_dec_steps=2
vocab_size=1546

lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 10

use_gpu=False

lr_coverage=0.15
