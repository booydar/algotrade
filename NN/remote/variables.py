# from sklearn.model_selection import ParameterGrid

# TASK_NAME = 'copy'
# TRAIN_SIZE = 2_0
# VAL_SIZE = 1_0
# TEST_SIZE = 3_0
# NUM_INITS = 1


# NUM_BATCHES = int(1e2)
# BATCH_SIZE = 32
# LEARNING_RATE = 3e-4
# GENERATE_EVERY = 100
# NUM_TOKENS = 16 + 2
# ENC_SEQ_LEN = 16
# DEC_SEQ_LEN = 32


# model_parameters = ParameterGrid({'dim': [64, 128],
#     'tie_token_embeds': [True],
#     'return_tgt_loss': [True],
#     'enc_num_tokens': [NUM_TOKENS],
#     'enc_depth': [1],
#     'enc_heads': [1],
#     'enc_max_seq_len': [ENC_SEQ_LEN],
#     'dec_num_tokens': [NUM_TOKENS],
#     'dec_depth': [1],
#     'dec_heads': [1], 
#     'dec_max_seq_len': [DEC_SEQ_LEN],
#     'enc_num_memory_tokens': [0, 1, 2, 3]})