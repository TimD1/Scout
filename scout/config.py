import multiprocessing as mp

# reset by each submodule, allow accessing args in helper functions
args = None

# global model
model = None

# globals to keep track of progress
test_count = mp.Value('i', 0) # total positions to test
cand_count = mp.Value('i', 0) # candidate positions selected
pos_count = mp.Value('i', 0)  # positions actually selected
