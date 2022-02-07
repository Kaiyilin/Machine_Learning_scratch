# A self-attention module takes in n inputs, and returns n outputs.
# The self-attention mechanism allows the inputs to interact with each other (“self”) and find out who they should pay more attention to (“attention”). 
# The outputs are aggregates of these interactions and attention scores.
# i.e. I: vector with size n, O: attention scores, self attention interactions

"""
Steps:
    1. Prepare inputs
    2. Initialise weights
    3. Derive key, query and value
    4. Calculate attention scores for Input 1
    5. Calculate softmax
    6. Multiply scores with values
    7. Sum weighted values to get Output 1
    8. Repeat steps 4 to 7 for Input 2 & Input 3
"""
# Load libraries
import tensorflow as tf 

# Prepare inputs
# 3 inputs, each with dimension of 4
x = [
    [1, 0, 1, 0],
    [0, 2, 0, 2],
    [1, 1, 1, 1]
]

x = tf.constant(x, dtype=tf.float32)

print(x)

# Initialise weights of 3 representations
# key, query, value
# since we have every input with size 4, and suppose we want these
# representations to have dimension of 3 thus we have 

w_key = [
    [0, 0, 1],
    [1, 1, 0],
    [0, 1, 0],
    [1, 1, 0]
]
w_query = [
    [1, 0, 1],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 1]
    ]
w_val = [
    [0, 2, 0],
    [0, 3, 0],
    [1, 0, 3],
    [1, 1, 0]
]

# change the weight to tf.Variable for training
w_key = tf.Variable(w_key, trainable=True, dtype=tf.float32)
w_query = tf.Variable(w_query, trainable=True, dtype=tf.float32)
w_val = tf.Variable(w_val, trainable=True, dtype=tf.float32)

# Derive key, query and value
key_rep = tf.matmul(x, w_key)
val_rep = tf.matmul(x, w_val)
query_rep = tf.matmul(x, w_query)

print(f"key representations:\n{key_rep}")
print(f"value representations:\n{val_rep}")
print(f"query representation:\n{query_rep}")

# Calculate attention scores for Inputs 
# To obtain attention scores, we start off with taking a dot product between Inputs query with all keys including itself.

attn_scrs = tf.matmul(query_rep, tf.transpose(key_rep))

print(f"attention scores:\n{attn_scrs}")

# Calculate softmax

attn_scrs_softmax = tf.nn.softmax(attn_scrs)

print(f"attn_scrs_softmax:\n{attn_scrs_softmax}")

# Multiply scores with values
# change the val_rep to shape (3, 1, 3)
# and the transposed attn_scrs_softmax to (3, 3, 1)
# The result is the weughted values with shape (3, 3, 3) 
weighted_values = val_rep[:, None] * tf.transpose(attn_scrs_softmax)[:, :, None]

print(f"weighted values:\n{weighted_values}")

# Sum the weighted values
outputs = tf.reduce_sum(weighted_values, axis=0)

print(f"Outputs:\n{outputs}")