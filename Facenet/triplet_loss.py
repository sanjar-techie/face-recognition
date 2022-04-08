from turtle import clear
import tensorflow as tf

# from the facenet paper, triplet loss function implementation
def triplet_loss(y_true:tf.Tensor, y_pred:tf.Tensor, alpha:float = 0.2):
    """
    -- Explanation :
    this function compares the triplet loss ()

    -- Args:
    y_true -- true labels, 
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    -- Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis = -1)

    # the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis = -1)

    # subtract the two previous distances and add alpha.
    basic_loss = pos_dist- neg_dist + alpha
    
    # Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss

# test
with tf.Session() as test:
    tf.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random.normal([3, 128], mean=6, stddev=0.1, seed = 1),
            tf.random.normal([3, 128], mean=1, stddev=1, seed = 1),
            tf.random.normal([3, 128], mean=3, stddev=4, seed = 1))
    loss = triplet_loss(y_true, y_pred)

    print("loss = " + str(loss.eval()))