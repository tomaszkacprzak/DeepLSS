import tensorflow_probability as tfp
import tensorflow as tf

@tf.function()
def tf_neg_likelihood_cholesky_parts(predictions, labels, n, m):
    """
    calculates the negative likelihood loss with tensorflow
    the first n values of predictions are the actual prediction
    from the next elements using a upper triangular matrix is generated
    :param predictions: Predictions of the network
    :param labels: true params
    :param n: number of labels (the rest is used for the covariance matrix)
    :returns: mean_norm, mean_log_det
    """

    eps = 1e-30

    pred_vals, cholesky = tf.split(predictions, [n, m], axis=1, name='likeloss_split_mean_cov')

    # subtract predictions and labels
    res = tf.subtract(pred_vals, labels, name='likeloss_diff_true_pred')

    # make upper triang matrix L^T
    upper_triang = tfp.math.fill_triangular(cholesky, upper=True, name='likeloss_fill_triangular')


    # Get diagonal
    diag = tf.linalg.diag_part(upper_triang, name='likeloss_diag_part')

    # add a small number such that the diag is never zero and log it
    diag += eps    

    # get log determinant
    # https://math.stackexchange.com/questions/3158303/using-cholesky-decomposition-to-compute-covariance-matrix-determinant
    log_det = tf.reduce_sum(tf.math.log(tf.square(diag+eps)), axis=1)
  
    # mean det
    mean_log_det = -tf.reduce_mean(log_det, name='likeloss_mean_det')

    # get norm(L^T*res) (second part of the likelihood loss)
    # https://stats.stackexchange.com/questions/503058/relationship-between-cholesky-decomposition-and-matrix-inversion
    Lt_res = tf.einsum('ijk,ik->ij', upper_triang, res, name='likeloss_Lt_res')
    Lt_res_norm = tf.reduce_sum(tf.square(Lt_res), axis=1,  name='likeloss_norm_Lt_res')
    mean_Lt_res_norm = tf.reduce_mean(Lt_res_norm)

    # return neg_likelihood
    return mean_Lt_res_norm, mean_log_det


@tf.function()
def tf_neg_likelihood_cholesky_total(mean_norm, mean_log_det):
    """
    """

    neg_likelihood = tf.add(mean_norm, mean_log_det)

    # return neg_likelihood
    return neg_likelihood


@tf.function()
def tf_neg_likelihood_cholesky(predictions, labels, n):
    """
    calculates the negative likelihood loss with tensorflow
    the first n values of predictions are the actual prediction
    from the next elements using a upper triangular matrix is generated
    :param predictions: Predictions of the network
    :param labels: true params
    :param n: number of labels (the rest is used for the covariance matrix)
    :returns: negative log likelihood of the predictions
    """

    mean_norm, mean_log_det = tf_neg_likelihood_cholesky_parts(predictions, labels, n)
    neg_likelihood = tf_neg_likelihood_cholesky_total(mean_norm, mean_log_det)

    # return neg_likelihood
    return neg_likelihood



class LossWrap(tf.keras.layers.Layer):

    def __init__(self, output_select, n_output):

        super(LossWrap, self).__init__()
        self.output_select = output_select
        self.n_output = tf.cast(n_output, tf.int32)
        self.n_chol = tf.cast(n_output * (n_output + 1) / 2, tf.int32)

    def call(self, predictions, y):

        y_select = tf.gather(y, self.output_select, axis=1, name='loss_gather_output')

        # calculate loss
        loss_part1, loss_part2 = tf_neg_likelihood_cholesky_parts(predictions=predictions,
                                                                  labels=y_select,
                                                                  n=self.n_output,
                                                                  m=self.n_chol)

        # combine two loss parts
        loss_total = tf_neg_likelihood_cholesky_total(mean_norm=loss_part1,
                                                       mean_log_det=loss_part2)

        # get the output parameter values and compute rms
        with tf.device('gpu'):
            y_pred = predictions[:,:self.n_output]
            err = tf.reduce_mean((y_pred-y_select)**2, axis=0, name='loss_calculate_err')

        return loss_total, loss_part1, loss_part2, err


