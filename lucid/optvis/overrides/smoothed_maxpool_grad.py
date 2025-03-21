import tensorflow.compat.v1 as tf

def make_smoothed_maxpool_grad(smooth_type="avg"):

  def MaxPoolGrad(op, grad):
    inp = op.inputs[0]

    op_args = [op.get_attr("ksize"), op.get_attr("strides"), op.get_attr("padding")]
    if smooth_type == "L2":
      smooth_out = tf.nn.avg_pool(inp**2, *op_args)/ (1e-2+tf.nn.avg_pool(tf.abs(inp), *op_args))
    elif smooth_type == "avg":
      smooth_out = tf.nn.avg_pool(inp, *op_args)
    else:
      raise RuntimeError("Invalid smooth_type")
    inp_smooth_grad = tf.gradients(smooth_out, [inp], grad)[0]

    return inp_smooth_grad
  return MaxPoolGrad


l2_smoothed_maxpool_grad = make_smoothed_maxpool_grad(smooth_type="L2")
avg_smoothed_maxpool_grad = make_smoothed_maxpool_grad(smooth_type="avg")
