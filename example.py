import lucid_video as L
import tensorflow.compat.v1 as tf


def main():
    transforms = [L.normalize_gradient_by_std()]
    optimizer = tf.train.GradientDescentOptimizer(1)
    param = lambda: L.uniform_video(79, 224)
    obj = "inceptioni3d/Mixed_5b/concat:120"
    thresholds = list(range(10, 1000, 100))
    model = L.I3D()
    model.load_graphdef()
    L.render_vis(model, obj, thresholds=thresholds,
                 param_f=param, transforms=transforms, optimizer=optimizer)


if __name__ == "__main__":
    main()
