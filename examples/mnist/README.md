
The examples in this directory have been adapted from the [TensorFlow tutorials](https://www.tensorflow.org/versions/r0.7/tutorials/mnist/pros/index.html). To execute these examples, you will have to unzip the [MNIST data files](http://yann.lecun.com/exdb/mnist/) in `data/`.

## Linear Classifier

The code can be found in `mnist_linear.ml`.

We first load the MNIST data. This is done using the the MNIST helper module, labels
are returned using one-hot encoding.

```ocaml
  let mnist = Mnist_helper.read_files () in
```

After that the computation graph is defined. Two placeholders are introduced
to store the input images and labels, these placeholders will be replaced with actual
tensors when calling `Session.run`. Train images and labels are used when
training the model.  Test images and labels are used to estimate the validation
error.

```ocaml
  let xs = O.placeholder [-1; image_dim] ~type_:Float in
  let ys = O.placeholder [-1; label_count] ~type_:Float in
```
In the linear model there are two variables, a
matrix and a bias and the output is computed by multiplying the matrix by the input
vector and adding the bias. To transform the output into a probability distribution
the softmax function is used.
This is all handled by the linear layer.
```ocaml
  let ys_ = Layer.linear (O.Placeholder.to_node xs) ~activation:Softmax ~output_dim:label_count in
```

The error measure that we will try to minimize is cross-entropy.

```ocaml
  let cross_entropy = O.cross_entropy ~ys:(O.Placeholder.to_node ys) ~y_hats:ys_ `mean in
```

Finally we use gradient descent to minimize cross-entropy with respect to variables
w and b and iterate this a couple hundred times.

```ocaml
  let gd = Optimizers.gradient_descent_minimizer ~learning_rate:(O.f 8.) cross_entropy in
  let print_err n =
    let accuracy =
      Mnist_helper.batch_accuracy mnist `test ~batch_size:1024 ~predict:(fun images ->
        Session.run (Session.Output.float ys_)
          ~inputs:Session.Input.[ float xs images ])
    in
    Stdio.printf "epoch %d, accuracy %.2f%%\n%!" n (100. *. accuracy)
  in
  for i = 1 to epochs do
    if i % 50 = 0 then print_err i;
    Session.run
      ~inputs:Session.Input.[ float xs mnist.train_images; float ys mnist.train_labels ]
      ~targets:gd
      Session.Output.empty;
  done
```

Running this code should build a model that has ~92% accuracy.

## A Simple Neural-Network

The code can be found in `mnist_nn.ml`, accuracy should reach ~96%.

## Convolutional Neural-Network

The code can be found in `mnist_conv.ml`, accuracy should reach ~99%.

## ResNet

A [Residual Neural Network](https://arxiv.org/abs/1512.03385) is trained
on the MNIST dataset in `mnist_resnet.ml`. Final accuracy should be ~99.3%.
