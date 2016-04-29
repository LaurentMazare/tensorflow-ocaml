
The examples in this directory have been adapted from the [TensorFlow tutorials](https://www.tensorflow.org/versions/r0.7/tutorials/mnist/pros/index.html). To execute these examples, you will have to unzip the [MNIST data files](http://yann.lecun.com/exdb/mnist/) in `data/`.

## Linear Classifier

The code can be found in `mnist_linear.ml`.

We first load the MNIST data. This is done using the the MNIST helper module, labels
are returned using one-hot encoding.

```ocaml
  let { Mnist_helper.train_images; train_labels; test_images; test_labels } =
    Mnist_helper.read_files ()
  in
```

After that the computation graph is defined. Two placeholders are introduced
to store the input images and labels, these placeholders will be replaced with actual
tensors when calling `Session.run`. Train images and labels are used when training the model.
Test images and labels are used to estimate the validation error.

```ocaml
  let xs = O.placeholder [] ~type_:Float in
  let ys = O.placeholder [] ~type_:Float in
  let train_inputs = Session.Input.[ float xs train_images; float ys train_labels ] in
  let validation_inputs =
    Session.Input.[ float xs test_images; float ys test_labels ]
  in
```
In the linear model there are two variables, a
matrix and a bias and the output is computed by multiplying the matrix by the input
vector and adding the bias. To transform the output into a probability distribution
the softmax function is used.
```ocaml
  let w = Var.f [ image_dim; label_count ] 0. in
  let b = Var.f [ label_count ] 0. in
  let ys_ = O.(Placeholder.to_node xs *^ w + b) |> O.softmax in
```

The error measure that we will try to minimize is cross-entropy. We also compute
the accuracy, i.e. the percentage of images that were correctly labeled, in order
to make the output easier to understand.

```ocaml
  let cross_entropy = O.(neg (reduce_mean (Placeholder.to_node ys * log ys_))) in
  let accuracy =
    O.(equal (argMax ys_ O.one32) (argMax (Placeholder.to_node ys) O.one32))
    |> O.cast ~type_:Float
    |> O.reduce_mean
  in
```

Finally we use gradient descent to minimize cross-entropy with respect to variables
w and b and iterate this a couple hundred times.

```ocaml
  let gd = Optimizers.gradient_descent_minimizer ~learning_rate:(O.f 8.) cross_entropy in
  let print_err n =
    let accuracy =
      Session.run
        ~inputs:validation_inputs
        (Session.Output.scalar_float accuracy)
    in
    printf "epoch %d, accuracy %.2f%%\n%!" n (100. *. accuracy)
  in
  for i = 1 to epochs do
    if i % 50 = 0 then print_err i;
    Session.run
      ~inputs:train_inputs
      ~targets:gd
      Session.Output.empty;
  done
```

Running this code should build a model that has ~92% accuracy.

## A Simple Neural-Network

The code can be found in `mnist_nn.ml`.

## Convolutional Neural-Network

The code can be found in `mnist_conv.ml`.
