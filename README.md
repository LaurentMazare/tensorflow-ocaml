The tensorflow-ocaml project provides some [ocaml](http://ocaml.org) bindings for [TensorFlow](http://tensorflow.org).

These bindings are in a very early stage of their development and are not ready for real-world usage. Expect some segfaults if you use them.

## Installation

* Install the dependencies `opam install ocamlbuild ctypes ctypes-foreign`.
* Clone the repo: `git clone https://github.com/LaurentMazare/tensorflow-ocaml.git`.
* Install [TensorFlow](http://tensorflow.org).
* Copy the TensorFlow shared library to `lib/libtensorflow.so` in the cloned repo. The following command may work: `cp ~/.local/lib/python2.7/site-packages/tensorflow/python/_pywrap_tensorflow.so lib/libtensorflow.so`

## Dependencies

* [ocaml-ctypes](https://github.com/ocamllabs/ocaml-ctypes) is used for the C bindings.
* [Core](https://github.com/janestreet/core) is only necessary for the operator code generation.
* The code in the piqi directory comes from the [Piqi project](http://piqi.org). There is no need to install piqi though.
