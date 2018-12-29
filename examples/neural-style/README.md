This contains an ocaml implementation of [Neural Style transfer](https://arxiv.org/abs/1508.06576) using a VGG-19 convolutional network.

## Installation

Use [opam](https://opam.ocaml.org/) to install the tensorflow-ocaml package and the other necessary packages.

```bash
opam install cmdliner npy stb_image stb_image_write tensorflow
```

Install the TensorFlow library by following these [instructions](https://github.com/LaurentMazare/tensorflow-ocaml). 

Compile [neural_style.ml](https://github.com/LaurentMazare/tensorflow-ocaml/tree/master/examples/neural-style/neural_style.ml) and the following command:
```bash
ocamlbuild neural_style.native -pkg tensorflow -pkg cmdliner -pkg npy -pkg stb_image -pkg stb_image_write -use-ocamlfind -tag thread
```

Download the [pre-trained weights](https://github.com/LaurentMazare/tensorflow-ocaml/releases/download/0.0.9/vgg19.npz) for the VGG-19 network.

You can then run the executable neural_style.native. Depending on the input, the loss function may be tuned using the following command line arguments `--content-weight`, `--style-weight`, and `--tv-weight`.

### Example 1: Starry Brooklyn

Original image

#![Brooklyn](https://raw.githubusercontent.com/LaurentMazare/tensorflow-ocaml/master/examples/neural-style/samples/brooklyn.jpg)

Style

#![Starry](https://raw.githubusercontent.com/LaurentMazare/tensorflow-ocaml/master/examples/neural-style/samples/style-starry.jpg)

Result

#![Starry Brooklyn](https://raw.githubusercontent.com/LaurentMazare/tensorflow-ocaml/master/examples/neural-style/samples/brooklyn-starry.jpg)

### Example 2: Cubist New-York

Original image

#![New-York](https://raw.githubusercontent.com/LaurentMazare/tensorflow-ocaml/master/examples/neural-style/samples/new-york.jpg)

Style

#![Cubist](https://raw.githubusercontent.com/LaurentMazare/tensorflow-ocaml/master/examples/neural-style/samples/style-cubist.jpg)

Result

#![Cubist New-York](https://raw.githubusercontent.com/LaurentMazare/tensorflow-ocaml/master/examples/neural-style/samples/new-york-cubist.jpg)

### Example 3: Eschery London

Original image

#![Lond](https://raw.githubusercontent.com/LaurentMazare/tensorflow-ocaml/master/examples/neural-style/samples/london.jpg)

Style

#![Escher](https://raw.githubusercontent.com/LaurentMazare/tensorflow-ocaml/master/examples/neural-style/samples/style-escher.jpg)

Result

#![Eschery London](https://raw.githubusercontent.com/LaurentMazare/tensorflow-ocaml/master/examples/neural-style/samples/london-escher.jpg)
