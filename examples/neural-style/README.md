## Installation

Use [opam](https://opam.ocaml.org/) to install the tensorflow-ocaml package and the other necessary packages.

```bash
opam install cmdliner npy camlimages tensorflow
```

Install the TensorFlow library by following these [instructions](https://github.com/LaurentMazare/tensorflow-ocaml). 

Compile [neural_style.ml](https://github.com/LaurentMazare/tensorflow-ocaml/tree/master/examples/neural-style/neural_style.ml) and the following command:
```bash
ocamlbuild neural_style.native -pkg tensorflow -pkg cmdliner -pkg npy -pkg camlimages.jpeg -pkg camlimages.png -use-ocamlfind
```

You can then run the executable neural_style.native.
