MNIST = examples/mnist
MNIST_ALL = $(MNIST)/mnist_conv.native $(MNIST)/mnist_nn.native $(MNIST)/mnist_linear.native
ALL = gen.native examples/load/load.native examples/basics/linear_regression.native $(MNIST_ALL)

%.native: .FORCE
	ocamlbuild $@

gen.native: .FORCE
	ocamlbuild src/gen_ops/gen.native

src/graph/ops_generated: gen.native
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) ./gen.native

load: examples/load/load.native
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) ./load.native

lr_gnuplot: examples/basics/linear_regression_gnuplot.native
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) ./linear_regression_gnuplot.native

nn_gnuplot: examples/basics/nn_gnuplot.native
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) ./nn_gnuplot.native

mnist_nn: examples/mnist/mnist_nn.native
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) ./mnist_nn.native

mnist_linear: examples/mnist/mnist_linear.native
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) ./mnist_linear.native

mnist_conv: examples/mnist/mnist_conv.native
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) ./mnist_conv.native

mnist_svm: examples/mnist/mnist_svm.native
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) ./mnist_svm.native

clean:
	rm -Rf _build/

.FORCE:

all: $(ALL)
