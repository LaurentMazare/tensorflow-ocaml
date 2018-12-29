MNIST = examples/mnist
MNIST_ALL = $(MNIST)/mnist_conv.exe $(MNIST)/mnist_nn.exe $(MNIST)/mnist_linear.exe $(MNIST)/mnist_svm.exe $(MNIST)/mnist_resnet.exe
GAN_ALL = examples/gan/mnist_gan.exe examples/gan/mnist_cgan.exe examples/gan/mnist_dcgan.exe
FNN_ALL =  examples/fnn/mnist_conv_fnn.exe examples/fnn/mnist_linear_fnn.exe examples/fnn/mnist_multi_fnn.exe
BASICS_ALL = examples/basics/forty_two.exe examples/basics/save_and_load.exe examples/basics/linear_regression.exe
NS_ALL = examples/neural-style/neural_style.exe examples/neural-style/vgg19.exe

ALL = $(MNIST_ALL) \
      $(GAN_ALL) \
      $(FNN_ALL) \
      $(BASICS_ALL) \
      $(NS_ALL) \
      examples/char_rnn/char_rnn.exe \
      examples/rnn/rnn.exe \
      examples/load/load.exe

%.exe: .FORCE
	dune build $@

src/graph/ops_generated: _build/default/src/gen_ops/gen.exe
	_build/default/src/gen_ops/gen.exe

utop: .FORCE
	dune build @install
	dune build bin/utop_tensorflow.bc
	dune exec bin/utop_tensorflow.bc

jupyter: .FORCE
	dune build @install
	dune exec jupyter lab

clean:
	rm -Rf _build/ *.exe

.FORCE:

test: .FORCE
	dune runtest

all: .FORCE
	dune build $(ALL)
