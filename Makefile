MNIST = examples/mnist
MNIST_ALL = $(MNIST)/mnist_conv.exe $(MNIST)/mnist_nn.exe $(MNIST)/mnist_linear.exe $(MNIST)/mnist_svm.exe
FNN_ALL =  examples/fnn/mnist_conv_fnn.exe examples/fnn/mnist_linear_fnn.exe examples/fnn/mnist_multi_fnn.exe
BASICS_ALL = examples/basics/forty_two.exe examples/basics/save_and_load.exe examples/basics/linear_regression.exe

ALL = $(MNIST_ALL) \
      $(FNN_ALL) \
      $(BASICS_ALL) \
      examples/char_rnn/char_rnn.exe \
      examples/rnn/rnn.exe \
      examples/neural-style/neural_style.exe \
      examples/neural-style/vgg19.exe \
      tests/operator_tests.exe tests/gradient_tests.exe \
      examples/load/load.exe

%.exe: .FORCE
	jbuilder build --dev $@

src/graph/ops_generated: _build/default/src/gen_ops/gen.exe
	_build/default/src/gen_ops/gen.exe

clean:
	rm -Rf _build/ *.exe

.FORCE:

runtests: tests/operator_tests.exe tests/gradient_tests.exe
	_build/default/tests/operator_tests.exe
	_build/default/tests/gradient_tests.exe

all: .FORCE
	jbuilder build --dev $(ALL)
