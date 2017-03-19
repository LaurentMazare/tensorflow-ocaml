MNIST = examples/mnist
MNIST_ALL = $(MNIST)/mnist_conv.exe $(MNIST)/mnist_nn.exe $(MNIST)/mnist_linear.exe $(MNIST)/mnist_svm.exe
FNN_ALL =  examples/fnn/mnist_conv_fnn.exe examples/fnn/mnist_linear_fnn.exe examples/fnn/mnist_multi_fnn.exe
ALL = $(MNIST_ALL) \
      $(FNN_ALL) \
      examples/char_rnn/char_rnn.exe \
      examples/rnn/rnn.exe \
      examples/neural-style/neural_style.exe \
      tests/operator_tests.exe tests/gradient_tests.exe \
      examples/load/load.exe

%.exe: .FORCE
	jbuilder build $@

src/graph/ops_generated: _build/default/src/gen_ops/gen.exe
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) _build/default/src/gen_ops/gen.exe

clean:
	rm -Rf _build/ *.exe

.FORCE:

runtests: tests/operator_tests.exe tests/gradient_tests.exe
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) _build/default/tests/operator_tests.exe
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) _build/default/tests/gradient_tests.exe

all: $(ALL)
