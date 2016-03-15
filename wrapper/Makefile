tensor.native: tensor.ml
	ocamlbuild -use-ocamlfind tensor.native -package ctypes -package ctypes.foreign -lflags -cclib,-L../../lib,-cclib,-ltensorflow,-cclib,-lpython2.7

run: tensor.native
	LD_LIBRARY_PATH=../lib:$(LD_LIBRARY_PATH) ./tensor.native
