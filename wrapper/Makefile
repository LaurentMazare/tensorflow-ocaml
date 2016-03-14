tensor.native: tensor.ml
	ocamlfind ocamlopt -package ctypes -package ctypes.foreign -linkpkg tensor.ml -cclib -L. -cclib -lstdc++ -cclib -ltensorflow -cclib -lpython2.7 -o tensor.native

run: tensor.native
	LD_LIBRARY_PATH=.:$(LD_LIBRARY_PATH) ./tensor.native
