example.native: .FORCE
	ocamlbuild -use-ocamlfind example.native -package ctypes -package ctypes.foreign -lflags -cclib,-L../../lib,-cclib,-ltensorflow,-cclib,-lpython2.7

run: example.native
	LD_LIBRARY_PATH=../lib:$(LD_LIBRARY_PATH) ./example.native

.FORCE:
