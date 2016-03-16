example.native: .FORCE
	ocamlbuild examples/example.native
run: example.native
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) ./example.native

.FORCE:
