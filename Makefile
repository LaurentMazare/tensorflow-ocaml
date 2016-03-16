example.native: .FORCE
	ocamlbuild examples/example.native
gen.native: .FORCE
	ocamlbuild gen_ops/gen.native
gen: gen.native
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) ./gen.native

run: example.native
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) ./example.native

.FORCE:
