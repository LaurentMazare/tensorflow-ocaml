example.native: .FORCE
	ocamlbuild examples/example.native

example_ops.native: .FORCE
	ocamlbuild examples/example_ops.native

gen.native: .FORCE
	ocamlbuild gen_ops/gen.native

src/ops.ml: gen.native
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) ./gen.native

run: example.native example_ops.native
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) ./example.native
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) ./example_ops.native

.FORCE:
