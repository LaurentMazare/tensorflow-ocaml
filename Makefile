simple.native: .FORCE
	ocamlbuild examples/simple.native

gen.native: .FORCE
	ocamlbuild gen_ops/gen.native

src/ops.ml: gen.native
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) ./gen.native

run: simple.native
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) ./simple.native

.FORCE:
