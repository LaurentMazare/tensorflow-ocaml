EXAMPLES = simple.native var.native load.native gradient.native linear_regression.native
%.native: .FORCE
	ocamlbuild examples/$@

gen.native: .FORCE
	ocamlbuild src/gen_ops/gen.native

src/graph/ops: gen.native
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) ./gen.native

run: $(EXAMPLES)
	$(foreach ex,$(EXAMPLES),LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) ./$(ex);)

clean:
	rm -Rf _build/

.FORCE:
