test: step.ml
#  -package  coq-core
	ocamlfind ocamlopt -thread -linkpkg \
	-package  coq-core \
	-package  coq \
	-package coq-serapi  \
	-package coq-serapi.serlib \
	-package coq-serapi.sertop_v8_12 \
	-g -fPIC -linkall -output-obj step.ml -o ocaml-example-script.o
