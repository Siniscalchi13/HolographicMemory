# Minimal build harness for Coq proofs under _CoqProject

COQMF=CoqMakefile

all: $(COQMF)
	$(MAKE) -f $(COQMF) all

$(COQMF): _CoqProject
	coq_makefile -f _CoqProject -o $(COQMF)

clean:
	@if [ -f $(COQMF) ]; then $(MAKE) -f $(COQMF) clean; fi
	@rm -f $(COQMF)

.PHONY: all clean

