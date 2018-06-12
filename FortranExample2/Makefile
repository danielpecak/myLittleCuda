all: bin/addvec bin/addvec_pinned

bin/%: %.cuf mod10mat.o
	pgf90 -o $@ $^ -mcmodel=medium

%.o: %.cuf
	pgf90 -c $^ -mcmodel=medium

clear:
	rm *.o *.mod bin/*
