all: bin/addvec

bin/addvec: addVec.cuf mod10mat.o
	pgf90 -o bin/addvec addVec.cuf mod10mat.o

%.o: %.cuf
	pgf -c $^


