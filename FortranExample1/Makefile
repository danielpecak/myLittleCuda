all: bin/saxpy bin/addvec

bin/saxpy: saxpy.cuf
	pgf90 -o bin/saxpy saxpy.cuf

bin/addvec: addVec.cuf
	pgf90 -o bin/addvec addVec.cuf
