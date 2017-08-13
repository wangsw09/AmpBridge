eqSolver.so: eqSolver.o
	gcc -Wall -shared eqSolver.o -o eqSolver.so 

eqSolver.o: eqSolver.c eqSolver.h
	gcc -fPIC -c eqSolver.c

cProxOperator.so: cProxOperator.o
	gcc -Wall -shared cProxOperator.o -lm -o cProxOperator.so 

cProxOperator.o: cProxOperator.c cProxOperator.h
	gcc -fPIC -c cProxOperator.c

.PHONY: clean all
clean:
	rm -f *.py~ .*.un~ *.pyc *~ *.o main

all: clean eqSolver.so cProxOperator.so

test:
	printf "\n" >> test_log.txt
	printf "##################################\n" >> test_log.txt
	date >> test_log.txt
	printf "##################################\n" >> test_log.txt
	python ctest.py >> test_log.txt
	printf "\n" >> test_log.txt
