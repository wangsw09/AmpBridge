.PHONY: clean

clean:
	rm -rf *.c build *.so __init__.py *~ .*~ __init__.pyc

compile:
	python setup.py build_ext --inplace

init.py:
	echo "from clib import *\nimport gaussian\nimport proximal" > __init__.py

all: clean compile init.py
