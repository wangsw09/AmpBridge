.PHONY: clean

clean:
	rm -rf AmpBridge/cplib/*.c build AmpBridge/cplib/*.so AmpBridge/cplib/*~ AmpBridge/cplib/.*~ AmpBridge/cplib/*.pyc

compile:
	python setup.py build_ext --inplace

post_clean:
	rm -rf build

all: clean compile post_clean
