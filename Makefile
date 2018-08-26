.PHONY: clean

clean:
	rm -rf AmpBridge/cscalar/*.c build AmpBridge/cscalar/*.so AmpBridge/cscalar/*~ AmpBridge/cscalar/.*~ AmpBridge/cscalar/*.pyc

compile:
	python setup.py build_ext --inplace

post_clean:
	rm -rf build

all: clean compile post_clean
