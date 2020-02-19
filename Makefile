.PHONY: clean

clean:
	rm -rf AmpBridge/cscalar/*.c AmpBridge/cscalar/*.so AmpBridge/cscalar/*~ AmpBridge/cscalar/__pycache__ AmpBridge/cscalar/.*~ AmpBridge/cscalar/*.pyc AmpBridge/coptimization/*.c AmpBridge/coptimization/*.so AmpBridge/coptimization/*~ AmpBridge/coptimization/.*~ AmpBridge/coptimization/*.pyc AmpBridge/__pycache__ build

compile:
	python setup.py build_ext --inplace

post_clean:
	rm -rf build

all: clean compile post_clean
