.PHONY: clean test

clean:
	rm -f *.py~ .*.un~ *.pyc *~ .*~

test:
	printf "\n" >> test_log.txt
	printf "##################################\n" >> test_log.txt
	date >> test_log.txt
	printf "##################################\n" >> test_log.txt
	python main.py --test all >> test_log.txt 2>&1
	printf "\n" >> test_log.txt
