FILES=cost_0.dat cost_1.dat cost_2.dat
FILES += load_balance_0.dat load_balance_1.dat load_balance_2.dat

costfig: $(FILES)
	python3 calc_average.py
	gnuplot cost.plt

enefig: e0.dat
	gnuplot plot.plt

clean:
	-rm *.cdv
	-rm *.dat
	-rm *.bmp
