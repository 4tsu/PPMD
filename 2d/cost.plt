set terminal pngcairo enhanced font'font,14'

file0 = "'cost_0.dat'"
file1 = "'cost_1.dat'"
file2 = "'cost_2.dat'"

set output 'calc.png'
plot @file0 using 1:2 title 'simple' w linespoints, @file1 using 1:2 title 'global sort' w linespoints, @file2 using 1:2 title 'voronoi' w linespoints

set output 'comm.png'
plot @file0 using 1:3 title 'simple' w linespoints, @file1 using 1:3 title 'global sort' w linespoints, @file2 using 1:3 title 'voronoi' w linespoints



file3 = "'load_balance_0.dat'"
file4 = "'load_balance_1.dat'"
file5 = "'load_balance_2.dat'"

set output 'load_balance.png'
set key right center
plot @file3 using 1:3 title 'simple' w linespoints, @file4 using 1:3 title 'global sort' w linespoints, @file5 using 1:3 title 'voronoi' w linespoints
