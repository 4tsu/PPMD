set terminal pngcairo enhanced font'font,14'

file0 = "'cost_0.dat'"
file1 = "'cost_1.dat'"
file2 = "'cost_2.dat'"

set output 'calc.png'
set format y "%.1t{/Symbol \26410^{%T}"
set key right bottom
set xlabel 'step'
set ylabel 'execution-time per step [s]' offset -0.5, 0.0
set yrange [0:]
plot @file0 using 1:2 title 'without load-balancer' w linespoints, @file1 using 1:2 title 'global sort' w linespoints, @file2 using 1:2 title 'voronoi' w linespoints
plot @file0 using 1:2:4 title '' w e, @file1 using 1:2:4 title '' w e, @file2 using 1:2:4 title '' w e

set output 'comm.png'
set xlabel 'step'
set ylabel 'communication-cost per step [byte]'
set key left center
set yrange [0:]
plot @file0 using 1:3 title 'without load-balancer' w linespoints, @file1 using 1:3 title 'global sort' w linespoints, @file2 using 1:3 title 'voronoi' w linespoints



file3 = "'load_balance_0.dat'"
file4 = "'load_balance_1.dat'"
file5 = "'load_balance_2.dat'"

set output 'load_balance.png'
set xlabel 'step'
set ylabel 'workload per step'
unset format y
set yrange [0:]
set key left center
plot @file3 using 1:3 title 'without load-balancer' w linespoints, @file4 using 1:3 title 'global sort' w linespoints, @file5 using 1:3 title 'voronoi' w linespoints
