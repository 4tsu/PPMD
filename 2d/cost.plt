set terminal pngcairo enhanced font'font,14'

file0 = "'cost_0.dat'"
file1 = "'cost_1.dat'"
file2 = "'cost_2.dat'"

set output 'calc.png'
plot @file0 using 1:2 title 'simple' w l, @file1 using 1:2 title 'global sort' w l, @file2 using 1:2 title 'voronoi' w l

set output 'comm.png'
plot @file0 using 1:3 title 'simple' w l, @file1 using 1:3 title 'global sort' w l, @file2 using 1:3 title 'voronoi' w l
