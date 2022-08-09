set terminal pngcairo enhanced font'font,14'
set output 'energy.png'


file = "'e2.dat'"

plot @file using 1:2 title 'kinetic', @file using 1:3 title 'potential', @file using 1:4 title 'total' w lines

