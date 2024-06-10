#! /usr/bin/awk -f

NR==1 { print; }
NR>1 { if ($2 > 20100101) { print; } }
