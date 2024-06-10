#! /usr/bin/awk -f
function abs(v) {if (v=="") return v; return v < 0 ? -v : v}
BEGIN {OFS=","}
{
    if ($3 == "" || $6 != abs($6) || $4 == "" || $8 == "" || $8 == "C") {
        next
    }
    if ($5 == -1) {
        $6 = 0; $7 = 0; $8 = -1; $9 = 0;
    }
    if ($8 == "B") {
        $6 = 0; $7 = 0; $8 = 0; $9 = 0;
    }
    sub(/^Z/, "-1", $4);
    print;
}