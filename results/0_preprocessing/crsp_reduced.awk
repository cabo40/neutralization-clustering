#! /usr/bin/awk -f
function abs(v) {if (v=="") return v; return v < 0 ? -v : v}
BEGIN {OFS=","}
{
    if ($6 == -1) {
        $7 = 0; $8 = 0; $9 = -1; $10 = 0;
    }
    if ($9 == "B" && $6 != "") {
        $7 = 0; $8 = 0; $9 = -1; $10 = 0;
    }
    if ($9 != "B" && $9 != "C") {
        sub(/^Z/, "0", $3);
        sub(/^Z/, "0", $5);
        $7 = abs($7);
        print;
    }
}