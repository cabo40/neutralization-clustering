#! /usr/bin/awk -f

NR==1 { print;x=$1;xx=$2 }
NR>1 {
    if ($1 != x || $2 != xx) {
        print;x=$1;xx=$2
    }
}