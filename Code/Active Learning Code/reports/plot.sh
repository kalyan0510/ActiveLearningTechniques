x=0;
for file in *.txt; do let "x += 1";./makefile.sh "$file" "$x"; done

python y.py;


