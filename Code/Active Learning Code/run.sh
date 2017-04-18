
for file in z/HoldOut/*.cfg; do  echo "doing $file\n"; java -jar jclal-1.1.jar -cfg "$file" > "$file.sim" ; echo "$file done"; done

