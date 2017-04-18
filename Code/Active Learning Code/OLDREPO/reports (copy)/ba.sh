output="$(grep -E -o 'Correctly.*' $1 | grep -E -o '[0-9.]+' )"

read -a arr <<<$output

IFS=$'\n'

val="$(echo "${arr[*]}" | sort -nr | head -n1)"
echo "$1 : $val"
