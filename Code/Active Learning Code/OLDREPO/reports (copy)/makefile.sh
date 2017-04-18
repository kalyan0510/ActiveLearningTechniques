output="$(grep -E -o 'Weighted FMeasur.*' $1 | grep -E -o '[0-9.]+' )"

echo $output > blas/"$1"
