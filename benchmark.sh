n=1
while [ $n -lt 1024 ]
do
	echo "Running with $n threads"
	julia --threads $n --project=. examples/main.jl >> result.txt
	n=$(( $n * 2 ))
done
