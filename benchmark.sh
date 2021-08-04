n=1
while [ $n -lt 16 ]
do
	echo "Running with $n threads" >> result.txt
	julia --threads $n --project=. examples/main.jl >> result.txt
	n=$(( $n * 2 ))
done
