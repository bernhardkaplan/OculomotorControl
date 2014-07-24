
wa_d1_d1_pos=0.0
wa_d1_d1_neg=1.0

#for (($(wa_d1=0.2);  $wa_d1 < $(0.7); $wa_d1 = $($wa_d1 + 0.1)))
for wa_d1_d1_neg in 1.0 5.0 10.0
do
    for wa_d1 in 0.1 0.2 0.3 0.4 0.5 0.6
    do
        for wa_d2 in 0.1 0.2 0.3 0.4 0.5 0.6
        do
            echo "submitting job" $n " with parameter x =" $wa_d1 $wa_d2 $wa_d1_d1_pos $wa_d1_d1_neg
            sbatch jobfile_testing_milner_with_params.sh $wa_d1 $wa_d2 $wa_d1_d1_pos $wa_d1_d1_neg
            sleep 0.2
        done
    done
done
