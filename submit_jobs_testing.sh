
wa_d1_d1_pos=0.0
#wa_d1_d1_neg=1.0

#for (($(wa_d1=0.2);  $wa_d1 < $(0.7); $wa_d1 = $($wa_d1 + 0.1)))

#w_to_action_inh=-8
#w_to_action_exc=8.

for w_to_action_inh in -6. -8. # weight from strD2 to action layer
do
    for w_to_action_exc in 6. 8. # weight from striatum D1 to action
    do
        #for wa_d1_d1_neg in 1.0 5.0 10. 20. # weight amplification d1 -> d1 negative weights
        for wa_d1_d1_neg in 1. 2. # weight amplification d1 -> d1 negative weights
        do
            for wa_d1 in 0.5 1.0 1.5 2.0 4.0 
            do
                for wa_d2 in 0.5 1.0 1.5 2.0 4.0 
                do
                    #wa_d2=$wa_d1
                    echo "submitting job" $n " with parameter x =" $wa_d1 $wa_d2 $wa_d1_d1_pos $wa_d1_d1_neg $w_to_action_exc $w_to_action_inh
                    sbatch jobfile_testing_milner_with_params.sh $wa_d1 $wa_d2 $wa_d1_d1_pos $wa_d1_d1_neg $w_to_action_exc $w_to_action_inh
                    sleep 0.5
                done
            done
        done
    done
done

#echo 'Submitting jobfile_remove_empty_files.sh ...'
#sleep 1
#sbatch jobfile_remove_empty_files.sh
