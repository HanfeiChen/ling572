The command lines to obtain the files (first 50 lines)  under this folder are

classifier2info --classifier q1/m1 > q1/m1.txt

./maxent_classify.sh test2.vectors.txt q1/m1.txt q2/res > q2/acc

./calc_emp_exp.sh train2.vectors.txt q3/emp_count

./calc_model_exp.sh train2.vectors.txt q4/model_count q1/m1.txt

./calc_model exp.sh train2.vectors.txt q4/model_count2


