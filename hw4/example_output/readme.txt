"sys_output" and "feat_list" show the first 50 lines of the expected output files when you run

./build_kNN.sh train.vectors.txt test.vectors.txt 5 2 sys_output > acc_file

cat train.vectors.txt | rank_feat_by_chi_square.sh > feat_list

