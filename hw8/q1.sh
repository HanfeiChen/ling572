# Usage: svm-train [options] training_set_file [model_file]
# options:
# -s svm_type : set type of SVM (default 0)
# 	0 -- C-SVC
# 	1 -- nu-SVC
# 	2 -- one-class SVM
# 	3 -- epsilon-SVR
# 	4 -- nu-SVR
# -t kernel_type : set type of kernel function (default 2)
# 	0 -- linear: u'*v
# 	1 -- polynomial: (gamma*u'*v + coef0)^degree
# 	2 -- radial basis function: exp(-gamma*|u-v|^2)
# 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
# 	4 -- precomputed kernel (kernel values in training_set_file)
# -d degree : set degree in kernel function (default 3)
# -g gamma : set gamma in kernel function (default 1/k)
# -r coef0 : set coef0 in kernel function (default 0)
# -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
# -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
# -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
# -m cachesize : set cache memory size in MB (default 100)
# -e epsilon : set tolerance of termination criterion (default 0.001)
# -h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
# -b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
# -wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)
# -v n: n-fold cross validation mode

# model 1
svm-train -t 0 examples/train model.1
svm-predict examples/train model.1 output.train.1
svm-predict examples/test model.1 output.test.1

# model 2
svm-train -t 1 -g 1 -r 0 -d 2 examples/train model.2
svm-predict examples/train model.2 output.train.2
svm-predict examples/test model.2 output.test.2

# model 3
svm-train -t 1 -g 0.1 -r 0.5 -d 2 examples/train model.3
svm-predict examples/train model.3 output.train.3
svm-predict examples/test model.3 output.test.3

# model 4
svm-train -t 2 -g 0.5 examples/train model.4
svm-predict examples/train model.4 output.train.4
svm-predict examples/test model.4 output.test.4

# model 5
svm-train -t 3 -g 0.5 -r -0.2 examples/train model.5
svm-predict examples/train model.5 output.train.5
svm-predict examples/test model.5 output.test.5
