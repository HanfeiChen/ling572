vectors2classify \
    --training-file examples/train2.vectors \
    --testing-file examples/test2.vectors \
    --trainer MaxEnt \
    --report test:raw test:accuracy test:confusion \
             train:confusion train:accuracy \
    --output-classifier q1/m1 \
    > q1.stdout 2> q1.stderr
classifier2info --classifier q1/m1 > q1/m1.txt
