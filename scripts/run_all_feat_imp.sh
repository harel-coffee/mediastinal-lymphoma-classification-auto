python binary_classification.py -m log-reg -s 60/20/20 -t rec70 -f yes -J 24;
python binary_classification.py -m log-reg -s 60/20/20 -t rec90 -f yes -J 24;

python binary_classification.py -m lin-svm -s 60/20/20 -t rec70 -f yes -J 24;
python binary_classification.py -m lin-svm -s 60/20/20 -t rec90 -f yes -J 24;

python binary_classification.py -m gaus-proc -s 60/20/20 -t rec70 -f yes -J 24;
python binary_classification.py -m gaus-proc -s 60/20/20 -t rec90 -f yes -J 24;

python binary_classification.py -m rnd-frs -s 60/20/20 -t rec70 -f yes -J 24;
python binary_classification.py -m rnd-frs -s 60/20/20 -t rec90 -f yes -J 24;

python binary_classification.py -m grad-bdt -s 60/20/20 -t rec70 -f yes -J 24;
python binary_classification.py -m grad-bdt -s 60/20/20 -t rec90 -f yes -J 24;