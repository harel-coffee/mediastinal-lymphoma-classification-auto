python multiclass_classification.py -m log-reg -s 60/20/20 -t rec70 -J 24;
python multiclass_classification.py -m log-reg -s 60/20/20 -t rec90 -J 24;

python multiclass_classification.py -m lin-svm -s 60/20/20 -t rec70 -J 24;
python multiclass_classification.py -m lin-svm -s 60/20/20 -t rec90 -J 24;

python multiclass_classification.py -m gaus-proc -s 60/20/20 -t rec70 -J 24;
python multiclass_classification.py -m gaus-proc -s 60/20/20 -t rec90 -J 24;

python multiclass_classification.py -m rnd-frs -s 60/20/20 -t rec70 -J 24;
python multiclass_classification.py -m rnd-frs -s 60/20/20 -t rec90 -J 24;

python multiclass_classification.py -m grad-bdt -s 60/20/20 -t rec70 -J 24;
python multiclass_classification.py -m grad-bdt -s 60/20/20 -t rec90 -J 24;

python multiclass_classification.py -m suv-max -s 60/20/20 -t rec70 -J 24;
python multiclass_classification.py -m suv-max -s 60/20/20 -t rec90 -J 24;

python multiclass_pr_roc_curves.py -t rec70;
python multiclass_pr_roc_curves.py -t rec90;
