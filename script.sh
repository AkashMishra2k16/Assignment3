# !/bin/bash
java -cp "/home1/e1-246-32/Assignment3/mallet-2.0.8/class:/home1/e1-246-32/Assignment3/mallet-2.0.8/lib/mallet-deps.jar" cc.mallet.fst.SimpleTagger --train true --model-file model_file --training-proportion 0.8 --test lab --threads 2 Generated_feature_Ner.txt
