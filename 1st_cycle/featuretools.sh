python pre_main_ft.py --amt_method linear_reg --normal_method most --nlp_method lda --vec_size 4
# python feature_creation.py --func Sum Average Min Max --types continuos --period 7d 15d
# python feature_selection.py --scaler robust
# python train_test_split.py --train_path 'train.csv' --test_path 'test.csv' --fold 5
# python train_main.py --savepath 'result' --fold_num 0 --data_path 'train.csv' --val_path 'test.csv' --model dnn_1 --sample_type both --over_method adasyn --under_method tomek --scaler robust --nb_class 15 --opts radam --lossfunc sparse_categorical_crossentropy --batch_size 64 --learning_rate 0.0001 --callback_type checkpoint earlystopping tensorboard rateschedule interval_check --epoch 100 --jobs 4