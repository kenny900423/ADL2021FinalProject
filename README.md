# ADL2021FinalProject

## Dependencies
You will need to install `transformers`, `datasets` and `nltk` to run below codes.

## Download Models
Before predicting, you should run below commands first to download the models.
```
bash download.sh
```

## DST Task 

#### train
```
python T5DST/train_process.py --data_dir=data_dir --save_dir=save_dir --schema_path=schema_path
python T5DST/train.py --save_dir=save_dir
```
#### predict
```
python T5DST/test_process.py --data_dir=data_dir --save_dir=save_dir --schema_path=schema_path
python T5DST/predict.py --save_dir=save_dir
```

## NLG Task

#### Train
```
python BERTNLG/train.py --data_dir=data_dir
```

#### Predict
```
python BERTNLG/predict.py --data_dir=data_dir --model_path='./BERTNLG/model' --output_path=output_json_path
```