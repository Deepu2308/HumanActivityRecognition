# HumanActivityRecognition
Code to recognize activity from smartphone accelerometer and gyro scope data.

## Create a folder 'input' and download files from this URL
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

## Confusion Matrix Test

|      Activity      | WALKING | WALKING_UPSTAIRS | WALKING_DOWNSTAIRS | SITTING | STANDING | LAYING |
|:------------------:|:-------:|:----------------:|:------------------:|:-------:|:--------:|:------:|
|       WALKING      |   491   |         5        |          0         |    0    |     0    |    0   |
|  WALKING_UPSTAIRS  |    52   |        405       |         14         |    0    |     0    |    0   |
| WALKING_DOWNSTAIRS |    75   |         6        |         339        |    0    |     0    |    0   |
|       SITTING      |    0    |         4        |          0         |   258   |    229   |    0   |
|      STANDING      |    0    |        13        |          0         |    22   |    497   |    0   |
|       LAYING       |    0    |         0        |          0         |    0    |     0    |   537  |