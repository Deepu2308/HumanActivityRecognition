# HumanActivityRecognition
Code to recognize activity from smartphone accelerometer and gyro scope data.

## Create a folder 'input' and download files from this URL
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

## Confusion Matrix Test

|      Activity      | WALKING | WALKING_UPSTAIRS | WALKING_DOWNSTAIRS | SITTING | STANDING | LAYING |
|:------------------:|:-------:|:----------------:|:------------------:|:-------:|:--------:|:------:|
|       WALKING      |   466   |         5        |         25         |    0    |     0    |    0   |
|  WALKING_UPSTAIRS  |    0    |        448       |         22         |    1    |     0    |    0   |
| WALKING_DOWNSTAIRS |    0    |         0        |         420        |    0    |     0    |    0   |
|       SITTING      |    0    |         0        |          0         |   416   |    69    |    6   |
|      STANDING      |    0    |         0        |          0         |    27   |    505   |    0   |
|       LAYING       |    0    |         0        |          0         |    0    |     0    |   537  |

## What the data looks like
![Result](src/plots/Subjects%201,3,26,27.png)

## Whats the 1st layer convolution output looks like
![Result](src/plot//../plots/Subjects%201.png)