# :triangular_flag_on_post: NumpyNet - Neural Network in Numpy

This is a simple neural network implementation in Numpy. Feel free to look around and see how your neural network operates inside when trained on MNIST! :blush:

## Requirements

Numpy only. I have made the code in 1.20.2, but any version compatible with this should work out fine.

##  Run

```bash
python train.py
```
Parameters
| Arg          | Description                                                   |
| ------------ | ------------------------------------------------------------- |
| --lr         | Learning rate. Default: 0.1                                   |
| --epochs     | Number of epochs. Default: 10                                 |
| --batch-size | Batch size. Default: 100                                      |
| --model      | Type of model. Choose from *dense* or *conv*. Default: *conv* |

## Results

Results of the two models with default settings:

| Model   | Train Accuracy | Val Accuracy |
| ------- | -------------- | ------------ |
| *dense* | 79.82%         | 81.50%       |
| *conv*  | 90.84%         | 91.93%       |

## Notes

 * I used Stochastic Gradient Descent (SGD)
 * Some layers were made but not used eventually (e.g. Dropout)
 * In Conv2D and MaxPool2D pool-size and stride must match.