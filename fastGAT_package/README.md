# Fast Graph ATtention neural networks
## What's that?
Fast Graph ATtention neural networks(fastGAT) is a new algorithm based on GAT(graph attention networks). We designed our algorithm with AngleLSH(Angle based Local Sensity Hashing), for a faster running speed and a less memory usage. 
In a word, you can use our algorithm instead of GAT in everywhere with more great effect.
## How to install it?
Just use:
```
pip install fastGAT
```

## How to use it?
### A demo for fast run.
you can just use:
```python
import fastGAT.fastGAT as ft

proc=ft.exe()
proc.tra_val()
proc.test()
```
for simple demo, which will running our method in CORA dataset.

### customize your own hyperparameters.
use
```python
from fasGAT.model import fastGAT as mft

blablabla..

```
### customize your own dataset
just try the numpy format matrix , which for example： 
```
adj: the adj matrix
features: feature vector combination of all nodes
labels: label of all nodes
```

### MORE INFORMATION
For more infomation, you might go to here:[fastGAT](https://github.com/liangzid/FastGraphATtention) 
## Does it well?
You can enjoy some images follow for the detail of it.


## concat me.
you can send any problem and BUG using the issue in github.  Or call me with: 2273067585@qq.com.






