# Neural Painter : A smart image manipulator based on simple line-drawings

Visual language is an important part of human communication, however most of people are not skilled at expressing themselves visually. This project aims to bridge the expertise gap of drawing by using interactive generation techniques.

## Running

Note that the most recent training is in branch `clean`, make sure you checkout to this branch before training any model.

```bash
# train 64x64 getchu
python gan_tf.py --gpu 0 --model_name simple --train_dir logs/simple_getchu1 
```

Jianjin Xu,
2018, Future Internet & Technology, Tsinghua University.
