# Neural Painter : A smart image manipulator based on simple line-drawings

Visual language is an important part of human communication, however most of people are not skilled at expressing themselves visually. This project aims to bridge the expertise gap of drawing by using interactive generation techniques.

# Demo: Anime face editing

Online demo is now unavailable, but there are demo videos in the project blog ![page](https://atlantixjj.github.io/jekyll/update/2018/09/01/NeuralPainter.html).

## Running

Note that the most recent training is in branch `clean`, make sure you checkout to this branch before training any model.

```bash
# train 64x64 getchu
python gan_tf.py --gpu 0 --model_name simple --train_dir logs/simple_getchu1 
```

Please use the repo according to GNU license.

Jianjin Xu,
2018, Future Internet & Technology, Tsinghua University.
