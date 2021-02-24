# Context-Aware Visual Compatibility Prediction

This repository is forked from [visual-compatibility](https://github.com/gcucurull/visual-compatibility).
Learning and inferencing your custom data is implemented.
Check more information in [visual-compatibility](https://github.com/gcucurull/visual-compatibility)

## Requirements

The model is implemented with Tensorflow. All necessary libraries can be installed with:

    pip install -r requirements.txt

## Data

### Polyvore
The [Polyvore dataset](https://github.com/xthan/polyvore-dataset) can be automatically downloaded by running the following script in `data/`:

    ./get_polyvore.sh
    

### Custom

You can train your custom data. Check `README.md` in `data/`

## Training
The model is trained with the following command:

    python train.py -d DATASET 

The most relevant arguments are the following:

 - `-d DATASET`: Choose from `(polyvore, custom)`
 - `-lr LR`: Learning rate. `0.001` by  default.
 - `-hi N N N`: One argument per layer that defines the number of hidden units in that layer. Default: `-hi 350 350 350`
 - `-deg D`: Size of the neighbourhood used around each node. Default `1`.


 Which will store the log and the weights of the model in `logs/`.


 ## Evaluation

Evaluate `test.json`

    python test_compatibility.py -lf PATH_TO_MODEL -k K