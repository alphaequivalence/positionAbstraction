# Reduction of the Position Bias via Multi-Level Learning for Activity Recognition

Code accompanying the paper

> [Reduction of the Position Bias via Multi-Level Learning for Activity Recognition](/docs/412.pdf)

The relative position of sensors on the human body generates two types of data: one relating to the movement of the position w.r.t. the body and another relating to the movement of the whole body w.r.t. the environment. These two data give two complementary and orthogonal components contributing differently to the activity recognition process. In this paper, we introduce an original approach that allows us to separate these data and to abstract away the exact on-body position of the sensors from the considered activities. We learn for these two totally orthogonal components (i) the bias that stems from the position and (ii) the actual patterns of the considered activities rid of these positional biases. We perform a thorough empirical evaluation of our approach on the SHL dataset featuring an on-body sensor deployment in a real-life setting. Obtained results show that we are able to substantially improve recognition performances. These results pave the way for the development of models that are agnostic to both the position of the data generators and the target users. Constructed models are found to be robust to evolving environments such as those we are confronted with in Internet of Things applications.



<p align="center">
    <img src="/figures/PositionAbstraction_componentsDecomposition.png" width="80%">
</p>
<p align="center">
Figure: The hand sensor undergoes two movements. The first is of the same nature as the torso sensor. It is linked to the translational movement of the body. The second is linked to the movement of the hand locally in relation to the body.
</p>

<p align="center">
    <img src="/figures/PositionAbstraction_conciliationInWeightSpace.png" width="80%">
</p>
<p align="center">
Figure: Framework of the proposed multi-level abstraction architecture.
The global learner $L_{\mathcal{S}}$ starts with an initial set of weights which are distributed to the local learners.
The local learners $L_p$, one for each position $p$, learn the two vector components $z_{A}$ and $z_{P}$, by performing independently a set of gradient steps which allows to get a newer version. These new versions are used during the conciliation step which gives us a new version of the global learner, and subsequently a more robust position-independent representation.
</p>


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

### Prerequisites
* `numpy`
* `TensorFlow`
* `nni` (https://github.com/microsoft/nni#installation)


If you are using `pip` package manager, you can simply install all requirements via the following command(s):

    python -m virtualenv .env -p python3 [optional]
    source .env/bin/activate [optional]
    pip3 install -r requirements.txt

### Installing
#### Get the dataset
1. You can get the preview of the SHL dataset (`zip` files) from [here](http://www.shl-dataset.org/activity-recognition-challenge-2020/). Make sure to put the downloaded files into `./data/` folder. Alternatively, you can run `scripts/get_data.sh` to download the dataset automatically.
2. Run `./scripts/extract_data.sh` script which will extract the dataset into `./generated/tmp/` folder.

## Running
### on your laptop:

    python cnn_split_channels.py

this will load the data and train the model defined inside `cnn_split_channels.py`. This same way, you can run other models.

### via Microsoft NNI

     nnictl create --config stjohns.yml --foreground

### on the computing platform Magi:

    /opt/slurm-19.05.1-2/bin/sbatch stjohns.slurm

information about the execution of the job can be found in `*.err` and `*.out` files, which output, respectively, messages sent to stderr and stdout streams.

## Results
Our model achieves the following performance:

<p align="center">
    <img src="/figures/per-position-recognition-performances.png" width="80%">
</p>
<p align="center">
Table: Summary of the recognition performances obtained using either the universal or the position-specific components learned in each position by the local learners. Recognition performances with and without the conciliation process are reported. For reference, the recognition of a baseline model which do not perform separation (nor conciliation) are additionally shown.
</p>


<p align="center">
    <img src="/figures/confusion-matrices_inferenceConfigurations.png" width="80%">
</p>
<p align="center">
Figure: Confusion matrices obtained using different inference configurations. Combination of the universal components $z_{A}$ and: (a) *Torso*-specific components; (b) *Hand*-specific components; (c) *Bag*-specific components; (d) *Hips*-specific components.  The activities are numbered as *1:Still*, *2:Walk*, *3:Run*, *4:Bike*, *5:Car*, *6:Bus*, *7:Train*, and *8:Subway*.
</p>


<p align="center">
    <img src="/figures/comparison-with-state-of-the-art.png" width="70%">
</p>
<p align="center">
Table: Summary of the evaluation of inference configurations.
Recognition performances (mean and std.) of the best inference configuration is shown along with the recognition performances (mean and std.) averaged over all evaluated configurations. Evaluations repeated 7 times.
The subscripts of the position-specific representations are shortened as $z_{b}$ (*Bag*), $z_{ha}$ (*Hand*), $z_{hi}$ (*Hips*), and $z_{t}$ (*Torso*).
Performance of the baseline models are also displayed.
</p>
