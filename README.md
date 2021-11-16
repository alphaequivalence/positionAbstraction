# Reduction of the Position Bias via Multi-Level Learning for Activity Recognition

Code accompanying the paper

> [Reduction of the Position Bias via Multi-Level Learning for Activity Recognition](/docs/417.pdf)
>
> [Supplementary file](/docs/417_suppl.pdf)

The relative position of sensors placed on specific body parts generates two types of data related to (1) the movement of the body part w.r.t. the body and (2) the whole body w.r.t. the environment. These two data provide orthogonal and complementary components contributing differently to the activity recognition process. In this paper, we introduce an original approach that separates these data and abstracts away the sensors' exact on-body position from the considered activities. We learn for these two totally orthogonal components (i) the bias that stems from the position and (ii) the actual patterns of the activities abstracted from these positional biases. We perform a thorough empirical evaluation of our approach on the various datasets featuring on-body sensor deployment in real-life settings. Obtained results show substantial improvements in performances measured by the f1-score and pave the way for developing models that are agnostic to both the position of the data generators and the target users.



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
The global learner <img src="https://render.githubusercontent.com/render/math?math=L_{\mathcal{S}}"> starts with an initial set of weights which are distributed to the local learners.
The local learners <img src="https://render.githubusercontent.com/render/math?math=L_p">, one for each position <img src="https://render.githubusercontent.com/render/math?math=p">, learn the two vector components <img src="https://render.githubusercontent.com/render/math?math=z_{A}"> and <img src="https://render.githubusercontent.com/render/math?math=z_{P}">, by performing independently a set of gradient steps which allows to get a newer version. These new versions are used during the conciliation step which gives us a new version of the global learner, and subsequently a more robust position-independent representation.
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

    python fl.py

this will load the data and run the local learner, defined inside `model.py`, in a federated learning-like fashion. In the other hand, to train the local learner in a standard fashion, issue `python standard.py`.

In order to display the usage, add `--help` option as:

    python fl.py --help

or

    python standard.py --help

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
Figure: Confusion matrices obtained using different inference configurations. Combination of the universal components <img src="https://render.githubusercontent.com/render/math?math=z_{A}"> and: (a) _Torso_-specific components; (b) _Hand_-specific components; (c) _Bag_-specific components; (d) _Hips_-specific components.  The activities are numbered as _1:Still_, _2:Walk_, _3:Run_, _4:Bike_, _5:Car_, _6:Bus_, _7:Train_, and _8:Subway_.
</p>


<p align="center">
    <img src="/figures/comparison-with-state-of-the-art.png" width="70%">
</p>
<p align="center">
Table: Summary of the evaluation of inference configurations.
Recognition performances (mean and std.) of the best inference configuration is shown along with the recognition performances (mean and std.) averaged over all evaluated configurations. Evaluations repeated 7 times.
The subscripts of the position-specific representations are shortened as <img src="https://render.githubusercontent.com/render/math?math=z_{b}"> (_Bag_), <img src="https://render.githubusercontent.com/render/math?math=z_{ha}"> (_Hand_), <img src="https://render.githubusercontent.com/render/math?math=z_{hi}"> (_Hips_), and <img src="https://render.githubusercontent.com/render/math?math=z_{t}"> (_Torso_).
Performance of the baseline models are also displayed.
</p>
