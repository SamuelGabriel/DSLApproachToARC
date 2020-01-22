# Very initial try to solve a subset of ARC

This is based on Few-shot Bayesian Imitation Learning with Policies as Logic over Programs and is a fork of the repo [https://github.com/tomsilver/policies_logic_programs](Tom Silver's code).
The setup right now only solves very few of the ARC problems, but maybe improvements in the DSL (`dsl.py`) might lead there.
We only consider the 66% of games in the training set with same size in- and output. To address the rest I guess we need to change the setup a litte.

## System Requirements
We use Python 3.5.6 on macOS High Sierra. Other setups may work but have not been tested.

## Installation

Run
```
git clone https://github.com/tomsilver/generalization_grid_games.git
git clone https://github.com/fchollet/ARC.git
pip install scikit-learn==0.20.3
pip install imageio-ffmpeg
```

## Usage Example

```
python pipeline.py
```

Settings are so far defined at the very bottom of `pipeline.py`.
