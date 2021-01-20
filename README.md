# Text2Keypoint

----------------------

The goal of text2keypoint is to **translate text(gloss) to continuous sequence of 2D sign pose keypoints**. We use **<u>Tacotron</u>** of Google as a base model and add masking layer and counter value to stabilize training process and better predict the length of sequence. Evaluation metric is DTW(Dynamic Time Warping) score. The source for modifying model was from [Progressive Transformers SLP (Ben Saunders)](https://github.com/BenSaunders27/ProgressiveTransformersSLP). 

## Installation

------------------------------

Install required packages using the requirements.txt file.

```
pip install -r requirements.txt
```

## Getting Started

---------------------

#### in Colab

- We recommend using GPU in Colab. You can change the runtime type by :[Runtime]-[Change runtime type]-[GPU] 
- `main.ipynb`: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tobigs-team/hand-to-hand/blob/text2keypoint/text2keypoint/main.ipynb)

#### in Web

* Create Virtual Envorionment in Anaconda Prompt

```
conda create -n virtual_env_name
conda activate virtual_env_name
```

* Git clone and Install requirements

```
$git clone https://github.com/Tobigs-team/hand-to-hand.git
cd hand-to-hand/tacotron
$git install -r requirements.txt
```

## Running Train, Test

-------------------

* Set  `mode`  in `Base.yaml`  before Train or Test


```
# default mode is "Train"
mode: "Train" / "Test"

# Set test_mode("recent" or "best") before inference
test_mode: "recent" / "best"
```

* Start Train, Test

```python
cd hand-to-hand/tacotron

!python __main__.py
```

## Reference

------------------------------------------

* [Progressive Transformers for End-to-End Sign Language Production](https://github.com/BenSaunders27/ProgressiveTransformersSLP)

* Hands-On Natural Language Processing with Python: A practical guide to applying deep learning architectures to your NLP applications 1st Edition, Kindle Edition
