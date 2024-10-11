# HTAMotr: A Novel Half-To-All MOTR Approach for Robust Video Text Tracking with Incomplete Annotations
## Introduction

![Overview](https://github.com/Paige-Norton/HTAMotr/blob/master/doc/network.png)

**Abstract.** This paper presents a novel Half-To-All Multiple Object Tracking and Recognition (HTAMotr) approach tailored to address the challenges posed by incomplete data annotations in video text tracking.  The proposed method introduces three key strategies: rotated queries to improve anchor alignment with text regions, the Proposal-For-Groundtruth Strong Correlation (PForG) strategy to mitigate the negative effects of incomplete annotations, and an overlapping anchor filter to resolve ID switching issues.  Experiments conducted on the DSText dataset demonstrate the effectiveness of HTAMotr, achieving state-of-the-art performance without requiring additional pre-training data or extensive epochs.  By addressing the limitations of traditional MOTR paradigms, this work contributes to advancing video text tracking techniques and facilitating the development of more robust and efficient algorithms.  The code and datasets are available at https://github.com/Paige-Norton/HTAMotr, complete with usage guides to facilitate the reproduction of experimental results.

## Main Results

### [DSText 2023](https://rrc.cvc.uab.es/?ch=22&com=evaluation&task=1)

Methods | MOTR paradigm | End2End | MOTA | MOTP | IDF1 | Mostly Matched |	Mostly Lost
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
TransDETR 			| √ | √ |  27.55 | 78.40 | 44.28 | 1583 | 9891
Liu et al.		 	| × | × |  36.87 | 79.24 | 48.99 | 2123 | 6829
HTAMotr(No filter) 	| √ | √ |  48.91 | 75.03 | 63.07 | 6394 | 2295
HTAMotr 			| √ | × |  56.22 | 75.15 | 65.08 | 6275 | 2361


#### Notes
- HTAMotr's  already trained models can be found in [Google Drive](https://drive.google.com/file/d/1FF8oRNjPEOksBmi9kihReRdDt08S-IlW/view?usp=drive_link)
- The results of the HTAMotr are available in [Google Drive](https://drive.google.com/file/d/11qqjjezKhv3rr1B3ROcdZTutg5JkggHc/view?usp=drive_link)
- All experiments were conducted using PyTorch on NVIDIA GeForce RTX 3090 GPUs.
- All experiments were not pre-trained on other datasets
### Visualization

<!-- |OC-SORT|MOTRv2| -->
|TransDETR|HTAMotr|
|:-:|:-:|
|![](https://github.com/Paige-Norton/HTAMotr/blob/master/doc/HTAMotr_Video_156_5_6.gif)|![](https://github.com/Paige-Norton/HTAMotr/blob/master/doc/HTAMotr_Video_156_5_6.gif)|
|![](https://github.com/Paige-Norton/HTAMotr/blob/master/doc/TransDETR_Video_214_1_4.gif)|![](https://github.com/Paige-Norton/HTAMotr/blob/master/doc/HTAMotr_Video_214_1_4.gif)|
|![](https://github.com/Paige-Norton/HTAMotr/blob/master/doc/TransDETR_Video_220_2_0.gif)|![](https://github.com/Paige-Norton/HTAMotr/blob/master/doc/HTAMotr_Video_220_2_0.gif)|


## Installation

The codebase is built on top of [MOTRv2](https://github.com/megvii-research/MOTRv2), and the idea of generating queries through proposals originates from this paper. Thanks for providing us with the code framework and idea.

### Requirements

* Install pytorch using conda (optional)

    ```bash
    conda create -n HTAMotr python=3.8
    conda activate HTAMotr
    conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=10.2 -c pytorch
    ```

* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

* Build MultiScaleDeformableAttention
    ```bash
    cd ./models/ops
    sh ./make.sh
    ```

## Usage


### Dataset preparation

Since the incomplete DSText data was completed when the author wrote the article, please download the [videos](https://rrc.cvc.uab.es/?ch=22&com=downloads) and [incomplete DSText annotation](https://drive.google.com/file/d/1TuQEC7f4d6lS36Z9Y2MJT9Idr-T1Xusp/view?usp=drive_link) to reproduce this code and organize them as following:

```
.
├── data
│	├── DSText
│	   ├── images
│	       ├── train
│	           ├── Activity
│	           ├── Driving
│	           ├── Game
│	           ├── ....
│	       ├── test
│	           ├── Activity
│	           ├── Driving
│	           ├── Game
│	           ├── ....
│	   ├── labels_with_ids
│	       ├── train
│	           ├── Activity
│	           ├── Driving
│	           ├── Game
│	           ├── ....
│
│	├── r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth
```

The data annotations we provide are incomplete, which is a key focus of this study and holds significant importance for reproducing the experiments presented in this paper.

### Training

Similar to MOTRv2, you may download the coco pretrained weight from [Deformable DETR (+ iterative bounding box refinement)](https://github.com/fundamentalvision/Deformable-DETR#:~:text=config%0Alog-,model,-%2B%2B%20two%2Dstage%20Deformable), and modify the `--pretrained` argument to the path of the weight. Then training HTAMotr on 2 GPUs as following:

```bash 
./tools/train.sh configs/motrv2DSText.args
```

### Inference on DSText Test Set

You can download the trained model checkpoint0006.pth in [Google Drive](https://drive.google.com/file/d/1FF8oRNjPEOksBmi9kihReRdDt08S-IlW/view?usp=drive_link) to perform inference directly.


```bash
# run a simple inference on our pretrained weights
./tools/simple_inference.sh ./exps/motrv2DSText/run1_20query/checkpoint0006.pth

# then zip the results
zip motrv2.zip tracker/ -r
```

