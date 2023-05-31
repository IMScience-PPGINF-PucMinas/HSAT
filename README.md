Hierarchical time-aware summarization with an adaptive transformer for video captioning
=====
PyTorch code for our paper "Hierarchical time-aware summarization with an adaptive transformer for video captioning" Enhanced by [Leonardo Vilela Cardoso](http://lattes.cnpq.br/6741312586742178), [Silvio Jamil F. Guimarães](http://lattes.cnpq.br/8522089151904453) and [Zenilton K. G. Patrocínio Jr](http://lattes.cnpq.br/8895634496108399), submitted to IJSC on march 2023.

A coherent description is an ultimate goal regarding video captioning via a couple of sentences because it might also affect the consistency and intelligibility of the generated results. In this context, a paragraph describing a video is affected by the activities used to both produce its specific narrative and provide some clues that can also assist in decreasing textual repetition. This work proposes a model, named Hierarchical timeaware Summarization with an Adaptive Transformer – HSAT, that uses a strategy to enhance the frame selection reducing the amount of information that needed to be processed along with attention mechanisms to enhance a memory-augmented transformer. This new approach increases the coherence among the generated sentences, assessing
data importance (about the video segments) contained in the self-attention results and uses that to improve readability using only a small fraction of time spent by the other methods. The test results show the potential of this new approach as it provides higher coherence among the various video segments, decreasing the repetition in the generated sentences and improving the description diversity in the ActivityNet Captions dataset.

## Getting started
### Prerequisites
0. Clone this repository
```
# no need to add --recursive as all dependencies are copied into this repo.
git clone https://github.com/IMScience-PPGINF-PucMinas/Video-Summarization.git
cd Video-Summarization
```

1. Download Files

This code is used for compute the time-aware graph for the [activitynet dataset](http://activity-net.org/download.html). But, can be applied for any video dataset

2. Install dependencies
- Python 3.6
- PyTorch 1.10.0
- torchvision 0.11.2
- opencv-python 4.7.0.72
- Pyllow 8.4.0
- imageio 2.15.0
- matplotlib 3.3.4
- scikit-learn 0.24.2
- networkx 2.5.1
- numpy 1.19.5
- scipy 1.5.4
- easydict
- protobuf
- youtube-dl
- Flask

### Frame Extraction and Time-Aware Video Processing


1. Frame extraction

To extract frames for the video command is:
```
python src/frame_extractor.py "your_dataset_path"
```

2. Time-Aware graph generation

To generate the graph for the video the command is:
```
python src/frame_similarity_evaluation.py "your_frame_path" "dela_time"
```

The dela_time are the threshold for the time and, are used to controller the time-aware.

The results are a list of Keyframes, that's possible to use as input for another method as the collection of frames selected with a hierarchical approach, considering or not one time-aware variation.

### Feature Extraction Code for Dense Video Captioning
This code extract two types of features
- RGB ResNet-200 feature
- Optical flow BN-Inception feature

1. Prepare Data

## Additional Requirements

NVIDIA GPU with CUDA support. At least 4GB display memory is needed to run the reference models.

For this step, the code use Caffe and OpenCV. 
Particularly, the OpenCV should be compiled with VideoIO support. GPU support will be good if possible.
If you use `build_all.sh`, it will locally install these dependencies for you.

## Single Video Classification

- Build all modules
In the root directory of the project, run the following command
```
bash build_all.sh
```
- Get the reference models
```
bash scripts/get_reference_models.sh
```
- Run the classification
There is a video clip in the `data/plastering.avi` for your example.
To do single video classification with RGB model one can run
```
python src/classify_video.py data/plastering.avi
```
It should print the top 3 prediction in the output.
To use the two-stream model, one can add `--use_flow` flag to the command. The framework will then extract optical flow on the fly.
```
python src/classify_video.py --use_flow data/plastering.avi
```
You can use your own video files by specifying the filename. 

One can also specify a youtube url here to do the classification, for example
```
python src/classify_video.py https://www.youtube.com/watch?v=QkuC0lvMAX0
```

The two-stream model here consists of one reset-200 model for RGB input and one BN-Inception model for optical flow input. The model spec and parameter files can be found in `models/`.

## Others
This code uses resources from the following projects: 
[Densecap](https://github.com/LuoweiZhou/densecap) and 
[CUHK & ETH & SIAT](https://github.com/yjxiong/anet2016-cuhk)

## Citations
If you find this code useful for your research, consider cite our paper:
```
Accepted on IJSC 2023
```
## Contact
Leonardo Vilela Cardoso with this e-mail: leonardocardoso@pucminas.br
