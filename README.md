# Language-Guided Image Colorization

## Data Preparation & Set-up
### Data sources
* Image: COCO 2017 train/test split
* Language: COCO 2015 captioning challenge
* Segmentation: [COCO-Stuff dataset](https://github.com/nightrome/cocostuff)
* Word embedding: [download](https://drive.google.com/open?id=1mRm68FDYak732h_bpaTIKyAt2Tvoke7x)
* Vocabulary indexing: [download](https://drive.google.com/open?id=1lHcJkbuNgrTU8zWw1DNbJhSMeGBg8n4x)
* Color vocabulary: [download](https://drive.google.com/open?id=1gD4-ItPIN2fL_y1VVsrT6Sd07Fkjir_s)
### Data pre-processing
* Scale the images and segmentation masks so that the shorter edge size equals 224.
* Convert the captions to lists of word indices according to the vocabulary indexing.
* Structure the images and segmentation masks in the following way:
'''
  DATA_ROOT
    annotations_224
      train2017
        *.png
      val2017
        *.png
    images_224
      train2017
        *.png
      test2017
        *.png
      val2017
        *.png
'''

Helper functions can be found in data_preprocess.py.

## Training
## Inference
Sample command (for our full method):
```
python -u colorize.py -g GPU_ID -d OUTPUT_DIR -m 10 --gru 0 --seg_ver 0 --weights PATH_TO_WEIGHTS
```
More detailed usage of the arguments can be found in colorize.py.

## Evaluation
### AMT
Helper functions for the evaluation on the AMT platform can be found in amt.py.
### Reference Metrics


## Pretrained Model
Download the pretrained weights for the full method [here](https://drive.google.com/open?id=1LGqmmiUok_Gwhvq0z5DYClOalLigMQQG).
@TODO: update.

## Acknowledgement
* [COCO-Stuff](https://github.com/nightrome/cocostuff)
* [Colorful Image Colorization](https://github.com/richzhang/colorization)
* [colorization-tf](https://github.com/nilboy/colorization-tf)
* [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch)
* [FiLM: Visual Reasoning with a General Conditioning Layer](https://github.com/ethanjperez/film)
* [Image Captions Generation with Spatial and Channel-wise Attention](https://github.com/zjuchenlong/sca-cnn.cvpr17)
* [Learning to Color from Language](https://github.com/superhans/colorfromlanguage)
