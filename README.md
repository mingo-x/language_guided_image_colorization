# Language-Guided Image Colorization

## Data Preparation & Set-up
### Data sources
* Image: COCO 2017 train/test split
* Language: COCO 2015 captioning challenge, or use our preprocessed caption data: [im2cap.p](https://drive.google.com/open?id=1J1UHKf2udKdyf6aozt-0ukKzI0u4bpKT), [captions.npy](https://drive.google.com/open?id=1qfvcldN3lR9kDQn43vTmQWHisPNq7Hux), [caption_lengths.npy](https://drive.google.com/open?id=1bUMJYCRXcrPaUtwpgz_dNsA7LKhDzIJn)
* Segmentation: [COCO-Stuff dataset](https://github.com/nightrome/cocostuff)
* Word embedding: [download](https://drive.google.com/open?id=1mRm68FDYak732h_bpaTIKyAt2Tvoke7x)
* Vocabulary indexing: [download](https://drive.google.com/open?id=1lHcJkbuNgrTU8zWw1DNbJhSMeGBg8n4x)
* Color vocabulary: [download](https://drive.google.com/open?id=1gD4-ItPIN2fL_y1VVsrT6Sd07Fkjir_s)
### Data pre-processing
* Scale the images and segmentation masks so that the shorter edge size equals 224.
* Convert the captions to lists of word indices according to the vocabulary indexing, or use our preprocessed caption data, and set _\_IM2CAP\_PATH_, _\_CAPTIONS\_PATH_, _\_CAPTION\_LENGTHS\_PATH_ in _datasets.py_ to the corresponding file locations.
* You may want to overwrite the _resources/val_filtered_gray.p_ file, which contains the IDs of grayscale images, if you use different image datasets.
* Structure the images and segmentation masks in the following way:
```
  DATA_ROOT
    - annotations_224
      -- train2017
        --- *.png
      -- val2017
        --- *.png
    - images_224
      -- train2017
        --- *.png
      -- test2017
        --- *.png
      -- val2017
        --- *.png
```
Helper functions can be found in _data\_preprocess.py_.

## Training
Sample command (for our full method):
```
python -u vgg_with_segmentation.py -g GPU_ID -d OUTPUT_DIR --lr 0.001 -wd 1e-5 -m 10 --gru 0 --seg_ver 0 --data_root DATA_ROOT --embedding_path PATH_TO_WORD_EMBEDDING --vocabulary_path PATH_TO_VOCABULARY
```
More detailed usage of the arguments can be found in _vgg_with_segmentation.py_.

## Inference
Sample command (for our full method):
```
python -u colorize.py -g GPU_ID -d OUTPUT_DIR -m 10 --gru 0 --seg_ver 0 --weights PATH_TO_WEIGHTS --data_root DATA_ROOT --embedding_path PATH_TO_WORD_EMBEDDING --vocabulary_path PATH_TO_VOCABULARY
```
More detailed usage of the arguments can be found in _colorize.py_.

## Evaluation
### AMT
Helper functions for the evaluation on the AMT platform can be found in _amt.py_.
### Reference Metrics
Set _\_GT\_DIR_ and _\_INFERENCE\_DIR_ in _evaluate.py_ and run. You may want to overwrite the _resources/val_filtered_gray.p_ file if you use different image datasets.

## Pretrained Model
Download the pretrained weights for the full method [here](https://drive.google.com/open?id=1o_avtX8iE9F-B78c4ypDKIYyJ2I6QESL).

## Acknowledgement
* [COCO-Stuff](https://github.com/nightrome/cocostuff)
* [Colorful Image Colorization](https://github.com/richzhang/colorization)
* [colorization-tf](https://github.com/nilboy/colorization-tf)
* [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch)
* [FiLM: Visual Reasoning with a General Conditioning Layer](https://github.com/ethanjperez/film)
* [Image Captions Generation with Spatial and Channel-wise Attention](https://github.com/zjuchenlong/sca-cnn.cvpr17)
* [Learning to Color from Language](https://github.com/superhans/colorfromlanguage)
