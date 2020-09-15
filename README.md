# respo.vision


ReSpo.Vision recruitment task - Deep Learning

Problem Definition

Aim of this task is to create a model, which will differentiate between video frames belonging to the match itself, which should be processed by our system and those, which do not contain valuable information - close-ups, audience and pre-match/post-match views.

The task is stated as a binary classification problem.
Data

Four datasets are given, each from a different set of videos. Each dataset is in a separate folder.

Three training datasets are given:

    FrameFilter-set1_4k
    FrameFilter-set2_fifawc2018
    FrameFilter-set3_fhd

And the test set, on which test prediction should be made: FrameFilter-test_set.

In each dataset, the sample are split into two classes:

    pos: positive instances, match frames
    neg: negative instances, non-match frames

All the available instances can be used.

Filenames have a certain structure:

000001-09eada874fc784a668315070-6-400-9400.jpg

extraction_index-match_hash-minute-frame_shift-frame_idx, where:

    extraction_index: index of the extracted frame in a sequence, just for sorting purposes, some indices can exist multiple times
    match_hash: hashed match name
    minute: minute of the match, from which the frame comes
    frame_shift: during extraction, random frame shift (by number of frames equal to  frame_shift) is applied to increase diversity
    frame_idx: global frame index from a video

Task

The task is to construct a model, which will be able to differentiate between the two classes of frames with high accuracy.

Solution of the task should consist of:

    Data loading & preprocessing (if necessary)
    Training/validation split
    Model training
    Model evaluation on the validation set

    Model should be evaluated using suitable metrics
    Short results description should be attached

    Prediction

    Should be made on the test set
    Conclusions: describe, according to your opinion, what are the key aspects to take into account when solving this problem
    Predictions should be attached in form of a .csv file, where the first column is the filename and the second is probability of an instance being a match frame 