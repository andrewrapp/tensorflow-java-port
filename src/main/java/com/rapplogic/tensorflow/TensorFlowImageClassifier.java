package com.rapplogic.tensorflow;

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import arapp.tensorflow.Utils;
import model.Recognition;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.Operation;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.*;
import java.util.List;

/** A classifier specialized to label images using TensorFlow. */
public class TensorFlowImageClassifier implements ImageRecognitionProcessor {
    // Only return this many results with at least this confidence.
    private static final int MAX_RESULTS = 10;

    static Logger log = LoggerFactory.getLogger(TensorFlowImageClassifier.class);

    // Config values.
    private String inputName;
    private String outputName;
    private int inputSize;
    private int imageMean;
    private float imageStd;

    // Pre-allocated buffers.
    private List<String> labels = new ArrayList<String>();;
    private float[] outputs;
    private String[] outputNames;

    private boolean logStats = false;

    private int maxResults;

    private TensorFlowInferenceInterface inferenceInterface;

    private TensorFlowImageClassifier() {}

    private final static int INPUT_SIZE = 299;
    private final static int IMAGE_MEAN = 0;
    private final static int IMAGE_STD = 255;
    private final static String INPUT_NAME = "input";
    private final static String OUTPUT_NAME = "InceptionV3/Predictions/Reshape_1";

    public static ImageRecognitionProcessor create(byte[] modelBytes, String labels) {
        TensorFlowImageClassifier tensorFlowImageClassifier = new TensorFlowImageClassifier();

        tensorFlowImageClassifier.maxResults = MAX_RESULTS;
        tensorFlowImageClassifier.inputName = INPUT_NAME;
        tensorFlowImageClassifier.outputName = OUTPUT_NAME;

        tensorFlowImageClassifier.labels.addAll(Arrays.asList(labels.split("\n")));

        tensorFlowImageClassifier.inferenceInterface = new TensorFlowInferenceInterface(modelBytes);

        // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
        final Operation operation = tensorFlowImageClassifier.inferenceInterface.graphOperation(OUTPUT_NAME);
        final int numClasses = (int) operation.output(0).shape().size(1);

        // Ideally, inputSize could have been retrieved from the shape of the input operation.  Alas,
        // the placeholder node for input in the graphdef typically used does not specify a shape, so it
        // must be passed in as a parameter.
        tensorFlowImageClassifier.inputSize = INPUT_SIZE;
        tensorFlowImageClassifier.imageMean = IMAGE_MEAN;
        tensorFlowImageClassifier.imageStd = IMAGE_STD;

        // Pre-allocate buffers.
        tensorFlowImageClassifier.outputNames = new String[] {OUTPUT_NAME};
        tensorFlowImageClassifier.outputs = new float[numClasses];

        return tensorFlowImageClassifier;
    }

    @Override
    public synchronized List<Recognition> recognizeImage(BufferedImage original, float minConfidence) throws Exception {

        BufferedImage imageScaled = Utils.toBufferedImage(original.getScaledInstance(inputSize, inputSize, Image.SCALE_DEFAULT));

        float[] floatValues = convertImage(imageScaled);

        // Copy the input data into TensorFlow.
        inferenceInterface.feed(inputName, floatValues, 1, inputSize, inputSize, 3); // why is this 3?
        // Run the inference call.
        inferenceInterface.run(outputNames, logStats);
        // Copy the output Tensor back into the output array.
        inferenceInterface.fetch(outputName, outputs);

        // Find the best classifications.
        PriorityQueue<Recognition> priorityQueue =
                new PriorityQueue<Recognition>(
                        maxResults,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (int i = 0; i < outputs.length; ++i) {
            Recognition recognition = new Recognition("" + i, labels.size() > i ? labels.get(i) : "unknown", outputs[i], Optional.empty());

            if (outputs[i] > minConfidence) {
                priorityQueue.add(recognition);
            } else if (outputs[i] > 0.1f) {
                log.debug("Recognition is below threshold.. dropping " + recognition);
            }
        }

        final List<Recognition> recognitions = new ArrayList<Recognition>();

        int recognitionsSize = Math.min(priorityQueue.size(), maxResults);

        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(priorityQueue.poll());
        }

        return recognitions;
    }

    private float[] convertImage(BufferedImage imageScaled) {
        int[] intValues = imageScaled.getRGB(0, 0, imageScaled.getWidth(), imageScaled.getHeight(), null, 0, imageScaled.getWidth());

        float[] floatValues = new float[imageScaled.getWidth() * imageScaled.getHeight()*3];

        for (int i = 0; i < intValues.length; i++) {
            int val = intValues[i];

            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
        }

        return floatValues;
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }
}

