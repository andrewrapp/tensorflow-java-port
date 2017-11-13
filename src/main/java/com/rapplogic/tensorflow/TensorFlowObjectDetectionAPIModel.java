package com.rapplogic.tensorflow;

import model.Recognition;
import model.Rectangle;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.Graph;
import org.tensorflow.Operation;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.*;
import java.util.List;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/object_detection
 */
public class TensorFlowObjectDetectionAPIModel implements ImageRecognitionProcessor {

    Logger log = LoggerFactory.getLogger(TensorFlowObjectDetectionAPIModel.class);

    // required size of model
    private final static int TF_OD_API_INPUT_SIZE = 300;

    // Config values.
    private String inputName;
    private int inputSize = TF_OD_API_INPUT_SIZE;

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private float[] outputLocations;
    private float[] outputScores;
    private float[] outputClasses;
    private float[] outputNumDetections;
    private String[] outputNames;

    private boolean logStats = false;

    private int maxResults;

    private TensorFlowInferenceInterface inferenceInterface;

    /**
     * Initializes a native TensorFlow session for classifying images.
     */
    public static ImageRecognitionProcessor create(
            final byte[] modelBytes,
            final String cocoLabels,
            final int maxResults) throws IOException {
        final TensorFlowObjectDetectionAPIModel instance = new TensorFlowObjectDetectionAPIModel();

        instance.maxResults = maxResults;

        instance.labels.addAll(Arrays.asList(cocoLabels.split("\n")));

        instance.inferenceInterface = new TensorFlowInferenceInterface(modelBytes);

        final Graph graph = instance.inferenceInterface.graph();

        instance.inputName = "image_tensor";
        // The inputName node has a shape of [N, H, W, C], where
        // N is the batch size
        // H = W are the height and width
        // C is the number of channels (3 for our purposes - RGB)
        final Operation inputOperation = graph.operation(instance.inputName);

        if (inputOperation == null) {
            throw new RuntimeException("Failed to find input Node '" + instance.inputName + "'");
        }

        // The outputScoresName node has a shape of [N, NumLocations], where N is the batch size.
        final Operation detectionScores = graph.operation("detection_scores");

        if (detectionScores == null) {
            throw new RuntimeException("Failed to find output Node 'detection_scores'");
        }

        final Operation detectionBoxes = graph.operation("detection_boxes");

        if (detectionBoxes == null) {
            throw new RuntimeException("Failed to find output Node 'detection_boxes'");
        }

        final Operation detectionClasses = graph.operation("detection_classes");

        if (detectionClasses == null) {
            throw new RuntimeException("Failed to find output Node 'detection_classes'");
        }

        // Pre-allocate buffers.
        instance.outputNames = new String[] {"detection_boxes", "detection_scores", "detection_classes", "num_detections"};
        instance.outputScores = new float[maxResults];
        instance.outputLocations = new float[maxResults * 4];
        instance.outputClasses = new float[maxResults];
        instance.outputNumDetections = new float[1];

        return instance;
    }

    private TensorFlowObjectDetectionAPIModel() {}

    @Override
    public synchronized List<Recognition> recognizeImage(BufferedImage image, float minConfidence) throws Exception {

        log.debug("Recognizing image: width " + image.getWidth() + ", height " + image.getHeight());

        // we are changing (most likely) the aspect ratio here.. won't that affect the result?
        BufferedImage imageScaled = Utils.toBufferedImage(image.getScaledInstance(inputSize, inputSize, Image.SCALE_DEFAULT));

        byte[] byteValues = convertImage(imageScaled);

        long start = System.currentTimeMillis();

        // Copy the input data into TensorFlow.
        inferenceInterface.feed(inputName, byteValues, 1, inputSize, inputSize, 3);

        // Run the inference call.
        inferenceInterface.run(outputNames, logStats);

        // Copy the output Tensor back into the output array.
        outputLocations = new float[maxResults * 4];
        outputScores = new float[maxResults];
        outputClasses = new float[maxResults];
        outputNumDetections = new float[1];
        inferenceInterface.fetch(outputNames[0], outputLocations);
        inferenceInterface.fetch(outputNames[1], outputScores);
        inferenceInterface.fetch(outputNames[2], outputClasses);
        inferenceInterface.fetch(outputNames[3], outputNumDetections);

        // Find the best detections.
        final PriorityQueue<Recognition> priorityQueue =
                new PriorityQueue<Recognition>(
                        1,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(final Recognition lhs, final Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        // Scale them back to the input size.
        for (int i = 0; i < outputScores.length; ++i) {
            Rectangle rectangle = new Rectangle(
                    outputLocations[4 * i + 1] * inputSize,
                    outputLocations[4 * i] * inputSize,
                    outputLocations[4 * i + 3] * inputSize,
                    outputLocations[4 * i + 2] * inputSize,
                    inputSize,
                    image.getWidth(),
                    image.getHeight()
            );

            Recognition recognition = new Recognition("" + i, labels.get((int) outputClasses[i]), outputScores[i], Optional.of(rectangle));

            if (recognition.getConfidence() >= minConfidence) {
                //                log.debug("Recogition is above threshold " + recognition);
                priorityQueue.add(recognition);
            } else if (recognition.getConfidence() >= 0.1f) {
                log.debug("Recogition is below threshold.. dropping " + recognition);
            }

        }

        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();

        for (int i = 0; i < priorityQueue.size(); ++i) {
            recognitions.add(priorityQueue.poll());
        }

//        log.debug("Identified image in " + (basicTimer.elapsedMillis()) + "ms");

        return recognitions;
    }

    private byte[] convertImage(BufferedImage imageScaled) {
        int[] intValues = imageScaled.getRGB(0, 0, imageScaled.getWidth(), imageScaled.getHeight(), null, 0, imageScaled.getWidth());
        //        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        byte[] byteValues = new byte[imageScaled.getWidth() * imageScaled.getHeight() * 3];

        for (int i = 0; i < intValues.length; ++i) {
            byteValues[i * 3 + 2] = (byte) (intValues[i] & 0xFF);
            byteValues[i * 3 + 1] = (byte) ((intValues[i] >> 8) & 0xFF);
            byteValues[i * 3 + 0] = (byte) ((intValues[i] >> 16) & 0xFF);
        }

        return byteValues;
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }

}