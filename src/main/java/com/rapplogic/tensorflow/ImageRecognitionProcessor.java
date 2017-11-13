package com.rapplogic.tensorflow;


import model.Recognition;

import java.awt.image.BufferedImage;
import java.util.List;


public interface ImageRecognitionProcessor {
    List<Recognition> recognizeImage(BufferedImage image, float minConfidence) throws Exception;
    void close();
}
