package com.rapplogic.tensorflow;

import com.google.common.base.Charsets;
import com.google.common.io.CharStreams;
import model.Recognition;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.util.List;

public class ClassifierMain {
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            throw new IllegalArgumentException("Usage: ClassifierMain <path to inception_v3_2016_08_28_frozen.pb> <path to image to process>");
        }

        System.out.println("Inception model " + args[0]);
        System.out.println("Image to process " + args[1]);

        byte[] modelBytes = Files.readAllBytes(new File(args[0]).toPath());
        String labels = CharStreams.toString(new InputStreamReader(IdentifierMain.class.getResourceAsStream("/imagenet_slim_labels.txt"), Charsets.UTF_8));

        ImageRecognitionProcessor classifierProcessor = TensorFlowImageClassifier.create(
                modelBytes,
                labels
        );

        BufferedImage image = ImageIO.read(new File(args[1]));

        List<Recognition> recognitions =  classifierProcessor.recognizeImage(image, 0.2f);

        System.out.println("Recognitions " + recognitions);
    }
}
