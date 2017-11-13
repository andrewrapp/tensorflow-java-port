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

public class IdentifierMain {

    /**
     * Run with args: <path to mobilenet model> <path to image to process>
     */
    public static void main(String[] args) throws Exception {

        if (args.length < 2) {
            throw new IllegalArgumentException("Usage: IdentifierMain <path to ssd_mobilenet_v1_android_export.pb> <path to image to process>");
        }

        System.out.println("Mobilenet model " + args[0]);
        System.out.println("Image to process " + args[1]);

        // path to ssd_mobilenet_v1_android_export.pb
        byte[] modelBytes = Files.readAllBytes(new File(args[0]).toPath());
        String labels = CharStreams.toString(new InputStreamReader(IdentifierMain.class.getResourceAsStream("/coco_labels_list.txt"), Charsets.UTF_8));

        ImageRecognitionProcessor identifierProcessor = TensorFlowObjectDetectionAPIModel.create(
                modelBytes,
                labels,
                100
        );

        BufferedImage image = ImageIO.read(new File(args[1]));

        List<Recognition> recognitions =  identifierProcessor.recognizeImage(image, 0.6f);

        System.out.println("Recognitions " + recognitions);
    }
}
