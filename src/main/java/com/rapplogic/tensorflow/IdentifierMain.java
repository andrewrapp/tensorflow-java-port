package com.rapplogic.tensorflow;

import com.google.common.base.Charsets;
import com.google.common.io.CharStreams;
import model.Recognition;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public class IdentifierMain {

    /**
     * Run with args: <path to mobilenet model> <directory of images to process>
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

        for (Path path : Files.newDirectoryStream(FileSystems.getDefault().getPath(args[1]))) {
            if (!path.getFileName().toString().toLowerCase().endsWith(".jpg")) {
                continue;
            }

            BufferedImage image = ImageIO.read(path.toFile());

            List<Recognition> recognitions =  identifierProcessor.recognizeImage(image, 0.6f);

            System.out.println("Recognitions for " + path.toString() + ": " + recognitions);

            if (recognitions.size() > 0) {
                TensorflowUtils.annotateImage(recognitions, image);
                File annotated = new File(path.getFileName().toString() + "-annotated.jpg");
                Files.write(annotated.toPath(), Utils.imageToBytes(image));
            }
        }
    }
}
