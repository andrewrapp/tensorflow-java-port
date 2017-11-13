package model;


import arapp.tensorflow.Utils;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Optional;

public class RecognitionResult {
    private Optional<ByteBuffer> maybeByteBuffer;
    private List<Recognition> recognitions;

    public RecognitionResult(Optional<BufferedImage> annotatedImage, List<Recognition> recognitions) throws IOException {
        if (annotatedImage.isPresent()) {
            this.maybeByteBuffer = Optional.of(imageToByteBuffer(annotatedImage.get()));
        } else {
            this.maybeByteBuffer = Optional.empty();
        }

        this.recognitions = recognitions;
    }

    public ByteBuffer imageToByteBuffer(BufferedImage image) throws IOException {
        return ByteBuffer.wrap(Utils.imageToBytes(image));
    }

    public Optional<ByteBuffer> getAnnotatedImageByteBuffer() {
        return maybeByteBuffer;
    }

    public List<Recognition> getRecognitions() {
        return recognitions;
    }
}
