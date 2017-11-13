package model;


import java.util.Optional;

/**
 * An immutable result of the recognitions
 */
public class Recognition {
    /**
     * A unique identifier for what has been recognized. Specific to the class, not the instance of
     * the object.
     */
    private final String id;

    /**
     * Display name for the recognition.
     */
    private final String title;

    /**
     * A sortable score for how good the recognition is relative to others. Higher should be better.
     */
    private final Float confidence;

    private Optional<Rectangle> rectangle;

    public Recognition(final String id, final String title, final Float confidence, Optional<Rectangle> rectangle) {
        this.id = id;
        this.title = title;
        this.confidence = confidence;
        this.rectangle = rectangle;
    }

    public String getId() {
        return id;
    }

    public String getTitle() {
        return title;
    }

    public Float getConfidence() {
        return confidence;
    }

    public Optional<Rectangle> getRectangle() {
        return rectangle;
    }

    @Override
    public String toString() {
        return "Recognition{" +
                "id='" + id + '\'' +
                ", title='" + title + '\'' +
                ", confidence=" + confidence +
                ", coordinates=" + rectangle +
                '}';
    }
}
