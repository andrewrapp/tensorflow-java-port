package model;

public class Rectangle {
    final int left;
    final int top;
    final int height;
    final int width;

    /**
     * Create rectangle with coordinates of the original image from from tensorflow scaled down image results
     */
    public Rectangle(float downScaledLeft, float downScaledTop, float downScaledRight, float downScaledBottom, int tfModelWidthHeight, int originalWidth, int originalHeight) {
        float widthScaleFactor = ((float) originalWidth) / ((float) tfModelWidthHeight);
        float heightScaleFactor = ((float) originalHeight) / ((float) tfModelWidthHeight);

        this.left = scaleBy(widthScaleFactor, downScaledLeft);
        this.top = scaleBy(heightScaleFactor, downScaledTop);
        this.width = scaleBy(widthScaleFactor, downScaledRight - downScaledLeft);
        this.height = scaleBy(heightScaleFactor, downScaledBottom - downScaledTop);
    }

    public int getLeft() {
        return left;
    }

    public int getTop() {
        return top;
    }

    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }

    private int scaleBy(float scaleBy, float position) {
        return (int) (scaleBy * position);
    }

    @Override
    public String toString() {
        return "Rectangle{" +
                "left=" + left +
                ", top=" + top +
                ", height=" + height +
                ", width=" + width +
                '}';
    }
}
