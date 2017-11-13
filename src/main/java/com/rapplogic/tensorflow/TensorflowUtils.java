package com.rapplogic.tensorflow;

import model.Recognition;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.util.List;

public class TensorflowUtils {

    static Logger log = LoggerFactory.getLogger(TensorflowUtils.class);

    /**
     * Draw boxes on image for each reconition and label
     */
    public static void annotateImage(List<Recognition> recognitions, BufferedImage original) {
        // TODO improve readability of boxes
        // FIXME don't draw text off the screen when box at edge of image
        // FIXME don't overlap text

        final Graphics2D graphics2D = original.createGraphics();

        for (Recognition recognition : recognitions) {
            log.debug("Drawing recognition " + recognition);

            graphics2D.setStroke(new BasicStroke(6));
            graphics2D.setColor(Color.YELLOW);

            graphics2D.drawRoundRect(
                    recognition.getRectangle().get().getLeft(),
                    recognition.getRectangle().get().getTop(),
                    recognition.getRectangle().get().getWidth(),
                    recognition.getRectangle().get().getHeight(),
                    20,
                    20
            );

            final String label = recognition.getTitle() + " (" + String.format("%.2f", recognition.getConfidence()) + ")";

            final Font currentFont = graphics2D.getFont();
            final Font newFont = currentFont.deriveFont(currentFont.getSize() * 3.5f);
            graphics2D.setFont(newFont);

            final FontMetrics fontMetrics = graphics2D.getFontMetrics();
            final Rectangle2D textRectangle = fontMetrics.getStringBounds(label, graphics2D);

            graphics2D.setColor(Color.BLACK);

            graphics2D.fillRoundRect(
                    recognition.getRectangle().get().getLeft(),
                    recognition.getRectangle().get().getTop() - (int)(fontMetrics.getHeight() * 0.75f),
                    (int) textRectangle.getWidth(),
                    (int) textRectangle.getHeight(),
                    20, 20);

            graphics2D.setColor(Color.WHITE);

            graphics2D.drawString(
                    label,
                    recognition.getRectangle().get().getLeft(),
                    recognition.getRectangle().get().getTop()
            );

            graphics2D.setFont(currentFont);
        }

        graphics2D.dispose();
    }
}
