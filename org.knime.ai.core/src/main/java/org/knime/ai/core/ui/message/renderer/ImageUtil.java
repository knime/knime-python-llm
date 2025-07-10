/*
 * ------------------------------------------------------------------------
 *
 *  Copyright by KNIME AG, Zurich, Switzerland
 *  Website: http://www.knime.com; Email: contact@knime.com
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License, Version 3, as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, see <http://www.gnu.org/licenses>.
 *
 *  Additional permission under GNU GPL version 3 section 7:
 *
 *  KNIME interoperates with ECLIPSE solely via ECLIPSE's plug-in APIs.
 *  Hence, KNIME and ECLIPSE are both independent programs and are not
 *  derived from each other. Should, however, the interpretation of the
 *  GNU GPL Version 3 ("License") under any applicable laws result in
 *  KNIME and ECLIPSE being a combined program, KNIME AG herewith grants
 *  you the additional permission to use and propagate KNIME together with
 *  ECLIPSE with only the license terms in place for ECLIPSE applying to
 *  ECLIPSE and the GNU GPL Version 3 applying for KNIME, provided the
 *  license terms of ECLIPSE themselves allow for the respective use and
 *  propagation of ECLIPSE together with KNIME.
 *
 *  Additional permission relating to nodes for KNIME that extend the Node
 *  Extension (and in particular that are based on subclasses of NodeModel,
 *  NodeDialog, and NodeView) and that only interoperate with KNIME through
 *  standard APIs ("Nodes"):
 *  Nodes are deemed to be separate and independent programs and to not be
 *  covered works.  Notwithstanding anything to the contrary in the
 *  License, the License does not apply to Nodes, you are not required to
 *  license Nodes under the License, and you are granted a license to
 *  prepare and propagate Nodes, in each case even if such Nodes are
 *  propagated with or for interoperation with KNIME.  The owner of a Node
 *  may freely choose the license terms applicable to such Node, including
 *  when such Node is propagated with or for interoperation with KNIME.
 * ---------------------------------------------------------------------
 *
 * History
 *   24 June 2025 (Ivan Prigarin, KNIME GmbH, Konstanz, Germany): created
 */
package org.knime.ai.core.ui.message.renderer;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Base64;

import javax.imageio.ImageIO;

/**
 * Utils to help the Message renderer handle PNG content parts.
 *
 * @author Ivan Prigarin, KNIME GmbH, Konstanz, Germany
 */
public class ImageUtil {

    private static final int LONGEST_SIDE = 200;

    /** Builds a data-URI of a LONGEST_SIDE px thumbnail for a PNG byte[].
     * @throws IOException */
    static String createThumbnailForPng(final byte[] pngBytes) throws IOException {
        var in = new ByteArrayInputStream(pngBytes);
        BufferedImage src = ImageIO.read(in);

        if (src == null) {
            throw new IOException("Image may be corrupt or not a valid PNG.");
        }

        // original dimensions
        var w = src.getWidth();
        var h = src.getHeight();
        var scale = Math.min(1f, LONGEST_SIDE / (float)Math.max(w, h));

        // thumb dimensions
        var tw = Math.round(w * scale);
        var th = Math.round(h * scale);

        // resample source PNG into the thumbnail
        BufferedImage dst = new BufferedImage(tw, th, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = dst.createGraphics();
        g.drawImage(src, 0, 0, tw, th, null);
        g.dispose();

        // stream out as HTML-ready binary
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ImageIO.write(dst, "png", out);
        return "data:image/png;base64," +
               Base64.getEncoder().encodeToString(out.toByteArray());
    }

    private ImageUtil() {}

}
