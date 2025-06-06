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
 *   June 4, 2025 (Ivan Prigarin, KNIME GmbH, Konstanz, Germany): created
 */
package org.knime.ai.core.ui.message.renderer;

import static j2html.TagCreator.body;
import static j2html.TagCreator.br;
import static j2html.TagCreator.div;
import static j2html.TagCreator.hr;
import static j2html.TagCreator.html;
import static j2html.TagCreator.img;
import static j2html.TagCreator.pre;
import static j2html.TagCreator.rawHtml;
import static j2html.TagCreator.span;

import java.util.Base64;
import java.util.List;
import java.util.Optional;

import org.knime.ai.core.data.message.ImageContentPart;
import org.knime.ai.core.data.message.MessageValue;
import org.knime.ai.core.data.message.MessageValue.MessageContentPart;
import org.knime.ai.core.data.message.MessageValue.MessageType;
import org.knime.ai.core.data.message.MessageValue.ToolCall;
import org.knime.ai.core.data.message.TextContentPart;
import org.knime.core.data.DataColumnSpec;
import org.knime.core.data.renderer.AbstractDataValueRendererFactory;
import org.knime.core.data.renderer.DataValueRenderer;
import org.knime.core.data.renderer.DefaultDataValueRenderer;
import org.knime.core.webui.node.view.table.data.render.DataCellContentType;

import j2html.tags.ContainerTag;

/**
 * Chat-style renderer for {@link MessageValue} (static HTML, no JS).
 *
 * @author Ivan Prigarin, KNIME GmbH, Konstanz, Germany
 */
@SuppressWarnings("serial")
public final class MessageValueRenderer extends DefaultDataValueRenderer
    implements org.knime.core.webui.node.view.table.data.render.DataValueRenderer {

    public static final class Factory extends AbstractDataValueRendererFactory {
        @Override
        public String getDescription() {
            return "Message";
        }

        @Override
        public DataValueRenderer createRenderer(final DataColumnSpec colSpec) {
            return new MessageValueRenderer(getDescription());
        }
    }

    private MessageValueRenderer(final String description) {
        super(description);
    }

    @Override
    public DataCellContentType getContentType() {
        return DataCellContentType.HTML;
    }

    @Override
    protected void setValue(final Object value) {
        final String html;

        if (value instanceof MessageValue messageValue) {
            html = generateMessageValueHTML(messageValue);
        } else {
            html = html(//
                body( //
                    span().withStyle("color: red").withText("?") //
                )).render();
        }

        super.setValue(html);
    }

    private static String esc(final String s) {
        return (s == null) ? "" : s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("\"", "&quot;").replace("'", "&#39;");
    }

    /* ---------- renderer core ---------- */
    public static String generateMessageValueHTML(final MessageValue messageValue) {
        final String html;

        /* ── basics ─────────────────────────────────────────────── */
        MessageType type = messageValue.getMessageType();
        List<MessageContentPart> parts = messageValue.getContent();
        Optional<List<ToolCall>> toolOps = messageValue.getToolCalls();
        Optional<String> toolIdOp = messageValue.getToolCallId();
        Optional<String> toolNameOp = messageValue.getToolName();

        String typeText = type.getLabel();

        /* ── builders for different sections ────────────────────── */
        StringBuilder toolCalls = new StringBuilder();
        StringBuilder toolHeader = new StringBuilder();
        StringBuilder userParts = new StringBuilder();

        /* ── aggregate content parts ────────────────────────────── */
        if (parts == null || parts.isEmpty()) {
            userParts.append("<i>No content parts defined.</i>");
        } else {
            for (int i = 0; i < parts.size(); i++) {
                if (i > 0) {
                    userParts.append(hr().withStyle("border:0;border-top:1px dotted #eee;margin:5px 0;").render());
                }

                MessageContentPart p = parts.get(i);

                if (p instanceof TextContentPart t) {
                    userParts.append(esc(t.getContent()).replace("\n", "<br>"));
                } else if (p instanceof ImageContentPart imgPart) {
                    byte[] raw = imgPart.getData();
                    String mime = imgPart.getType();
                    if (raw != null && raw.length > 0 && mime != null && !mime.isBlank()) {
                        String uri = "data:%s;base64,%s".formatted(mime, Base64.getEncoder().encodeToString(raw));
                        userParts.append(img().withSrc(uri).attr("alt", "[image]").withStyle("""
                                max-width:240px;
                                height:auto;
                                border-radius:6px;
                                border:1px solid #ccc;""").render());
                    } else {
                        userParts.append(span().withStyle("color:#555;font-style:italic;")
                            .withText("[Image part is empty]").render());
                    }
                } else if (p != null) {
                    userParts.append(span().withStyle("""
                            color:#777;
                            font-style:italic;""").withText("[Unsupported Content Part: " + esc(p.getType()) + "]")
                        .render());
                } else {
                    userParts.append("<i>Null content part</i>");
                }
            }
        }

        /* ── tool-calls section (AI) ─────────────── */
        if (MessageType.AI.equals(type) && toolOps.isPresent() && !toolOps.get().isEmpty()) {

            toolCalls.append(div().withStyle("""
                    font-weight:bold;
                    margin-bottom:4px;""").withText("Tool calls:").render());

            for (ToolCall tc : toolOps.get()) {
                toolCalls.append(div().withStyle("""
                        background:#f5f5f5;
                        border:1px solid #ddd;
                        border-radius:4px;
                        padding:8px;
                        margin-bottom:6px;
                        font-size:0.9em;""")
                    .with(div().with(span().withStyle("font-weight:bold;").withText("Tool: "))
                        .withText(esc(tc.toolName())))
                    .with(div().with(span().withStyle("font-weight:bold;").withText("Tool call ID: "))
                        .withText(esc(tc.id())))
                    .with(br()).with(br())
                    .with(div().with(span().withStyle("font-weight:bold;").withText("Arguments:")))
                    .with(pre().withText((tc.arguments() == null || tc.arguments().isBlank()) ? "{}" : tc.arguments()))
                    .render());
            }
        }

        /* ── header for TOOL messages ──────────────────────────── */
        if (MessageType.TOOL.equals(type)) {
            // Tool line only if name present
            if (toolNameOp.isPresent() && !toolNameOp.get().isBlank()) {
                toolHeader.append(div().with(span().withStyle("font-weight:bold;").withText("Tool: "))
                    .withText(esc(toolNameOp.get())).render());
            }
            toolHeader.append(div().with(span().withStyle("font-weight:bold;").withText("Tool call ID: "))
                .withText(esc(toolIdOp.orElse("-"))).render());
            // dashed separator
            toolHeader.append(div().withStyle("border-bottom:1px dashed #ddd;margin:6px 0;").render());
        }

        /* ── assemble final body ───────────────────────────────── */
        String finalBody = "%s%s%s".formatted(toolCalls, toolHeader, userParts);

        /* ── pill background & constant text colour ────────────── */
        String pillBg = switch (type) {
            case USER -> "#d9eeff";
            case AI -> "#e8feec";
            case TOOL -> "#eff1f2";
            default -> "#f0f0f0";
        };
        String pillText = "#616161";

        /* ── build HTML ───────────────────────────────────────── */
        ContainerTag wrapper = div().withStyle("margin-bottom:8px;font-family: Roboto, sans-serif;")
            .with(div().withStyle("""
                    min-width:20%;
                    max-width:80%;
                    width:100%;
                    max-width:600px;
                    box-sizing:border-box;""").with(div().withStyle("""
                    position:relative;
                    width:100%;
                    font-size:13px;
                    font-weight:400;""").with(
                /* type pill */
                div().withStyle("""
                        position:relative;z-index:2;
                        display:inline-block;
                        margin-left:10px;
                        background:%s;
                        outline:1px solid #ffffff;
                        border-radius:16px;
                        padding:0 8px;
                        min-width:40px;
                        text-align:center;
                        font-size:.75em;font-weight:bold;
                        color:%s;""".formatted(pillBg, pillText)).withText(typeText),

                /* body bubble */
                div().withStyle("""
                        margin-top:-6px;
                        border:1px solid #d1d1d1;
                        border-radius:0 4px 4px 4px;
                        background:#ffffff;
                        padding:18px 8px 12px;
                        color:#333;
                        overflow-wrap:break-word;
                        white-space:pre-wrap;""").with(rawHtml(finalBody)))));

        html = wrapper.render();

        return html;
    }
}
