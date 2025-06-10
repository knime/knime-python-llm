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
import static j2html.TagCreator.head;
import static j2html.TagCreator.hr;
import static j2html.TagCreator.html;
import static j2html.TagCreator.i;
import static j2html.TagCreator.img;
import static j2html.TagCreator.pre;
import static j2html.TagCreator.span;
import static j2html.TagCreator.style;

import java.util.Base64;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

import org.knime.ai.core.data.message.PngContentPart;
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

import j2html.tags.DomContent;
import j2html.tags.specialized.DivTag;

/**
 * HTML-based renderer for {@link MessageValue}. The generated HTML is used for rendering messages both in cells and in
 * pop-out views.
 *
 * @author Ivan Prigarin, KNIME GmbH, Konstanz, Germany
 */
@SuppressWarnings("serial")
public final class MessageValueRenderer extends DefaultDataValueRenderer
    implements org.knime.core.webui.node.view.table.data.render.DataValueRenderer {

    /**
     * Factory necessary for making the renderer available for the data type. Otherwise, the default string renderer is
     * used.
     */
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
        if (value instanceof MessageValue message) {
            super.setValue(renderMessageForCell(message));
        } else {
            // TODO: clarify whether it's possible for value to not be MessageValue at this point
            super.setValue(span("?").withClass("error-text").render());
        }
    }

    /**
     * Generate the HTML representation of the given message to be used in its cell.
     *
     * @param message the message for which to generate the HTML
     * @return a string containing the generated HTML
     */
    public static String renderMessageForCell(final MessageValue message) {
        return buildMessageHtml(message, BASE_CSS);
    }

    /**
     * Generate the HTML representation of the given message to be used in its pop-out view. Such views don't have
     * access to global CSS variables available, so we manually inject them into the CSS before rendering.
     *
     * @param message the message for which to generate the HTML
     * @return a string containing the generated HTML
     */
    public static String renderMessageForView(final MessageValue message) {
        String cssVars = StyleUtil.getCssVariables();
        String fullCss = cssVars + "\n\n" + BASE_CSS;
        return buildMessageHtml(message, fullCss);
    }

    /* --------------------------------------------------------------------- */
    /* Builders                                                              */
    /* --------------------------------------------------------------------- */

    private static String buildMessageHtml(final MessageValue message, final String cssContent) {
        DivTag messageFrame = buildMessageFrame(message);
        return html(head(style(cssContent).withType("text/css")), body(messageFrame)).render();
    }

    private static DivTag buildMessageFrame(final MessageValue message) {
        DivTag typePill = buildTypePill(message.getMessageType());
        DivTag messageBody = buildMessageBody(message);

        // frame -> content-wrapper -> (pill, body)
        return div().withClass("message-frame")
            .with(div().withClass("message-content-wrapper").with(typePill, messageBody));
    }

    private static DivTag buildMessageBody(final MessageValue message) {
        DivTag body = div().withClass("msg-body");
        List<MessageContentPart> parts = Optional.ofNullable(message.getContent()).orElse(Collections.emptyList());

        switch (message.getMessageType()) {
            case AI:
                message.getToolCalls().filter(list -> !list.isEmpty())
                    .ifPresent(calls -> body.with(buildToolCallsSection(calls)));
                addContentPartsToBody(body, parts);
                break;
            case TOOL:
                body.with(buildToolHeader(message.getName(), message.getToolCallId()));
                addContentPartsToBody(body, parts);
                break;
            case USER:
                addContentPartsToBody(body, parts);
                break;
        }
        return body;
    }

    private static DivTag buildTypePill(final MessageType type) {
        String typeClass = switch (type) {
            case USER -> "msg-pill-user";
            case AI -> "msg-pill-ai";
            case TOOL -> "msg-pill-tool";
        };
        return div(type.getLabel()).withClasses("msg-pill", typeClass);
    }

    private static void addContentPartsToBody(final DivTag container, final List<MessageContentPart> parts) {
        if (parts.isEmpty()) {
            // TODO: clarify whether we should render this.
            container.with(i("Empty message."));
        } else {
            for (int i = 0; i < parts.size(); i++) {
                if (i > 0) {
                    container.with(hr().withClass("content-separator"));
                }
                container.with(buildContentPart(parts.get(i)));
            }
        }
    }

    private static DomContent buildContentPart(final MessageContentPart part) {
        if (part instanceof TextContentPart t) {
            // TODO: add Markdown support.
            return span(t.getContent());
        }

        if (part instanceof PngContentPart imgPart) {
            byte[] raw = imgPart.getData();
            String mime = imgPart.getType().getMimeType();
            if (raw != null && raw.length > 0 && mime != null && !mime.isBlank()) {
                String uri = "data:%s;base64,%s".formatted(mime, Base64.getEncoder().encodeToString(raw));
                return img().withSrc(uri).attr("alt", "[image]").withClass("content-image");
            } else {
                return i("Unable to render image.").withClass("content-unsupported");
            }
        }

        if (part != null) {
            return i("Unsupported content part type: " + part.getType() + ".").withClass("content-unsupported");
        }

        return i("Unable to render content part.");
    }

    private static DivTag buildToolCallsSection(final List<ToolCall> toolCalls) {
        return div(div("Tool calls:").withClass("tool-calls-header"),
            div().with(toolCalls.stream().map(MessageValueRenderer::buildToolCallTag)));
    }

    private static DivTag buildToolHeader(final Optional<String> toolName, final Optional<String> toolCallId) {
        return div().with(
            toolName.filter(name -> !name.isBlank())
                .map(name -> div(span("Tool: ").withClass("tool-call-label"), span(name))).orElse(null), // Using map/orElse to make it more functional
            div(span("Tool call ID: ").withClass("tool-call-label"), span(toolCallId.orElse("-"))),
            div().withClass("tool-header-separator"));
    }

    private static DivTag buildToolCallTag(final ToolCall tc) {
        return div().withClass("tool-call").with(div(span("Tool: ").withClass("tool-call-label"), span(tc.toolName())),
            div(span("Tool call ID: ").withClass("tool-call-label"), span(tc.id())), br(), br(),
            div(span("Arguments:").withClass("tool-call-label")),
            pre((tc.arguments() == null || tc.arguments().isBlank()) ? "{}" : tc.arguments()));
    }

    // Note the use of CSS variables.
    private static final String BASE_CSS = """
            .message-frame {
                font-family: Roboto, sans-serif;
                min-width: 20%;
                max-width: 80%;
                width: 100%;
                max-width: 600px;
                box-sizing: border-box;
            }

            .message-content-wrapper {
                position: relative;
                width: 100%;
                font-size: 13px;
                font-weight: 400;
            }

            .msg-pill {
                position: relative;
                z-index: 2;
                display: inline-block;
                margin-left: 10px;
                outline: 1px solid #ffffff;
                border-radius: 16px;
                padding: 0 8px;
                min-width: 40px;
                text-align: center;
                font-size: .75em;
                font-weight: bold;
                color: #616161;
            }

            .msg-pill-user { background: var(--knime-cornflower-semi); }
            .msg-pill-ai { background: var(--knime-wood-light); }
            .msg-pill-tool { background: var(--knime-porcelain); }

            .msg-body {
                margin-top: -6px;
                border: 1px solid #d1d1d1;
                border-radius: 0 4px 4px 4px;
                background: #ffffff;
                padding: 18px 8px 12px;
                color: #333;
                overflow-wrap: break-word;
                white-space: pre-wrap;
            }

            .content-separator {
                border: 0;
                border-top: 1px dotted #eee;
                margin: 5px 0;
            }

            .content-image {
                max-width: 240px;
                height: auto;
                border-radius: 6px;
                border: 1px solid #ccc;
            }

            .content-unsupported {
                color: #555;
                font-style: italic;
            }

            .tool-calls-header {
                font-weight: bold;
                margin-bottom: 4px;
            }

            .tool-header-separator {
                border-bottom: 1px dashed #ddd;
                margin: 6px 0;
            }

            .tool-call {
                background: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                margin-bottom: 6px;
                font-size: 0.9em;
            }

            .tool-call-label {
                font-weight: bold;
            }

            .error-text {
                color: red;
            }
                        """;
}
