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
import static j2html.TagCreator.div;
import static j2html.TagCreator.head;
import static j2html.TagCreator.hr;
import static j2html.TagCreator.html;
import static j2html.TagCreator.i;
import static j2html.TagCreator.img;
import static j2html.TagCreator.pre;
import static j2html.TagCreator.rawHtml;
import static j2html.TagCreator.span;
import static j2html.TagCreator.style;

import java.util.Base64;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

import org.commonmark.parser.Parser;
import org.commonmark.renderer.html.HtmlRenderer;
import org.knime.ai.core.data.message.MessageValue;
import org.knime.ai.core.data.message.MessageValue.MessageContentPart;
import org.knime.ai.core.data.message.MessageValue.ToolCall;
import org.knime.ai.core.data.message.PngContentPart;
import org.knime.ai.core.data.message.TextContentPart;
import org.knime.core.data.DataColumnSpec;
import org.knime.core.data.renderer.AbstractDataValueRendererFactory;
import org.knime.core.data.renderer.DataValueRenderer;
import org.knime.core.data.renderer.DefaultDataValueRenderer;
import org.knime.core.webui.node.view.table.data.render.DataCellContentType;
import org.owasp.html.HtmlPolicyBuilder;
import org.owasp.html.PolicyFactory;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import com.fasterxml.jackson.dataformat.yaml.YAMLGenerator;

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

    // JSON parsing for Tool Call Arguments.
    private static final ObjectMapper JSON_MAPPER = new ObjectMapper();

    private static final ObjectWriter PRETTY_PRINTER =
        new ObjectMapper(createYamlFactory()).writerWithDefaultPrettyPrinter();

    private static YAMLFactory createYamlFactory() {
        var yamlFactory = new YAMLFactory();
        yamlFactory.disable(YAMLGenerator.Feature.WRITE_DOC_START_MARKER);
        return yamlFactory;
    }

    /**
     * Pretty-prints a JSON string.
     * @param json the raw JSON string
     * @return a formatted JSON string, or the original string if parsing fails.
     */
    private static String prettyPrintJson(final String json) {
        if (json == null || json.isBlank()) {
            return "";
        }
        try {
            Object jsonObject = JSON_MAPPER.readValue(json, Object.class);
            return PRETTY_PRINTER.writeValueAsString(jsonObject);
        } catch (JsonProcessingException e) {
            // If it's not valid JSON, return the original content
            return json;
        }
    }

    // MD to HTML parser
    private static final Parser MARKDOWN_PARSER = Parser.builder().build();
    private static final HtmlRenderer HTML_RENDERER = HtmlRenderer.builder().build();

    private static final PolicyFactory HTML_SANITIZER_POLICY = new HtmlPolicyBuilder()
            .allowCommonInlineFormattingElements()
            .allowCommonBlockElements()
            .allowElements("h1", "h2", "h3", "h4", "h5", "h6")
            .allowElements("pre", "code")

            // Controversial (check with Peter?): allow tables
            .allowElements("table", "thead", "tbody", "tfoot", "tr", "th", "td")
            .allowAttributes("colspan", "rowspan").onElements("th", "td")
            .allowAttributes("align").onElements("table", "tr", "th", "td")

            .toFactory();

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
    /* Message type icons                                                    */
    /* --------------------------------------------------------------------- */

    // Taken directly from webapps-common
    private static final String AI_ICON =
        """
            data:image/svg+xml,%3Csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20viewBox=%270%200%2032%2032%27%20stroke=%27%23000%27%20fill=%27none%27%20stroke-linejoin=%27round%27%3E%3Cpath%20d=%27M19.632%2019.8495L19.5523%2020.0523L19.3495%2020.132L15.8674%2021.5L19.3495%2022.868L19.5523%2022.9477L19.632%2023.1505L21%2026.6326L22.368%2023.1505L22.4477%2022.9477L22.6505%2022.868L26.1326%2021.5L22.6505%2020.132L22.4477%2020.0523L22.368%2019.8495L21%2016.3674L19.632%2019.8495Z%27/%3E%3Cpath%20d=%27M3.5%2017V30H29.5V4H16.5%27/%3E%3Cpath%20d=%27M8.23219%208.40239L8.15424%208.65424L7.90239%208.73219L2.191%2010.5L7.90239%2012.2678L8.15424%2012.3458L8.23219%2012.5976L10%2018.309L11.7678%2012.5976L11.8458%2012.3458L12.0976%2012.2678L17.809%2010.5L12.0976%208.73219L11.8458%208.65424L11.7678%208.40239L10%202.691L8.23219%208.40239Z%27/%3E%3C/svg%3E
            """;

    private static final String USER_ICON =
        """
            data:image/svg+xml,%3Csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20viewBox=%270%200%2032%2032%27%20stroke=%27%23000%27%20fill=%27none%27%20stroke-linejoin=%27round%27%3E%3Cpath%20d=%27M6.4%2027.1c0-3.5%204.1-6.4%209.1-6.4s9.1%202.9%209.1%206.4%27/%3E%3Ccircle%20cx=%2715.5%27%20cy=%2711.2%27%20r=%276.3%27/%3E%3C/svg%3E
            """;

    private static final String TOOL_ICON =
        """
            data:image/svg+xml,%3Csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20viewBox=%270%200%2032%2032%27%20fill=%27none%27%20stroke=%27%23000%27%20stroke-linejoin=%27round%27%3E%3Cpath%20d=%27M4.857%2023.007c-1.142%201.141-1.142%202.995%200%204.137%201.142%201.142%202.995%201.141%204.136%200l9.318-9.754c2.302.652%205.879%20.073%207.69-1.74%201.642-1.64%202.272-3.907%201.892-6.03l-3.27%203.273-4.277-1.238-1.237-4.276%203.27-3.272v-.001c-2.122-.377-4.389.253-6.028%201.894-1.811%201.81-2.392%205.385-1.742%207.686l-9.752%209.321z%27/%3E%3C/svg%3E
            """;

    /* --------------------------------------------------------------------- */
    /* Builders                                                              */
    /* --------------------------------------------------------------------- */

    private static String buildMessageHtml(final MessageValue message, final String cssContent) {
        DivTag messageFrame = buildMessageFrame(message);
        return html(head(style(cssContent).withType("text/css")), body(messageFrame)).render();
    }

    private static DivTag buildMessageFrame(final MessageValue message) {
        DivTag messageHeader = buildMessageHeader(message);
        DivTag messageBody = buildMessageBody(message);

        return div().withClass("message-frame").with(messageHeader, messageBody);
    }

    private static DivTag buildMessageHeader(final MessageValue message) {
        final String iconUri;
        final String label;

        switch (message.getMessageType()) {
            case AI:
                iconUri = AI_ICON;
                label = "AI";
                break;
            case USER:
                iconUri = USER_ICON;
                label = "User"; // TODO: use Human instead?
                break;
            case TOOL:
                iconUri = TOOL_ICON;
                label = message.getName().orElse("Tool");
                break;
            default:
                // Should not be reachable
                return div();
        }

        DomContent icon = img().withSrc(iconUri).withClass("msg-header-icon").attr("alt", label + " message icon");
        DomContent text = span(label).withClass("msg-header-label");

        return div(icon, text).withClass("msg-header");
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
                addContentPartsToBody(body, parts);
                break;
            case USER:
                addContentPartsToBody(body, parts);
                break;
        }
        return body;
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
            String rawHtml = HTML_RENDERER.render(MARKDOWN_PARSER.parse(t.getContent()));
            String sanitizedHtml = HTML_SANITIZER_POLICY.sanitize(rawHtml);

            return rawHtml(sanitizedHtml);
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
        return div(div().with(toolCalls.stream().map(MessageValueRenderer::buildToolCallTag)));
    }

    private static DivTag buildToolCallTag(final ToolCall tc) {
        DivTag header = div().withClass("tool-call-header").with(
            img().withSrc(TOOL_ICON).withClass("tool-call-icon"),
            span(tc.toolName()).withClass("tool-call-name"));

        DomContent args = pre(prettyPrintJson(tc.arguments())).withClass("tool-call-args");

        return div().withClass("tool-call").with(header, args);
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
                font-size: 13px;
                font-weight: 400;
            }

            .msg-header {
                display: flex;
                align-items: center;
                gap: 4px;
                margin-bottom: 8px;
            }

            .msg-header-icon {
                width: 16px;
                height: 16px;
            }

            .msg-header-label {
                color: #333;
            }

            .msg-body {
                color: #333;
                overflow-wrap: break-word;
                white-space: pre-wrap;
                font-weight: 400;
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
                margin-bottom: 80px;
            }

            .tool-call {
                margin-bottom: 10px;
                display: flex;
                flex-direction: column;
                gap: 5px;
            }

            .tool-call-icon {
                width: 16px;
                height: 16px;
            }

            .tool-call-name {
                font-size: 12px;
            }

            .tool-call-args {
                background: var(--knime-porcelain);
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Menlo', 'Consolas', monospace;
                font-size: 10px;
                white-space: pre-wrap;
                word-break: break-all;
                margin: 0;
            }

            .error-text {
                color: red;
            }
                        """;
}
