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
 *   May 23, 2025 (Adrian Nembach, KNIME GmbH, Konstanz, Germany): created
 */
package org.knime.ai.core.data.message;

import java.util.List;
import java.util.Optional;

import org.knime.core.data.DataValue;
import org.knime.core.data.ExtensibleUtilityFactory;

/**
 * Represents a message to or from an AI model.
 *
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 */
public interface MessageValue extends DataValue {

    @SuppressWarnings("javadoc")
    UtilityFactory UTILITY = new ExtensibleUtilityFactory(MessageValue.class) {

        @Override
        public String getName() {
            return "Message";
        }
    };

    /**
     * Enumeration of the different types of messages.
     *
     * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
     */
    public enum MessageType {
            /**
             * User message, typically input from the user.
             */
            USER("User"),
            /**
             * A message that is the result of a tool call.
             */
            TOOL("Tool"),
            /**
             * AI message, typically a response from the AI model.
             */
            AI("AI");

        private final String m_label;

        MessageType(final String label) {
            this.m_label = label;
        }

        @Override
        public String toString() {
            return m_label;
        }

        /**
         * @return the label of the message type, e.g., "User", "Tool", or "AI"
         */
        public String getLabel() {
            return m_label;
        }
    }

    /**
     * Represents a part of the message content.
     *
     * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
     */
    public interface MessageContentPart {

        /**
         * The type of content part, e.g., "text", "image/png", etc.
         *
         * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
         */
        enum MessageContentPartType {
            TEXT("text/markdown"), PNG("image/png");

            private final String m_id;

            private final String m_mimeType;

            private MessageContentPartType(final String mimeType) {
                m_id = name().toLowerCase();
                m_mimeType = mimeType;
            }

            public String getId() {
                return m_id;
            }

            public String getMimeType() {
                return m_mimeType;
            }

            public static MessageContentPartType fromId(final String id) {
                for (MessageContentPartType type : values()) {
                    if (type.getId().equals(id)) {
                        return type;
                    }
                }
                throw new IllegalArgumentException("Unknown content part type: " + id);
            }

        }

        /**
         * @return the type of the content part, e.g., "text", "image", etc.
         */
        MessageContentPartType getType();

        /**
         * @return the data of the content part as a byte array.
         */
        byte[] getData();
    }

    /**
     * Represents a tool call within an AI message.
     *
     * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
     * @param toolName the name of the tool being called
     * @param id the unique identifier for the tool call
     * @param arguments the arguments passed to the tool in JSON format
     */
    public record ToolCall(String toolName, String id, String arguments) {

    }

    /**
     * @return the type of the message, e.g., USER, TOOL, or AI
     */
    MessageType getMessageType();

    /**
     * @return the content of the message, which can be a list of different content parts
     */
    List<MessageContentPart> getContent();

    /**
     * @return an optional list of tool calls associated with the message, if any
     */
    Optional<List<ToolCall>> getToolCalls();

    /**
     * @return an optional ID of the tool call associated with this message, if any
     */
    Optional<String> getToolCallId();

    /**
     * @return an optional name for the participant. Provides the model information to differentiate between
     *         participants of the same role.
     */
    Optional<String> getName();

}
