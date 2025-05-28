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

/**
 * Represents a message to or from an AI model.
 *
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 */
public interface MessageValue extends DataValue {

    /**
     * Enumeration of the different types of messages.
     *
     * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
     */
    public enum MessageType {
            USER, TOOL, AI
    }

    /**
     * Represents a part of the message content.
     *
     * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
     */
    public interface MessageContentPart {

        /**
         * @return the type of the content part, e.g., "text", "image", etc.
         */
        String getType();

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

}
