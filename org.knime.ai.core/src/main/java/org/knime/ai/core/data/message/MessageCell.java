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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

import org.knime.core.data.DataCell;
import org.knime.core.data.DataType;
import org.knime.core.node.util.CheckUtils;

/**
 * MessageCell is a representation of a message that can be sent to or received from an AI model.
 *
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 */
public final class MessageCell extends DataCell implements MessageValue {

    private static final long serialVersionUID = 1L;

    private final MessageType m_messageType;

    private final List<MessageContentPart> m_content;

    private final List<ToolCall> m_toolCalls;

    private final String m_toolCallId;

    private final String m_toolName;

    /**
     * Data type for MessageCell.
     */
    public static final DataType TYPE = DataType.getType(MessageCell.class);

    /**
     * Constructor for MessageCell.
     *
     * @param messageType The type of the message
     * @param content The content of the message
     */
    MessageCell(final MessageType messageType, final List<MessageContentPart> content, final List<ToolCall> toolCalls,
        final String toolCallId, final String toolName) {
        m_messageType = messageType;
        m_content = immutableCopy(content);
        m_toolCalls = toolCalls != null ? immutableCopy(toolCalls) : null;
        m_toolCallId = toolCallId;
        m_toolName = toolName;
    }

    private static <T> List<T> immutableCopy(final List<T> list) {
        return Collections.unmodifiableList(new ArrayList<>(list));
    }

    /**
     * Creates an AI message with the given content.
     * @param content the content of the message
     * @return a new MessageCell representing an AI message
     */
    public static MessageCell createAIMessageCell(final List<MessageContentPart> content) {
        return createAIMessageCell(content, null);
    }


    /**
     * Creates an AI message.
     *
     * @param content the content of the message
     * @param toolCalls the tool calls associated with the message, can be null
     * @return a new MessageCell representing an AI message
     */
    public static MessageCell createAIMessageCell(final List<MessageContentPart> content,
        final List<ToolCall> toolCalls) {
        checkContent(content);
        return new MessageCell(MessageType.AI, content, toolCalls, null, null);
    }

    private static void checkContent(final List<MessageContentPart> content) {
        CheckUtils.checkNotNull(content, "Content cannot be null");
        CheckUtils.checkArgument(!content.isEmpty(), "Content cannot be empty");
    }

    /**
     * Creates a user message cell with the given content.
     *
     * @param content the content of the message
     * @return a new MessageCell representing a user message
     */
    public static MessageCell createUserMessageCell(final List<MessageContentPart> content) {
        checkContent(content);
        return new MessageCell(MessageType.USER, content, null, null, null);
    }

    /**
     * Creates a tool message cell with the given content and tool call ID.
     *
     * @param content the content of the message
     * @param toolCallId the ID of the tool call associated with this message, must not be null
     * @param toolName
     * @return a new MessageCell representing a tool message
     */
    public static MessageCell createToolMessageCell(final List<MessageContentPart> content, final String toolCallId,
        final String toolName) {
        CheckUtils.checkNotNull(toolCallId, "Tool call ID cannot be null");
        checkContent(content);
        return new MessageCell(MessageType.TOOL, content, null, toolCallId, toolName);
    }

    @Override
    public MessageType getMessageType() {
        return m_messageType;
    }

    @Override
    public List<MessageContentPart> getContent() {
        return m_content;
    }

    @Override
    public Optional<List<ToolCall>> getToolCalls() {
        return Optional.ofNullable(m_toolCalls);
    }

    @Override
    public Optional<String> getToolCallId() {
        return Optional.ofNullable(m_toolCallId);
    }

    @Override
    public Optional<String> getToolName() {
        return Optional.ofNullable(m_toolName);
    }

    @Override
    public String toString() {
        return "MessageCell [messageType=" + m_messageType + ", content=" + m_content + ", toolCalls=" + m_toolCalls
            + ", toolCallId=" + m_toolCallId + ", toolName=" + m_toolName + "]";
    }

    @Override
    protected boolean equalsDataCell(final DataCell dc) {
        MessageCell other = (MessageCell)dc;
        return m_messageType == other.m_messageType && m_content.equals(other.m_content)
            && Objects.equals(m_toolCalls, other.m_toolCalls) && Objects.equals(m_toolCallId, other.m_toolCallId);
    }

    @Override
    public int hashCode() {
        return Objects.hash(m_messageType, m_content, m_toolCalls, m_toolCallId);
    }



}
