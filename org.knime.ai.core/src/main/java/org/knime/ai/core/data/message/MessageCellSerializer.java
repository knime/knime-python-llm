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
 *   May 28, 2025 (Adrian Nembach, KNIME GmbH, Konstanz, Germany): created
 */
package org.knime.ai.core.data.message;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.knime.ai.core.data.message.MessageValue.MessageContentPart;
import org.knime.ai.core.data.message.MessageValue.MessageType;
import org.knime.ai.core.data.message.MessageValue.ToolCall;
import org.knime.core.data.DataCellDataInput;
import org.knime.core.data.DataCellDataOutput;
import org.knime.core.data.DataCellSerializer;

/**
 * {@link DataCellSerializer} for {@link MessageCell}.
 *
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 */
public final class MessageCellSerializer implements DataCellSerializer<MessageCell> {

    @Override
    public void serialize(final MessageCell cell, final DataCellDataOutput output) throws IOException {
        output.writeUTF(cell.getMessageType().name());
        writeContentParts(cell.getContent(), output);
        writeToolCalls(cell.getToolCalls().orElse(null), output);
        output.writeUTF(cell.getToolCallId().orElse(""));
    }

    @Override
    public MessageCell deserialize(final DataCellDataInput input) throws IOException {
        MessageType messageType = MessageType.valueOf(input.readUTF());
        List<MessageContentPart> content = readContentParts(input);
        List<ToolCall> toolCalls = readToolCalls(input);
        String toolCallId = input.readUTF();
        return new MessageCell(messageType, content, toolCalls.isEmpty() ? null : toolCalls,
            toolCallId.isEmpty() ? null : toolCallId);
    }

    private static void writeContentParts(final List<MessageContentPart> content, final DataCellDataOutput output)
        throws IOException {
        output.writeInt(content.size());
        for (MessageContentPart part : content) {
            output.writeUTF(part.getType());
            byte[] data = part.getData();
            output.write(data.length);
            output.write(data);
        }
    }

    private static List<MessageContentPart> readContentParts(final DataCellDataInput input) throws IOException {
        int size = input.readInt();
        List<MessageContentPart> content = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            String type = input.readUTF();
            var data = new byte[input.readInt()];
            input.readFully(data);
            content.add(createContentPart(type, data));
        }
        return content;
    }

    // TODO unify content part creation
    private static MessageContentPart createContentPart(final String type, final byte[] value) {
        switch (type) {
            case "text":
                return new TextContentPart(new String(value));
            case "image":
                return new ImageContentPart(value);
            default:
                throw new IllegalArgumentException("Unknown content type: " + type);
        }
    }

    private static void writeToolCalls(final List<ToolCall> toolCalls, final DataCellDataOutput output)
        throws IOException {
        if (toolCalls == null) {
            output.writeInt(0);
            return;
        }
        output.writeInt(toolCalls.size());
        for (ToolCall toolCall : toolCalls) {
            output.writeUTF(toolCall.toolName());
            output.writeUTF(toolCall.id());
            output.writeUTF(toolCall.arguments());
        }
    }

    private static List<ToolCall> readToolCalls(final DataCellDataInput input) throws IOException {
        int size = input.readInt();
        List<ToolCall> toolCalls = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            String toolName = input.readUTF();
            String toolId = input.readUTF();
            String toolParameters = input.readUTF();
            toolCalls.add(new ToolCall(toolName, toolId, toolParameters));
        }
        return toolCalls;
    }
}

