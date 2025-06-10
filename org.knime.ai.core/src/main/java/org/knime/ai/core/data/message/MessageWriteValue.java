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
 *   Jun 11, 2025 (Adrian Nembach, KNIME GmbH, Konstanz, Germany): created
 */
package org.knime.ai.core.data.message;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

import org.knime.ai.core.data.message.MessageValue.MessageContentPart;
import org.knime.core.data.v2.WriteValue;
import org.knime.core.table.access.ListAccess.ListWriteAccess;
import org.knime.core.table.access.StringAccess.StringWriteAccess;
import org.knime.core.table.access.StructAccess.StructWriteAccess;
import org.knime.core.table.access.VarBinaryAccess.VarBinaryWriteAccess;

/**
 * {@link WriteValue} implementation for {@link MessageValue}.
 *
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 */
final class MessageWriteValue implements WriteValue<MessageValue> {

    private final Consumer<String> m_typeWriter;

    private final Consumer<List<MessageContentPart>> m_contentWriter;

    private final Consumer<Optional<List<MessageValue.ToolCall>>> m_toolCallsWriter;

    private final Consumer<Optional<String>> m_toolCallIdWriter;

    private final Consumer<Optional<String>> m_nameWriter;

    MessageWriteValue(final StructWriteAccess access) {
        m_typeWriter = access.<StringWriteAccess>getWriteAccess(0)::setStringValue;
        ListWriteAccess contentAccess = access.getWriteAccess(1);
        m_contentWriter = ValueFactoryUtils.writeList(contentAccess,
            createContentPartWriter(contentAccess.getWriteAccess()));
        ListWriteAccess toolCallsAccess = access.getWriteAccess(2);
        m_toolCallsWriter = ValueFactoryUtils.writeOptional(toolCallsAccess,
            ValueFactoryUtils.writeList(toolCallsAccess, createToolCallWriter(toolCallsAccess.getWriteAccess())));
        StringWriteAccess toolCallIdAccess = access.getWriteAccess(3);
        m_toolCallIdWriter = ValueFactoryUtils.writeOptional(toolCallIdAccess, toolCallIdAccess::setStringValue);
        StringWriteAccess nameAccess = access.getWriteAccess(4);
        m_nameWriter = ValueFactoryUtils.writeOptional(nameAccess, nameAccess::setStringValue);

    }

    @Override
    public void setValue(final MessageValue value) {
        m_typeWriter.accept(value.getMessageType().name());
        m_contentWriter.accept(value.getContent());
        m_toolCallsWriter.accept(value.getToolCalls());
        m_toolCallIdWriter.accept(value.getToolCallId());
        m_nameWriter.accept(value.getName());
    }

    private static Consumer<MessageContentPart> createContentPartWriter(final StructWriteAccess access) {
        StringWriteAccess typeAccess = access.getWriteAccess(0);
        VarBinaryWriteAccess dataAccess = access.getWriteAccess(1);
        return (part) -> {
            typeAccess.setStringValue(part.getType().getId());
            dataAccess.setByteArray(part.getData());
        };
    }

    private static Consumer<MessageValue.ToolCall> createToolCallWriter(final StructWriteAccess access) {
        StringWriteAccess idAccess = access.getWriteAccess(0);
        StringWriteAccess nameAccess = access.getWriteAccess(1);
        StringWriteAccess argumentsAccess = access.getWriteAccess(2);
        return (toolCall) -> {
            idAccess.setStringValue(toolCall.id());
            nameAccess.setStringValue(toolCall.toolName());
            argumentsAccess.setStringValue(toolCall.arguments());
        };
    }

}