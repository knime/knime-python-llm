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
import java.util.function.Supplier;

import org.knime.ai.core.data.message.MessageValue.MessageContentPart.MessageContentPartType;
import org.knime.core.data.DataCell;
import org.knime.core.data.v2.ReadValue;
import org.knime.core.table.access.ListAccess.ListReadAccess;
import org.knime.core.table.access.StringAccess.StringReadAccess;
import org.knime.core.table.access.StructAccess.StructReadAccess;
import org.knime.core.table.access.VarBinaryAccess.VarBinaryReadAccess;

/**
 * {@link ReadValue} implementation for {@link MessageValue}.
 *
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 */
final class MessageReadValue implements ReadValue, MessageValue {

    private final Supplier<MessageType> m_typeReader;

    private final Supplier<Optional<List<ToolCall>>> m_toolCallsReader;

    private final Supplier<Optional<String>> m_toolCallIdReader;

    private final Supplier<Optional<String>> m_nameReader;

    private final Supplier<List<MessageContentPart>> m_contentReader;

    MessageReadValue(final StructReadAccess access) {
        StringReadAccess typeAccess = access.getAccess(0);
        m_typeReader = ValueFactoryUtils.chain(typeAccess::getStringValue, MessageType::valueOf);
        ListReadAccess contentAccess = access.getAccess(1);
        m_contentReader =
                ValueFactoryUtils.readList(contentAccess, createContentPartReader(contentAccess.getAccess()));
        ListReadAccess toolCallsAccess = access.getAccess(2);
        m_toolCallsReader = ValueFactoryUtils.readOptional(toolCallsAccess,
            ValueFactoryUtils.readList(toolCallsAccess, createToolCallReader(toolCallsAccess.getAccess())));
        StringReadAccess toolCallIdAccess = access.getAccess(3);
        m_toolCallIdReader = ValueFactoryUtils.readOptional(toolCallIdAccess, toolCallIdAccess::getStringValue);
        StringReadAccess toolNameAccess = access.getAccess(4);
        m_nameReader = ValueFactoryUtils.readOptional(toolNameAccess, toolNameAccess::getStringValue);
    }

    @Override
    public MessageType getMessageType() {
        return m_typeReader.get();
    }

    @Override
    public Optional<String> getToolCallId() {
        return m_toolCallIdReader.get();
    }

    @Override
    public DataCell getDataCell() {
        return new MessageCell(//
            m_typeReader.get(), //
            m_contentReader.get(), //
            m_toolCallsReader.get().orElse(null), //
            m_toolCallIdReader.get().orElse(null), //
            m_nameReader.get().orElse(null));
    }

    @Override
    public List<MessageContentPart> getContent() {
        return m_contentReader.get();
    }

    @Override
    public Optional<List<ToolCall>> getToolCalls() {
        return m_toolCallsReader.get();
    }

    @Override
    public Optional<String> getName() {
        return m_nameReader.get();
    }

    private static Supplier<ToolCall> createToolCallReader(final StructReadAccess access) {
        StringReadAccess idAccess = access.getAccess(0);
        StringReadAccess nameAccess = access.getAccess(1);
        StringReadAccess argumentsAccess = access.getAccess(2);
        return () -> new ToolCall(nameAccess.getStringValue(), idAccess.getStringValue(),
            argumentsAccess.getStringValue());
    }

    private static Supplier<MessageContentPart> createContentPartReader(final StructReadAccess access) {
        StringReadAccess typeAccess = access.getAccess(0);
        VarBinaryReadAccess dataAccess = access.getAccess(1);
        // TODO use deserializer instead?
        return () -> readContentPart(typeAccess.getStringValue(), dataAccess.getByteArray());
    }

    private static MessageContentPart readContentPart(final String type, final byte[] data) {
        var partType = MessageContentPartType.fromId(type);
        return switch (partType) {
            case TEXT -> new TextContentPart(new String(data));
            case PNG -> new PngContentPart(data);
            // Add other content types as needed
            default -> throw new IllegalArgumentException("Unknown content type: " + type);
        };
    }

}