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
import java.util.List;
import java.util.Optional;
import java.util.function.Supplier;

import org.knime.ai.core.data.message.MessageValue.ToolCall;
import org.knime.core.data.DataCell;
import org.knime.core.data.v2.ReadValue;
import org.knime.core.data.v2.ValueFactory;
import org.knime.core.data.v2.WriteValue;
import org.knime.core.table.access.ListAccess.ListReadAccess;
import org.knime.core.table.access.ReadAccess;
import org.knime.core.table.access.StringAccess.StringReadAccess;
import org.knime.core.table.access.StructAccess.StructReadAccess;
import org.knime.core.table.access.StructAccess.StructWriteAccess;
import org.knime.core.table.schema.DataSpec;
import org.knime.core.table.schema.ListDataSpec;
import org.knime.core.table.schema.StringDataSpec;
import org.knime.core.table.schema.StructDataSpec;
import org.knime.core.table.schema.VarBinaryDataSpec;

/**
 *
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 */
public final class MessageValueFactory implements ValueFactory<StructReadAccess, StructWriteAccess> {

    @Override
    public DataSpec getSpec() {
        return new StructDataSpec(//
            StringDataSpec.INSTANCE, // type
            new ListDataSpec(// content
                new StructDataSpec(// content
                    StringDataSpec.INSTANCE, // content type
                    VarBinaryDataSpec.INSTANCE // content data
                )//
            ), //
            new ListDataSpec(// tool calls
                new StructDataSpec(// tool call
                    StringDataSpec.INSTANCE, // id
                    StringDataSpec.INSTANCE, // name
                    StringDataSpec.INSTANCE // arguments
                )//
            ), //
            StringDataSpec.INSTANCE // tool call id (only present for tool messages)
        );

    }

    @Override
    public ReadValue createReadValue(final StructReadAccess access) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public WriteValue<?> createWriteValue(final StructWriteAccess access) {
        // TODO Auto-generated method stub
        return null;
    }

    private static final class MessageReadValue implements ReadValue, MessageValue {

        private final StringReadAccess m_typeAccess;

        private final ListReadAccess m_contentAccess;

        private final OptionalReader<List<ToolCall>> m_toolCallsReader;

        private final OptionalReader<String> m_toolCallIdReader;

        protected MessageReadValue(final StructReadAccess access) {
            m_typeAccess = access.getAccess(0);
            m_contentAccess = access.getAccess(1);
            ListReadAccess toolCallsAccess = access.getAccess(2);
            m_toolCallsReader = new OptionalReader<>(toolCallsAccess,
                    new ListReader<>(toolCallsAccess, new ToolCallReader(toolCallsAccess.getAccess())));
            StringReadAccess toolCallIdAccess = access.getAccess(3);
            m_toolCallIdReader = new OptionalReader<>(toolCallIdAccess, toolCallIdAccess::getStringValue);
        }

        @Override
        public MessageType getMessageType() {
            return MessageType.valueOf(m_typeAccess.getStringValue());
        }

        @Override
        public Optional<String> getToolCallId() {
            return m_toolCallIdReader.get();
        }

        @Override
        public DataCell getDataCell() {
            // TODO Auto-generated method stub
            return null;
        }

        @Override
        public List<MessageContentPart> getContent() {
            // TODO Auto-generated method stub
            return null;
        }

        @Override
        public Optional<List<ToolCall>> getToolCalls() {
            return null;
        }
    }

    private static final class OptionalReader<T> implements Supplier<Optional<T>> {
        private final ReadAccess m_access;

        private final Supplier<T> m_valueReader;

        OptionalReader(final ReadAccess access, final Supplier<T> valueReader) {
            m_access = access;
            m_valueReader = valueReader;
        }

        @Override
        public Optional<T> get() {
            if (m_access.isMissing()) {
                return Optional.empty();
            }
            return Optional.of(m_valueReader.get());
        }
    }

    private static final class ListReader<T> implements Supplier<List<T>> {
        private final ListReadAccess m_listAccess;

        private final Supplier<T> m_itemReader;

        ListReader(final ListReadAccess listAccess, final Supplier<T> itemReader) {
            m_listAccess = listAccess;
            m_itemReader = itemReader;
        }

        @Override
        public List<T> get() {
            var list = new ArrayList<T>();
            var numItems = m_listAccess.size();
            for (int i = 0; i < numItems; i++) {
                m_listAccess.setIndex(i);
                list.add(m_itemReader.get());
            }
            return list;
        }

    }

    private static final class ToolCallReader implements Supplier<ToolCall> {

        private final StringReadAccess m_idAccess;

        private final StringReadAccess m_nameAccess;

        private final StringReadAccess m_argumentsAccess;

        ToolCallReader(final StructReadAccess access) {
            m_idAccess = access.getAccess(0);
            m_nameAccess = access.getAccess(1);
            m_argumentsAccess = access.getAccess(2);
        }

        @Override
        public ToolCall get() {
            return new ToolCall(m_nameAccess.getStringValue(), m_idAccess.getStringValue(),
                m_argumentsAccess.getStringValue());
        }
    }

    private static final class MessageContentReadValue {

        MessageContentReadValue(final StructReadAccess access) {
            // TODO Auto-generated constructor stub
        }
    }

}
