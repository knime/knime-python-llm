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
 *   May 30, 2025 (Adrian Nembach, KNIME GmbH, Konstanz, Germany): created
 */
package org.knime.ai.core.node.message.extract;

import java.util.List;
import java.util.function.Function;

import org.knime.ai.core.data.message.MessageValue;
import org.knime.ai.core.data.message.MessageValue.ToolCall;
import org.knime.core.data.DataCell;
import org.knime.core.data.DataColumnSpecCreator;
import org.knime.core.data.DataRow;
import org.knime.core.data.DataTableSpec;
import org.knime.core.data.DataType;
import org.knime.core.data.collection.CollectionCellFactory;
import org.knime.core.data.collection.ListCell;
import org.knime.core.data.container.ColumnRearranger;
import org.knime.core.data.container.SingleCellFactory;
import org.knime.core.data.def.StringCell;
import org.knime.core.node.BufferedDataTable;
import org.knime.core.node.ExecutionContext;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.port.PortType;
import org.knime.core.webui.node.impl.WebUINodeModel;

/**
 * Node model for the Message Part Extractor node.
 *
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 */
@SuppressWarnings("restriction")
final class MessagePartExtractorNodeModel extends WebUINodeModel<MessagePartExtractorSettings> {

    protected MessagePartExtractorNodeModel() {
        super(new PortType[]{BufferedDataTable.TYPE}, new PortType[]{BufferedDataTable.TYPE},
            MessagePartExtractorSettings.class);
    }

    @Override
    protected DataTableSpec[] configure(final DataTableSpec[] inSpecs, final MessagePartExtractorSettings modelSettings)
        throws InvalidSettingsException {
        var messageTableSpec = inSpecs[0];
        if (!messageTableSpec.containsName(modelSettings.m_messageColumn)) {
            throw new InvalidSettingsException("The specified message column '" + modelSettings.m_messageColumn
                + "' does not exist in the input table.");
        }

        return new DataTableSpec[]{createColumnRearranger(modelSettings, messageTableSpec).createSpec()};
    }

    @Override
    protected BufferedDataTable[] execute(final BufferedDataTable[] inData, final ExecutionContext exec,
        final MessagePartExtractorSettings modelSettings) throws Exception {
        var messageTable = inData[0];
        var messageTableSpec = messageTable.getDataTableSpec();
        var columRearranger = createColumnRearranger(modelSettings, messageTableSpec);

        var resultTable = exec.createColumnRearrangeTable(messageTable, columRearranger, exec);

        return new BufferedDataTable[]{resultTable};
    }

    private static ColumnRearranger createColumnRearranger(final MessagePartExtractorSettings modelSettings,
        final DataTableSpec messageTableSpec) {
        var columnRearranger = new ColumnRearranger(messageTableSpec);
        int columnIndex = messageTableSpec.findColumnIndex(modelSettings.m_messageColumn);

        if (modelSettings.m_extractRole) {
            columnRearranger.append(new PartExtractorCellFactory("Role", StringCell.TYPE,
                MessagePartExtractorNodeModel::extractRole, columnIndex));
        }
        if (modelSettings.m_extractTextParts) {
            columnRearranger
                .append(new PartExtractorCellFactory("Text Content", ListCell.getCollectionType(StringCell.TYPE),
                    MessagePartExtractorNodeModel::extractTextContent, columnIndex));
        }
        if (modelSettings.m_extractToolCalls) {
            columnRearranger
                .append(new PartExtractorCellFactory("Tool Calls", ListCell.getCollectionType(StringCell.TYPE),
                    MessagePartExtractorNodeModel::extractToolCalls, columnIndex));
        }
        if (modelSettings.m_extractToolCallIds) {
            columnRearranger.append(new PartExtractorCellFactory("Tool Call ID", StringCell.TYPE,
                MessagePartExtractorNodeModel::extractToolCallId, columnIndex));
        }

        if (!modelSettings.m_keepOriginalColumn) {
            columnRearranger.remove(columnIndex);
        }
        return columnRearranger;
    }

    private static final class PartExtractorCellFactory extends SingleCellFactory {

        private final int m_messageColumnIndex;

        private final Function<MessageValue, DataCell> m_extractor;

        PartExtractorCellFactory(final String columnName, final DataType contentType,
            final Function<MessageValue, DataCell> extractor, final int messageColumnIndex) {
            super(true, new DataColumnSpecCreator(columnName, contentType).createSpec());
            m_extractor = extractor;
            m_messageColumnIndex = messageColumnIndex;
        }

        @Override
        public DataCell getCell(final DataRow row) {
            var messageCell = row.getCell(m_messageColumnIndex);
            if (messageCell.isMissing()) {
                return DataType.getMissingCell();
            }
            var message = (MessageValue)messageCell;
            return m_extractor.apply(message);
        }
    }

    private static StringCell extractRole(final MessageValue message) {
        return new StringCell(message.getMessageType().name());
    }

    private static final DataCell extractTextContent(final MessageValue message) {
        return CollectionCellFactory
            .createListCell(message.getContent().stream().filter(part -> "text".equals(part.getType()))
                .map(part -> new StringCell(new String(part.getData()))).toList());
    }

    private static DataCell extractToolCallId(final MessageValue message) {
        return message.getToolCallId().map(s -> (DataCell)new StringCell(s)).orElseGet(DataType::getMissingCell);
    }

    private static DataCell extractToolCalls(final MessageValue message) {
        var toolCalls = message.getToolCalls();
        return toolCalls.map(list -> toListCell(list, MessagePartExtractorNodeModel::toJsonCell))
            .orElseGet(DataType::getMissingCell);
    }

    private static <T> DataCell toListCell(final List<T> list, final Function<T, DataCell> mapper) {
        return CollectionCellFactory.createListCell(list.stream().map(mapper).toList());
    }

    private static DataCell toJsonCell(final ToolCall toolCall) {
        return new StringCell("{\"toolName\":\"" + toolCall.toolName() + "\", \"id\":\"" + toolCall.id()
            + "\", \"arguments\":" + toolCall.arguments() + "}");
    }

}
