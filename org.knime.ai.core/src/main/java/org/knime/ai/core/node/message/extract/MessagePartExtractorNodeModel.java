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

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.BinaryOperator;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.Predicate;
import java.util.function.Supplier;
import java.util.stream.IntStream;

import org.knime.ai.core.data.message.MessageValue;
import org.knime.ai.core.data.message.MessageValue.MessageContentPart;
import org.knime.ai.core.data.message.MessageValue.ToolCall;
import org.knime.core.data.DataCell;
import org.knime.core.data.DataColumnSpec;
import org.knime.core.data.DataColumnSpecCreator;
import org.knime.core.data.DataRow;
import org.knime.core.data.DataTableSpec;
import org.knime.core.data.DataType;
import org.knime.core.data.container.AbstractCellFactory;
import org.knime.core.data.container.CellFactory;
import org.knime.core.data.container.ColumnRearranger;
import org.knime.core.data.container.SingleCellFactory;
import org.knime.core.data.def.StringCell;
import org.knime.core.data.image.png.PNGImageCellFactory;
import org.knime.core.data.json.JSONCellFactory;
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
        int messageColumnIndex = messageTableSpec.findColumnIndex(modelSettings.m_messageColumn);
        var splitterFactories = createCellSplitterFactories(modelSettings, messageColumnIndex);
        // If any splitterFactory is stateful (i.e., depends on data), return null DataTableSpec
        boolean hasStateful = splitterFactories.stream().anyMatch(f -> f.isStateful());
        if (hasStateful) {
            return new DataTableSpec[]{null};
        }
        // Otherwise, safe to simulate scan with empty state
        List<CellFactory> cellFactories = splitterFactories.stream().map(Supplier::get).toList();
        var columnRearranger = createColumnRearranger(modelSettings, messageTableSpec, cellFactories);
        return new DataTableSpec[]{columnRearranger.createSpec()};
    }

    @Override
    protected BufferedDataTable[] execute(final BufferedDataTable[] inData, final ExecutionContext exec,
        final MessagePartExtractorSettings modelSettings) throws Exception {
        var messageTable = inData[0];
        var messageTableSpec = messageTable.getDataTableSpec();
        int messageColumnIndex = messageTableSpec.findColumnIndex(modelSettings.m_messageColumn);
        var splitterFactories = createCellSplitterFactories(modelSettings, messageColumnIndex);
        // Scan the data for extractors that need it
        for (var row : messageTable) {
            var cell = row.getCell(messageColumnIndex);
            for (var splitter : splitterFactories) {
                splitter.accept(cell);
            }
        }
        List<CellFactory> cellFactories = splitterFactories.stream().map(Supplier::get).toList();
        var columnRearranger = createColumnRearranger(modelSettings, messageTableSpec, cellFactories);
        var resultTable = exec.createColumnRearrangeTable(messageTable, columnRearranger, exec);
        return new BufferedDataTable[]{resultTable};
    }

    private static ColumnRearranger createColumnRearranger(final MessagePartExtractorSettings modelSettings,
        final DataTableSpec messageTableSpec, final List<CellFactory> cellFactories) {
        var columnRearranger = new ColumnRearranger(messageTableSpec);
        int columnIndex = messageTableSpec.findColumnIndex(modelSettings.m_messageColumn);
        for (CellFactory factory : cellFactories) {
            columnRearranger.append(factory);
        }
        if (!modelSettings.m_keepOriginalColumn) {
            columnRearranger.remove(columnIndex);
        }
        return columnRearranger;
    }

    private static List<CellSplitterFactory<?>> createCellSplitterFactories(final MessagePartExtractorSettings settings,
        final int messageColumnIndex) {
        var list = new ArrayList<CellSplitterFactory<?>>();
        if (settings.m_extractRole) {
            list.add(
                createStatelessCellSplitterFactory(() -> new PartExtractorSingleCellFactory(settings.m_roleColumnName,
                    StringCell.TYPE, MessagePartExtractorNodeModel::extractRole, messageColumnIndex)));
        }
        if (settings.m_extractTextParts) {
            list.add(createMultiCellSplitterFactory(createContentCounter("text"),
                createColumnSpecCreator(settings.m_textPartsPrefix, StringCell.TYPE), messageColumnIndex,
                createContentPartExtractor("text", data -> new StringCell(new String(data)))));
        }
        if (settings.m_extractImageParts) {
            list.add(createMultiCellSplitterFactory(createContentCounter("image"),
                createColumnSpecCreator(settings.m_imagePartsPrefix, PNGImageCellFactory.TYPE), messageColumnIndex,
                createContentPartExtractor("image", PNGImageCellFactory::create)));
        }
        if (settings.m_extractToolCalls) {
            list.add(createMultiCellSplitterFactory(
                MessagePartExtractorNodeModel::countToolCalls,
                createColumnSpecCreator(settings.m_toolCallsPrefix, JSONCellFactory.TYPE),
                messageColumnIndex,
                MessagePartExtractorNodeModel::extractToolCalls
            ));
        }
        if (settings.m_extractToolCallIds) {
            list.add(createStatelessCellSplitterFactory(
                () -> new PartExtractorSingleCellFactory(settings.m_toolCallIdColumnName, StringCell.TYPE,
                    MessagePartExtractorNodeModel::extractToolCallId, messageColumnIndex)));
        }
        return list;
    }

    private static IntFunction<DataColumnSpec> createColumnSpecCreator(final String prefix, final DataType type) {
        return j -> new DataColumnSpecCreator(prefix + (j + 1), type).createSpec();
    }

    private static CellSplitterFactory<Integer> createMultiCellSplitterFactory(
        final Function<DataCell, Integer> numCellsCalculator, final IntFunction<DataColumnSpec> columnSpecCreator,
        final int messageColumnIndex, final Function<MessageValue, DataCell[]> cellExtractor) {
        return new CellSplitterFactory<>(0, numCellsCalculator, Integer::max,
            i -> new MultiCellFactory(IntStream.range(0, i).mapToObj(columnSpecCreator).toArray(DataColumnSpec[]::new),
                r -> Optional.of(r.getCell(messageColumnIndex)).filter(Predicate.not(DataCell::isMissing))//
                    .map(MessageValue.class::cast)//
                    .map(cellExtractor)//
                    .map(c -> padWithMissing(c, i))//
                    .orElseGet(() -> padWithMissing(new DataCell[0], i))));
    }

    private static CellSplitterFactory<?>
        createStatelessCellSplitterFactory(final Supplier<CellFactory> cellFactorySupplier) {
        return new CellSplitterFactory<>(null, noopExtractor(), noopAggregator(), c -> cellFactorySupplier.get(),
            false);
    }

    private static Function<DataCell, Integer> createContentCounter(final String contentType) {
        return cell -> {
            if (cell.isMissing()) {
                return 0;
            }
            var message = (MessageValue)cell;
            return (int)message.getContent().stream()//
                .filter(part -> contentType.equals(part.getType()))//
                .count();
        };
    }


    private static Integer countToolCalls(final DataCell cell) {
        if (cell.isMissing()) {
            return 0;
        }
        var message = (MessageValue)cell;
        return message.getToolCalls().map(List::size).orElse(0);
    }

    private static DataCell[] extractToolCalls(final MessageValue message) {
        return message.getToolCalls()
            .map(list -> list.stream().map(MessagePartExtractorNodeModel::toJsonCell).toArray(DataCell[]::new))
            .orElseGet(() -> new DataCell[0]);
    }

    private static <S, T> Function<S, T> noopExtractor() {
        return s -> null;
    }

    private static <T> BinaryOperator<T> noopAggregator() {
        return (a, b) -> null;
    }

    private static final class CellSplitterFactory<T> implements Consumer<DataCell>, Supplier<CellFactory> {

        private final BinaryOperator<T> m_stateAggregator;

        private final Function<DataCell, T> m_stateExtractor;

        private final Function<T, CellFactory> m_cellFactoryCreator;

        private final boolean m_stateful;

        private T m_state;

        CellSplitterFactory(final T initialState, final Function<DataCell, T> stateExtractor,
            final BinaryOperator<T> stateAggregator, final Function<T, CellFactory> cellFactoryCreator) {
            this(initialState, stateExtractor, stateAggregator, cellFactoryCreator, true);
        }

        CellSplitterFactory(final T initialState, final Function<DataCell, T> stateExtractor,
            final BinaryOperator<T> stateAggregator, final Function<T, CellFactory> cellFactoryCreator,
            final boolean stateful) {
            m_state = initialState;
            m_stateExtractor = stateExtractor;
            m_stateAggregator = stateAggregator;
            m_cellFactoryCreator = cellFactoryCreator;
            m_stateful = stateful;
        }

        @Override
        public void accept(final DataCell t) {
            m_state = m_stateAggregator.apply(m_state, m_stateExtractor.apply(t));
        }

        @Override
        public CellFactory get() {
            return m_cellFactoryCreator.apply(m_state);
        }

        public boolean isStateful() {
            return m_stateful;
        }
    }

    private static final class MultiCellFactory extends AbstractCellFactory {

        private final Function<DataRow, DataCell[]> m_extractor;

        MultiCellFactory(final DataColumnSpec[] columnSpecs, final Function<DataRow, DataCell[]> extractor) {
            super(columnSpecs);
            m_extractor = extractor;
        }

        @Override
        public DataCell[] getCells(final DataRow row) {
            return m_extractor.apply(row);
        }
    }

    private static Function<MessageValue, DataCell[]> createContentPartExtractor(final String contentType,
        final Function<byte[], DataCell> contentCellFactory) {
        return message -> message.getContent().stream()//
            .filter(part -> contentType.equals(part.getType()))//
            .map(MessageContentPart::getData)//
            .map(contentCellFactory)//
            .toArray(DataCell[]::new);
    }

    private static DataCell[] padWithMissing(final DataCell[] cells, final int length) {
        if (cells.length >= length) {
            return cells;
        }
        var paddedCells = new DataCell[length];
        System.arraycopy(cells, 0, paddedCells, 0, cells.length);
        for (int i = cells.length; i < length; i++) {
            paddedCells[i] = DataType.getMissingCell();
        }
        return paddedCells;
    }

    private static final class PartExtractorSingleCellFactory extends SingleCellFactory {

        private final int m_messageColumnIndex;

        private final Function<MessageValue, DataCell> m_extractor;

        PartExtractorSingleCellFactory(final String columnName, final DataType contentType,
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

    private static DataCell extractToolCallId(final MessageValue message) {
        return message.getToolCallId().map(s -> (DataCell)new StringCell(s)).orElseGet(DataType::getMissingCell);
    }

    private static DataCell toJsonCell(final ToolCall toolCall) {
        try {
            return JSONCellFactory.create("{\"toolName\":\"" + toolCall.toolName() + "\", \"id\":\"" + toolCall.id()
                + "\", \"arguments\":" + toolCall.arguments() + "}");
        } catch (IOException e) {
            throw new RuntimeException("Failed to create JSON cell for tool call: " + toolCall, e);
        }
    }

}
