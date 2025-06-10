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
 *   Jun 10, 2025 (Adrian Nembach, KNIME GmbH, Konstanz, Germany): created
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
import org.knime.ai.core.data.message.MessageValue.MessageContentPart.MessageContentPartType;
import org.knime.ai.core.data.message.MessageValue.ToolCall;
import org.knime.core.data.DataCell;
import org.knime.core.data.DataColumnSpec;
import org.knime.core.data.DataColumnSpecCreator;
import org.knime.core.data.DataType;
import org.knime.core.data.container.CellFactory;
import org.knime.core.data.def.StringCell;
import org.knime.core.data.image.png.PNGImageCellFactory;
import org.knime.core.data.json.JSONCellFactory;

/**
 * Contains the logic for creating cell splitter factories for the Message Extractor node.
 *
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 */
final class MessageCellSplitterFactories {

    static List<CellSplitterFactory<?>> createCellSplitterFactories(final MessagePartExtractorSettings settings,
        final int messageColumnIndex) {
        var list = new ArrayList<CellSplitterFactory<?>>();
        settings.m_roleColumnName.map(c -> new PartExtractorSingleCellFactory(c, StringCell.TYPE,
            MessageCellSplitterFactories::extractRole, messageColumnIndex))
            .ifPresent(factory -> list.add(createStatelessCellSplitterFactory(() -> factory)));
        settings.m_nameColumnName.map(c -> new PartExtractorSingleCellFactory(c, StringCell.TYPE,
            MessageCellSplitterFactories::extractName, messageColumnIndex))
            .ifPresent(factory -> list.add(createStatelessCellSplitterFactory(() -> factory)));
        settings.m_textPartsPrefix.map(prefix -> createMultiCellSplitterFactory(
            createContentCounter(MessageContentPartType.TEXT), createColumnSpecCreator(prefix, StringCell.TYPE), messageColumnIndex,
            createContentPartExtractor(MessageContentPartType.TEXT, data -> new StringCell(new String(data)))))
            .ifPresent(list::add);
        settings.m_imagePartsPrefix.map(prefix -> createMultiCellSplitterFactory(
            createContentCounter(MessageContentPartType.PNG), createColumnSpecCreator(prefix, PNGImageCellFactory.TYPE), messageColumnIndex,
            createContentPartExtractor(MessageContentPartType.PNG, PNGImageCellFactory::create)))
            .ifPresent(list::add);
        settings.m_toolCallsPrefix.map(prefix -> createMultiCellSplitterFactory(
            MessageCellSplitterFactories::countToolCalls,
            createColumnSpecCreator(prefix, JSONCellFactory.TYPE), messageColumnIndex,
            MessageCellSplitterFactories::extractToolCalls))
            .ifPresent(list::add);
        settings.m_toolCallIdColumnName.map(c -> new PartExtractorSingleCellFactory(c, StringCell.TYPE,
            MessageCellSplitterFactories::extractToolCallId, messageColumnIndex))
            .ifPresent(factory -> list.add(createStatelessCellSplitterFactory(() -> factory)));
        return list;
    }

    static IntFunction<DataColumnSpec> createColumnSpecCreator(final String prefix, final DataType type) {
        return j -> new DataColumnSpecCreator(prefix + (j + 1), type).createSpec();
    }

    static CellSplitterFactory<Integer> createMultiCellSplitterFactory(
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

    static CellSplitterFactory<?>
        createStatelessCellSplitterFactory(final Supplier<CellFactory> cellFactorySupplier) {
        return new CellSplitterFactory<>(null, noopExtractor(), noopAggregator(), c -> cellFactorySupplier.get(),
            false);
    }

    static Function<DataCell, Integer> createContentCounter(final MessageContentPartType contentType) {
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


    static Integer countToolCalls(final DataCell cell) {
        if (cell.isMissing()) {
            return 0;
        }
        var message = (MessageValue)cell;
        return message.getToolCalls().map(List::size).orElse(0);
    }

    static DataCell[] extractToolCalls(final MessageValue message) {
        return message.getToolCalls()
            .map(list -> list.stream().map(MessageCellSplitterFactories::toJsonCell)//
                .toArray(DataCell[]::new))
            .orElseGet(() -> new DataCell[0]);
    }

    private static <S, T> Function<S, T> noopExtractor() {
        return s -> null;
    }

    private static <T> BinaryOperator<T> noopAggregator() {
        return (a, b) -> null;
    }

    static final class CellSplitterFactory<T> implements Consumer<DataCell>, Supplier<CellFactory> {

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

    static Function<MessageValue, DataCell[]> createContentPartExtractor(final MessageContentPartType contentType,
        final Function<byte[], DataCell> contentToCellFn) {
        return message -> message.getContent().stream()//
            .filter(part -> contentType.equals(part.getType()))//
            .map(MessageContentPart::getData)//
            .map(contentToCellFn)//
            .toArray(DataCell[]::new);
    }

    static DataCell[] padWithMissing(final DataCell[] cells, final int length) {
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

    static StringCell extractRole(final MessageValue message) {
        return new StringCell(message.getMessageType().name());
    }

    static DataCell extractToolCallId(final MessageValue message) {
        return message.getToolCallId().map(s -> (DataCell)new StringCell(s)).orElseGet(DataType::getMissingCell);
    }

    static DataCell extractName(final MessageValue message) {
        return message.getName().map(s -> (DataCell) new StringCell(s)).orElse(DataType.getMissingCell());
    }

    static DataCell toJsonCell(final ToolCall toolCall) {
        try {
            return JSONCellFactory.create("{\"toolName\":\"" + toolCall.toolName() + "\", \"id\":\"" + toolCall.id()
                + "\", \"arguments\":" + toolCall.arguments() + "}");
        } catch (IOException e) {
            throw new RuntimeException("Failed to create JSON cell for tool call: " + toolCall, e);
        }
    }
}
