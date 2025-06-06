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
 *   May 30, 2025 (Seray Arslan, KNIME GmbH, Konstanz, Germany): created
 */
package org.knime.ai.core.node.message.creator;

import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Stream;

import org.knime.ai.core.data.message.ImageContentPart;
import org.knime.ai.core.data.message.MessageCell;
import org.knime.ai.core.data.message.MessageValue.MessageContentPart;
import org.knime.ai.core.data.message.MessageValue.MessageType;
import org.knime.ai.core.data.message.MessageValue.ToolCall;
import org.knime.ai.core.data.message.TextContentPart;
import org.knime.ai.core.node.message.creator.MessageCreatorNodeSettings.Contents.ContentType;
import org.knime.core.data.DataCell;
import org.knime.core.data.DataColumnSpec;
import org.knime.core.data.DataColumnSpecCreator;
import org.knime.core.data.DataRow;
import org.knime.core.data.DataTableSpec;
import org.knime.core.data.StringValue;
import org.knime.core.data.container.ColumnRearranger;
import org.knime.core.data.container.SingleCellFactory;
import org.knime.core.data.image.png.PNGImageValue;
import org.knime.core.node.BufferedDataTable;
import org.knime.core.node.ExecutionContext;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.webui.node.impl.WebUINodeConfiguration;
import org.knime.core.webui.node.impl.WebUINodeModel;

/**
 * Node model of the Message Creator node.
 *
 * @author Seray Arslan, KNIME GmbH, Konstanz, Germany
 */
@SuppressWarnings("restriction")
final class MessageCreatorNodeModel extends WebUINodeModel<MessageCreatorNodeSettings> {

    /**
     * @param configuration of the node
     */
    protected MessageCreatorNodeModel(final WebUINodeConfiguration configuration) {
        super(configuration, MessageCreatorNodeSettings.class);
    }

    @Override
    protected DataTableSpec[] configure(final DataTableSpec[] inSpecs, final MessageCreatorNodeSettings modelSettings)
        throws InvalidSettingsException {
        return new DataTableSpec[]{createRearranger(modelSettings, inSpecs[0]).createSpec()};
    }

    @Override
    protected BufferedDataTable[] execute(final BufferedDataTable[] inData, final ExecutionContext exec,
        final MessageCreatorNodeSettings modelSettings) throws Exception {
        var table = inData[0];
        var messageTable = exec.createColumnRearrangeTable(table, createRearranger(modelSettings, table.getDataTableSpec()), exec);
        return new BufferedDataTable[]{messageTable};
    }

    private static ColumnRearranger createRearranger(final MessageCreatorNodeSettings modelSettings,
        final DataTableSpec inSpec) {
        var rearranger = new ColumnRearranger(inSpec);
        var messageCellCreator = createMessageCellCreator(modelSettings, inSpec);
        var columnSpec = new DataColumnSpecCreator(modelSettings.m_messageColumnName, MessageCell.TYPE).createSpec();
        rearranger.append(new SingleCellFactoryImpl(columnSpec, messageCellCreator));
        rearranger.remove(columnsToRemove(modelSettings).toArray(String[]::new));
        return rearranger;
    }

    private static Set<String> columnsToRemove(final MessageCreatorNodeSettings modelSettings) {
        if (!modelSettings.m_removeInputColumns) {
            return Set.of();
        }
        var columnsToRemove = new HashSet<String>();
        // Role column
        if (modelSettings.m_roleInputType == MessageCreatorNodeSettings.InputType.COLUMN && modelSettings.m_roleColumn != null) {
            columnsToRemove.add(modelSettings.m_roleColumn);
        }
        // Tool call id column
        if (modelSettings.m_toolCallIdColumn.getEnumChoice().isEmpty()) {
            columnsToRemove.add(modelSettings.m_toolCallIdColumn.getStringChoice());
        }
        // Content columns
        for (var content : modelSettings.m_content) {
            if (content.m_contentType == MessageCreatorNodeSettings.Contents.ContentType.TEXT && content.m_inputType == MessageCreatorNodeSettings.InputType.COLUMN) {
                columnsToRemove.add(content.m_textColumn);
            }
            if (content.m_contentType == MessageCreatorNodeSettings.Contents.ContentType.IMAGE) {
                columnsToRemove.add(content.m_imageColumn);
            }
        }
        // Tool call columns
        for (var tc : modelSettings.m_toolCalls) {
            columnsToRemove.add(tc.m_toolNameColumn);
            columnsToRemove.add(tc.m_toolIdColumn);
            columnsToRemove.add(tc.m_argumentsColumn);
        }
        return columnsToRemove;
    }



    private static Function<DataRow, DataCell> createMessageCellCreator(
        final MessageCreatorNodeSettings modelSettings, final DataTableSpec inSpec) {
        var roleExtractor = createRoleExtractor(modelSettings, inSpec);
        var contentExtractor = createContentExtractor(modelSettings, inSpec);
        var toolCallIdExtractor = createToolCallIdExtractor(modelSettings, inSpec);
        var toolCallsExtractor = createToolCallsExtractor(modelSettings, inSpec);
        return r -> {
            var role = roleExtractor.apply(r);
            var parts = contentExtractor.apply(r);
            var toolCallId = toolCallIdExtractor.apply(r);
            var toolCalls = toolCallsExtractor.apply(r);
            return new MessageCell(role, parts, toolCalls.isEmpty() ? null : toolCalls, toolCallId);
        };
    }

    private static Function<DataRow, List<MessageContentPart>>
        createContentExtractor(final MessageCreatorNodeSettings modelSettings, final DataTableSpec inSpec) {
        var extractors = Stream.of(modelSettings.m_content)//
                .map(c -> createContentPartExtractor(c, inSpec))//
                .toList();
        return r -> extractors.stream()
            .map(extractor -> extractor.apply(r))
            .flatMap(Optional::stream)
            .toList();
    }

    private static Function<DataRow, Optional<MessageContentPart>>
        createContentPartExtractor(final MessageCreatorNodeSettings.Contents contentPartSettings, final DataTableSpec inSpec) {
        if (contentPartSettings.m_contentType == ContentType.TEXT) {
            return createTextContentExtractor(contentPartSettings, inSpec);
        } else if (contentPartSettings.m_contentType == ContentType.IMAGE) {
            return createImageContentExtractor(contentPartSettings, inSpec);
        }
        throw new IllegalArgumentException("Unknown content type: " + contentPartSettings.m_contentType);
    }

    private static Function<DataRow, Optional<MessageContentPart>>
        createTextContentExtractor(final MessageCreatorNodeSettings.Contents content, final DataTableSpec inSpec) {
        if (content.m_inputType == MessageCreatorNodeSettings.InputType.COLUMN) {
            int textColIdx = inSpec.findColumnIndex(content.m_textColumn);
            return r -> {
                var cell = r.getCell(textColIdx);
                if (cell.isMissing()) {
                    return Optional.empty();
                }
                return Optional.of(new TextContentPart(((StringValue)cell).getStringValue()));
            };
        }
        return r -> Optional.of(new TextContentPart(content.m_textValue));
    }

    private static Function<DataRow, Optional<MessageContentPart>>
        createImageContentExtractor(final MessageCreatorNodeSettings.Contents content, final DataTableSpec inSpec) {
        int imageColIdx = inSpec.findColumnIndex(content.m_imageColumn);
        return r -> {
            var cell = r.getCell(imageColIdx);
            if (cell.isMissing()) {
                return Optional.empty();
            }
            return Optional.of(new ImageContentPart(((PNGImageValue)cell).getImageContent().getByteArray()));
        };
    }

    private static Function<DataRow, MessageType> createRoleExtractor(final MessageCreatorNodeSettings modelSettings,
        final DataTableSpec inSpec) {
        if (modelSettings.m_roleInputType == MessageCreatorNodeSettings.InputType.VALUE) {
            return r -> modelSettings.m_roleValue;
        } else {
            int roleColIdx = inSpec.findColumnIndex(modelSettings.m_roleColumn);
            return r -> {
                var cell = r.getCell(roleColIdx);
                if (cell.isMissing()) {
                    throw new IllegalArgumentException(
                        "Role column '%s' contains missing values.".formatted(modelSettings.m_roleColumn));
                }
                return MessageType.valueOf(((StringValue)cell).getStringValue().toUpperCase());
            };
        }
    }

    private static Function<DataRow, String> createToolCallIdExtractor(final MessageCreatorNodeSettings modelSettings, final DataTableSpec inSpec) {
        if (modelSettings.m_toolCallIdColumn.getEnumChoice().isPresent()) {
            return r -> null;
        }
        int colIdx = inSpec.findColumnIndex(modelSettings.m_toolCallIdColumn.getStringChoice());
        return r -> {
            var cell = r.getCell(colIdx);
            if (cell.isMissing()) {
                return null;
            }
            return ((StringValue)cell).getStringValue();
        };
    }

    private static Function<DataRow, List<ToolCall>> createToolCallsExtractor(final MessageCreatorNodeSettings modelSettings, final DataTableSpec inSpec) {
        var toolCallExtractors = Stream.of(modelSettings.m_toolCalls)
            .map(tc -> createToolCallExtractor(tc, inSpec))
            .toList();
        return r -> toolCallExtractors.stream()
            .map(f -> f.apply(r))
            .flatMap(Optional::stream)
            .toList();
    }

    private static Function<DataRow, Optional<ToolCall>> createToolCallExtractor(final MessageCreatorNodeSettings.ToolCallSettings tc, final DataTableSpec inSpec) {
        int nameIdx = inSpec.findColumnIndex(tc.m_toolNameColumn);
        int idIdx = inSpec.findColumnIndex(tc.m_toolIdColumn);
        int argsIdx = inSpec.findColumnIndex(tc.m_argumentsColumn);
        return r -> {
            var nameCell = r.getCell(nameIdx);
            var idCell = r.getCell(idIdx);
            var argsCell = r.getCell(argsIdx);
            if (nameCell.isMissing() || idCell.isMissing() || argsCell.isMissing()) {
                return Optional.empty();
            }
            return Optional.of(new ToolCall(
                ((StringValue)nameCell).getStringValue(),
                ((StringValue)idCell).getStringValue(),
                ((StringValue)argsCell).getStringValue()
            ));
        };
    }

    private static final class SingleCellFactoryImpl extends SingleCellFactory {

        private final Function<DataRow, DataCell> m_mapper;

        SingleCellFactoryImpl(final DataColumnSpec columnSpec, final Function<DataRow, DataCell> mapper) {
            super(columnSpec);
            m_mapper = mapper;
        }

        @Override
        public DataCell getCell(final DataRow row) {
            return m_mapper.apply(row);
        }

    }

}
