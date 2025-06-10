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
package org.knime.ai.core.node.message.create;

import java.util.List;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Stream;

import org.knime.ai.core.data.message.PngContentPart;
import org.knime.ai.core.data.message.MessageCell;
import org.knime.ai.core.data.message.MessageValue.MessageContentPart;
import org.knime.ai.core.data.message.MessageValue.MessageType;
import org.knime.ai.core.data.message.MessageValue.ToolCall;
import org.knime.ai.core.data.message.TextContentPart;
import org.knime.ai.core.node.message.create.MessageCreatorNodeSettings.Contents.ContentType;
import org.knime.core.data.DataCell;
import org.knime.core.data.DataRow;
import org.knime.core.data.DataTableSpec;
import org.knime.core.data.StringValue;
import org.knime.core.data.image.png.PNGImageValue;

/**
 * Creates a {@link MessageCell} from a {@link DataRow}.
 *
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 */
@SuppressWarnings("restriction")
final class MessageCellCreator {

    private final MessageCreatorNodeSettings m_modelSettings;

    private final DataTableSpec m_inSpec;

    MessageCellCreator(final MessageCreatorNodeSettings modelSettings, final DataTableSpec inSpec) {
        m_modelSettings = modelSettings;
        m_inSpec = inSpec;
    }

    Function<DataRow, DataCell> createMessageCellCreator() {
        var roleExtractor = createRoleExtractor();
        var contentExtractor = createContentExtractor();
        var nameExtractor = createNameExtractor();
        var toolCallIdExtractor = createToolCallIdExtractor();
        var toolCallsExtractor = createToolCallsExtractor();
        return r -> {
            var role = roleExtractor.apply(r);
            var parts = contentExtractor.apply(r);
            var name = nameExtractor.apply(r);
            var toolCallId = toolCallIdExtractor.apply(r);
            var toolCalls = toolCallsExtractor.apply(r);
            return new MessageCell(role, parts, toolCalls.isEmpty() ? null : toolCalls, toolCallId, name.orElse(null));
        };
    }

    Function<DataRow, List<MessageContentPart>> createContentExtractor() {
        var extractors = Stream.of(m_modelSettings.m_content).map(c -> createContentPartExtractor(c)).toList();
        return r -> extractors.stream().map(extractor -> extractor.apply(r)).flatMap(Optional::stream).toList();
    }

    Function<DataRow, Optional<String>> createNameExtractor() {
        if (m_modelSettings.m_nameColumn.getEnumChoice().isPresent()) {
            return r -> Optional.empty();
        } else {
            int nameColIdx = m_inSpec.findColumnIndex(m_modelSettings.m_nameColumn.getStringChoice());
            return r -> {
                var cell = r.getCell(nameColIdx);
                if (cell.isMissing()) {
                    return Optional.empty();
                }
                return Optional.of(((StringValue)cell).getStringValue());
            };
        }
    }

    Function<DataRow, Optional<MessageContentPart>>
        createContentPartExtractor(final MessageCreatorNodeSettings.Contents contentPartSettings) {
        if (contentPartSettings.m_contentType == ContentType.TEXT) {
            return createTextContentExtractor(contentPartSettings);
        } else if (contentPartSettings.m_contentType == ContentType.IMAGE) {
            return createImageContentExtractor(contentPartSettings);
        }
        throw new IllegalArgumentException("Unknown content type: " + contentPartSettings.m_contentType);
    }

    Function<DataRow, Optional<MessageContentPart>>
        createTextContentExtractor(final MessageCreatorNodeSettings.Contents content) {
        if (content.m_inputType == MessageCreatorNodeSettings.InputType.COLUMN) {
            int textColIdx = m_inSpec.findColumnIndex(content.m_textColumn);
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

    Function<DataRow, Optional<MessageContentPart>>
        createImageContentExtractor(final MessageCreatorNodeSettings.Contents content) {
        int imageColIdx = m_inSpec.findColumnIndex(content.m_imageColumn);
        return r -> {
            var cell = r.getCell(imageColIdx);
            if (cell.isMissing()) {
                return Optional.empty();
            }
            return Optional.of(new PngContentPart(((PNGImageValue)cell).getImageContent().getByteArray()));
        };
    }

    Function<DataRow, MessageType> createRoleExtractor() {
        if (m_modelSettings.m_roleInputType == MessageCreatorNodeSettings.InputType.VALUE) {
            return r -> m_modelSettings.m_roleValue;
        } else {
            int roleColIdx = m_inSpec.findColumnIndex(m_modelSettings.m_roleColumn);
            return r -> {
                var cell = r.getCell(roleColIdx);
                if (cell.isMissing()) {
                    throw new IllegalArgumentException(
                        "Role column '%s' contains missing values.".formatted(m_modelSettings.m_roleColumn));
                }
                return MessageType.valueOf(((StringValue)cell).getStringValue().toUpperCase());
            };
        }
    }

    Function<DataRow, String> createToolCallIdExtractor() {
        if (m_modelSettings.m_toolCallIdColumn.getEnumChoice().isPresent()) {
            return r -> null;
        }
        int colIdx = m_inSpec.findColumnIndex(m_modelSettings.m_toolCallIdColumn.getStringChoice());
        return r -> {
            var cell = r.getCell(colIdx);
            if (cell.isMissing()) {
                return null;
            }
            return ((StringValue)cell).getStringValue();
        };
    }

    Function<DataRow, List<ToolCall>> createToolCallsExtractor() {
        var toolCallExtractors = Stream.of(m_modelSettings.m_toolCalls).map(tc -> createToolCallExtractor(tc)).toList();
        return r -> toolCallExtractors.stream().map(f -> f.apply(r)).flatMap(Optional::stream).toList();
    }

    Function<DataRow, Optional<ToolCall>>
        createToolCallExtractor(final MessageCreatorNodeSettings.ToolCallSettings tc) {
        int nameIdx = m_inSpec.findColumnIndex(tc.m_toolNameColumn);
        int idIdx = m_inSpec.findColumnIndex(tc.m_toolIdColumn);
        int argsIdx = m_inSpec.findColumnIndex(tc.m_argumentsColumn);
        return r -> {
            var nameCell = r.getCell(nameIdx);
            var idCell = r.getCell(idIdx);
            var argsCell = r.getCell(argsIdx);
            if (nameCell.isMissing() || idCell.isMissing() || argsCell.isMissing()) {
                return Optional.empty();
            }
            return Optional.of(new ToolCall(((StringValue)nameCell).getStringValue(),
                ((StringValue)idCell).getStringValue(), ((StringValue)argsCell).getStringValue()));
        };
    }
}
