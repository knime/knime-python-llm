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

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.knime.ai.core.data.message.MessageCell;
import org.knime.ai.core.data.message.MessageValue.MessageContentPart;
import org.knime.ai.core.data.message.MessageValue.MessageType;
import org.knime.ai.core.data.message.MessageValue.ToolCall;
import org.knime.ai.core.data.message.PngContentPart;
import org.knime.ai.core.data.message.TextContentPart;
import org.knime.ai.core.node.message.create.MessageCreatorNodeSettings.Contents.ContentType;
import org.knime.core.data.DataCell;
import org.knime.core.data.DataRow;
import org.knime.core.data.DataTableSpec;
import org.knime.core.data.DataValue;
import org.knime.core.data.StringValue;
import org.knime.core.data.image.png.PNGImageValue;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.util.CheckUtils;

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


    Function<DataRow, DataCell> createMessageCellCreator() throws InvalidSettingsException {
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

    Function<DataRow, List<MessageContentPart>> createContentExtractor() throws InvalidSettingsException{
        List<Function<DataRow, Optional<MessageContentPart>>> extractors = new ArrayList<>();
        for (int i = 0; i < m_modelSettings.m_content.length; i++) {
            MessageCreatorNodeSettings.Contents c = m_modelSettings.m_content[i];
            extractors.add(createContentPartExtractor(c, (i+1)));
        }

        return r -> extractors.stream()
            .map(extractor -> extractor.apply(r))
            .flatMap(Optional::stream)
            .toList();
    }

    Function<DataRow, Optional<String>> createNameExtractor() throws InvalidSettingsException {
        if (m_modelSettings.m_nameColumn.getEnumChoice().isPresent()) {
            return r -> Optional.empty();
        } else {
            int nameColIdx = m_inSpec.findColumnIndex(m_modelSettings.m_nameColumn.getStringChoice());
            CheckUtils.checkSetting(nameColIdx >= 0, "The selected column '%s' is not part of the input table.", m_modelSettings.m_nameColumn.getStringChoice());
            return r -> getValue(r.getCell(nameColIdx), StringValue.class).map(StringValue::getStringValue);
        }
    }

    Function<DataRow, Optional<MessageContentPart>>
        createContentPartExtractor(final MessageCreatorNodeSettings.Contents contentPartSettings, final int index) throws InvalidSettingsException {
        if (contentPartSettings.m_contentType == ContentType.TEXT) {
            return createTextContentExtractor(contentPartSettings, index);
        } else if (contentPartSettings.m_contentType == ContentType.IMAGE) {
            return createImageContentExtractor(contentPartSettings, index);
        }
        throw new IllegalArgumentException("Unknown content type: " + contentPartSettings.m_contentType);
    }

    Function<DataRow, Optional<MessageContentPart>>
        createTextContentExtractor(final MessageCreatorNodeSettings.Contents content, final int index) throws InvalidSettingsException {

        if (content.m_inputType == MessageCreatorNodeSettings.InputType.COLUMN) {
            CheckUtils.checkSettingNotNull(content.m_textColumn,
                "Please select a valid column for the Text column in Content "  + index + ".");

            int textColIdx = m_inSpec.findColumnIndex(content.m_textColumn);
            CheckUtils.checkSetting(textColIdx >= 0, "The selected column '%s' is not part of the input table.", content.m_textColumn);
            return r -> getValue(r.getCell(textColIdx), StringValue.class).map(v -> new TextContentPart(v.getStringValue()));
        } else {
            CheckUtils.checkSetting(!content.m_textValue.isBlank(),
                "Please enter a value for the Text value in Content "  + index + ".");
        }
        return r -> Optional.of(new TextContentPart(content.m_textValue));
    }

    Function<DataRow, Optional<MessageContentPart>>
        createImageContentExtractor(final MessageCreatorNodeSettings.Contents content, final int index) throws InvalidSettingsException {

        if (content.m_contentType == MessageCreatorNodeSettings.Contents.ContentType.IMAGE) {
            CheckUtils.checkSetting(content.m_imageColumn != null,
                "Please select a valid column for the Image column in Content " + index + ".");
        }

        int imageColIdx = m_inSpec.findColumnIndex(content.m_imageColumn);
        CheckUtils.checkSetting(imageColIdx >= 0, "The selected column '%s' is not part of the input table.", content.m_imageColumn);

        return r -> getValue(r.getCell(imageColIdx), PNGImageValue.class).map(v -> new PngContentPart(v.getImageContent().getByteArray()));
    }

    Function<DataRow, MessageType> createRoleExtractor() throws InvalidSettingsException {

        if (m_modelSettings.m_roleInputType == MessageCreatorNodeSettings.InputType.COLUMN) {
            CheckUtils.checkSettingNotNull(m_modelSettings.m_roleColumn,
                    "Please select a valid column for the Role column.");
        }

        if (m_modelSettings.m_roleInputType == MessageCreatorNodeSettings.InputType.VALUE) {
            return r -> m_modelSettings.m_roleValue;
        } else {
            int roleColIdx = m_inSpec.findColumnIndex(m_modelSettings.m_roleColumn);
            CheckUtils.checkSetting(roleColIdx >= 0, "The selected column '%s' is not part of the input table.", m_modelSettings.m_roleColumn);

            var validRoles = Stream.of(MessageType.values())
                .map(Enum::name)
                .map(String::toUpperCase)
                .collect(Collectors.toSet());
            return r -> {
                String value = getValue(r.getCell(roleColIdx), StringValue.class).map(StringValue::getStringValue).map(String::toUpperCase).orElseThrow(() ->
                    new IllegalArgumentException(
                        "Role column '%s' contains missing or invalid values.".formatted(m_modelSettings.m_roleColumn)));
                if (!validRoles.contains(value)) {
                    throw new IllegalArgumentException(
                        "Role column '%s' contains invalid value '%s'. Valid roles are: %s".formatted(
                            m_modelSettings.m_roleColumn, value, validRoles));
                }
                return MessageType.valueOf(value);
            };
        }
    }

    static <T extends DataValue> Optional<T> getValue(final DataCell cell, final Class<T> valueClass) {
        if (cell.isMissing()) {
            return Optional.empty();
        }
        return Optional.of(valueClass.cast(cell));
    }

    Function<DataRow, String> createToolCallIdExtractor() throws InvalidSettingsException {
        if (m_modelSettings.m_toolCallIdColumn.getEnumChoice().isPresent()) {
            return r -> null;
        }
        int colIdx = m_inSpec.findColumnIndex(m_modelSettings.m_toolCallIdColumn.getStringChoice());
        CheckUtils.checkSetting(colIdx >= 0, "The selected column '%s' is not part of the input table.", m_modelSettings.m_toolCallIdColumn.getStringChoice());
        return r -> getValue(r.getCell(colIdx), StringValue.class).map(StringValue::getStringValue).orElse(null);
    }

    Function<DataRow, List<ToolCall>> createToolCallsExtractor() throws InvalidSettingsException {
        var toolCallExtractors = new ArrayList<Function<DataRow, Optional<ToolCall>>>();
        for (int i = 0; i < m_modelSettings.m_toolCalls.length; i++) {
            var tc = m_modelSettings.m_toolCalls[i];
            toolCallExtractors.add(createToolCallExtractor(tc, (i + 1)));
        }
        return r -> toolCallExtractors.stream().map(f -> f.apply(r)).flatMap(Optional::stream).toList();
    }

    Function<DataRow, Optional<ToolCall>> createToolCallExtractor(final MessageCreatorNodeSettings.ToolCallSettings tc, final int index) throws InvalidSettingsException {
        int nameIdx = m_inSpec.findColumnIndex(tc.m_toolNameColumn);
        int idIdx = m_inSpec.findColumnIndex(tc.m_toolIdColumn);
        int argsIdx = m_inSpec.findColumnIndex(tc.m_argumentsColumn);

        CheckUtils.checkSetting(nameIdx >= 0, "Please select a valid column for the Tool name column in Tool Call " + index + ".");
        CheckUtils.checkSetting(idIdx >= 0,  "Please select a valid column for the Tool ID column in Tool Call " + index + ".");
        CheckUtils.checkSetting(argsIdx >= 0,  "Please select a valid column for the Arguments column in Tool Call " + index + ".");

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
