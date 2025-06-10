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

import java.util.Optional;

import org.knime.ai.core.data.message.MessageValue;
import org.knime.core.data.DataColumnSpec;
import org.knime.core.data.DataTableSpec;
import org.knime.core.webui.node.dialog.defaultdialog.DefaultNodeSettings;
import org.knime.core.webui.node.dialog.defaultdialog.layout.After;
import org.knime.core.webui.node.dialog.defaultdialog.layout.HorizontalLayout;
import org.knime.core.webui.node.dialog.defaultdialog.layout.Layout;
import org.knime.core.webui.node.dialog.defaultdialog.widget.Widget;
import org.knime.core.webui.node.dialog.defaultdialog.widget.choices.ChoicesProvider;
import org.knime.core.webui.node.dialog.defaultdialog.widget.choices.column.CompatibleColumnsProvider;
import org.knime.core.webui.node.dialog.defaultdialog.widget.updates.Reference;

/**
 * Settings for the Message Part Extractor node.
 *
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 */
@SuppressWarnings("restriction")
final class MessagePartExtractorSettings implements DefaultNodeSettings {

    @HorizontalLayout
    interface RoleSettings {
    }

    @HorizontalLayout
    @After(RoleSettings.class)
    interface NameSettings {
    }

    @HorizontalLayout
    @After(NameSettings.class)
    interface TextPartsSettings {
    }

    @HorizontalLayout
    @After(TextPartsSettings.class)
    interface ImagePartsSettings {
    }

    @HorizontalLayout
    @After(ImagePartsSettings.class)
    interface ToolCallsSettings {
    }

    @HorizontalLayout
    @After(ToolCallsSettings.class)
    interface ToolCallIdSettings {
    }

    MessagePartExtractorSettings() {
    }

    MessagePartExtractorSettings(final DefaultNodeSettingsContext context) {
        m_messageColumn = context.getDataTableSpec(0)//
            .map(MessagePartExtractorSettings::autoGuessColumn)//
            .orElse(null);
    }

    @Widget(title = "Message column", description = "The message column to extract parts from.")
    @ChoicesProvider(MessageColumnProvider.class)
    public String m_messageColumn;

    @Widget(title = "Keep message column",
        description = "Whether to keep the original message column in the output table.")
    public boolean m_keepOriginalColumn = true;

    @Widget(title = "Role column name",
        description = "If enabled, a new column with the role will be added under the given name.")
    @Layout(RoleSettings.class)
    public Optional<String> m_roleColumnName = Optional.of("Role");

    @Widget(title = "Name column name",
        description = "If enabled, a new column with the name will be added under the given name.")
    @Layout(NameSettings.class)
    public Optional<String> m_nameColumnName = Optional.of("Name");

    @Widget(title = "Text parts column prefix",
        description = "If selected columns with the text contents are appended. Specify the prefix for the output columns here.")
    @Layout(TextPartsSettings.class)
    public Optional<String> m_textPartsPrefix = Optional.of("Text Content ");

    @Widget(title = "Image parts column prefix",
        description = "Prefix for the output columns for extracted image parts.")
    @Layout(ImagePartsSettings.class)
    public Optional<String> m_imagePartsPrefix = Optional.of("Image Content ");

    @Widget(title = "Tool calls column prefix", description = "Prefix for the output columns for extracted tool calls.")
    @Layout(ToolCallsSettings.class)
    public Optional<String> m_toolCallsPrefix = Optional.of("Tool Call ");

    @Widget(title = "Tool call ID column name", description = "Name of the output column for extracted tool call IDs.")
    @Layout(ToolCallIdSettings.class)
    public Optional<String> m_toolCallIdColumnName = Optional.of("Tool Call ID");

    // Reference classes for effect wiring
    public static class RoleExtractedRef implements Reference<Boolean> {
    }

    public static class NameExtractedRef implements Reference<Boolean> {
    }

    public static class TextPartsExtractedRef implements Reference<Boolean> {
    }

    public static class ImagePartsExtractedRef implements Reference<Boolean> {
    }

    public static class ToolCallsExtractedRef implements Reference<Boolean> {
    }

    public static class ToolCallIdExtractedRef implements Reference<Boolean> {
    }

    static final class MessageColumnProvider extends CompatibleColumnsProvider {
        MessageColumnProvider() {
            super(MessageValue.class);
        }
    }

    static String autoGuessColumn(final DataTableSpec spec) {
        if (spec == null) {
            return null;
        }
        return spec.stream()//
            .filter(colSpec -> colSpec.getType().isCompatible(MessageValue.class))//
            .map(DataColumnSpec::getName)//
            .reduce((l, r) -> r)// get the last compatible column
            .orElse(null);
    }
}
