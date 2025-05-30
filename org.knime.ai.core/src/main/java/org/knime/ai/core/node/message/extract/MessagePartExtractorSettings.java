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

import org.knime.ai.core.data.message.MessageValue;
import org.knime.core.data.DataColumnSpec;
import org.knime.core.data.DataTableSpec;
import org.knime.core.webui.node.dialog.defaultdialog.DefaultNodeSettings;
import org.knime.core.webui.node.dialog.defaultdialog.widget.Widget;
import org.knime.core.webui.node.dialog.defaultdialog.widget.choices.ChoicesProvider;
import org.knime.core.webui.node.dialog.defaultdialog.widget.choices.column.CompatibleColumnsProvider;

/**
 * Settings for the Message Part Extractor node.
 *
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 */
@SuppressWarnings("restriction")
final class MessagePartExtractorSettings implements DefaultNodeSettings {

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

    @Widget(title = "Extract role",
        description = "Whether to extract the role of the message (e.g., user, AI, tool). If enabled, a new column with the role will be added.")
    public boolean m_extractRole = true;

    @Widget(title = "Extract text parts",
        description = "Whether to extract text parts from the messages. If enabled, a new column with the text content will be added.")
    public boolean m_extractTextParts = true;

    @Widget(title = "Extract tool calls",
        description = "Whether to extract tool calls from the messages. If enabled, a new column with the tool call content will be added.")
    public boolean m_extractToolCalls = true;

    @Widget(title = "Extract tool call IDs",
        description = "Whether to extract tool call IDs from the messages. If enabled, a new column with the tool call IDs will be added.")
    public boolean m_extractToolCallIds = true;

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
