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
 *   May 21, 2025 (hornm): created
 */
package org.knime.ai.core.node.tool.output;

import java.net.URI;

import org.knime.core.data.DataTableSpec;
import org.knime.core.node.BufferedDataTable;
import org.knime.core.node.ExecutionContext;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.agentic.tool.WorkflowToolCell;
import org.knime.core.node.dialog.ContentType;
import org.knime.core.node.dialog.ExternalNodeData;
import org.knime.core.node.dialog.OutputNode;
import org.knime.core.webui.node.dialog.defaultdialog.widget.validation.internal.WorkflowIOParameterNameValidation;
import org.knime.core.webui.node.impl.WebUINodeConfiguration;
import org.knime.core.webui.node.impl.WebUINodeModel;

/**
 * @author Martin Horn, KNIME GmbH, Konstanz, Germany
 */
public final class ToolMessageOutputNodeModel extends WebUINodeModel<ToolMessageOutputNodeSettings>
    implements OutputNode {

    ToolMessageOutputNodeModel(final WebUINodeConfiguration configuration) {
        super(configuration, ToolMessageOutputNodeSettings.class);
    }

    @Override
    public ExternalNodeData getExternalOutput() {
        var config = getSettings();
        var parameterName =
            config.map(c -> c.m_parameterName).orElse(ToolMessageOutputNodeSettings.DEFAULT_PARAMETER_NAME);

        var contentType = ContentType.CONTENT_TYPE_DEF_PREFIX + BufferedDataTable.class.getName();
        return ExternalNodeData.builder(parameterName)//
            // TODO AP-24396 - serve table content (see WorkflowOutputNodeModel)
            .resource(URI.create("file:/dev/null"))//
            .contentType(contentType) //
            .build();
    }

    @Override
    protected DataTableSpec[] configure(final DataTableSpec[] inSpecs,
        final ToolMessageOutputNodeSettings modelSettings) throws InvalidSettingsException {
        return null;
    }

    @Override
    protected BufferedDataTable[] execute(final BufferedDataTable[] inData, final ExecutionContext exec,
        final ToolMessageOutputNodeSettings modelSettings) throws Exception {
        WorkflowToolCell.extractToolMessageContent(inData[0]);
        return null;
    }

    @Override
    protected void validateSettings(final ToolMessageOutputNodeSettings settings) throws InvalidSettingsException {
        WorkflowIOParameterNameValidation.validateParameterName(settings.m_parameterName);
    }

}
