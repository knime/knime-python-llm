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
 *   24 Jun 2020 (Marc Bux, KNIME GmbH, Berlin, Germany): created
 */
package org.knime.python.llm.java.reader;

import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.context.NodeCreationConfiguration;
import org.knime.core.node.defaultnodesettings.SettingsModelBoolean;
import org.knime.core.node.defaultnodesettings.SettingsModelString;

/**
 * @author Martin Horn, KNIME GmbH, Konstanz, Germany
 * @author Marc Bux, KNIME GmbH, Berlin, Germany
 */
final class WorkflowReaderNodeConfig {

    /** The name of the optional connection input port group. */
    static final String CONNECTION_INPUT_PORT_GRP_NAME = "File System Connection";

    private final SettingsModelString m_workflowName = new SettingsModelString("custom-name", "");

    private final SettingsModelBoolean m_removeIONodes = new SettingsModelBoolean("remove-io-nodes", false);

    private final SettingsModelString m_inputIdPrefix = new SettingsModelString("input-id-prefix", "input");

    private final SettingsModelString m_outputIdPrefix = new SettingsModelString("output-id-prefix", "output");

    private final SettingsModelWorkflowChooser m_workflowChooser;

    WorkflowReaderNodeConfig(final NodeCreationConfiguration creationConfig) {
        m_workflowChooser = new SettingsModelWorkflowChooser("workflow-chooser", CONNECTION_INPUT_PORT_GRP_NAME,
            creationConfig.getPortConfig().orElseThrow(IllegalStateException::new));
        m_removeIONodes.addChangeListener(e -> {
            m_inputIdPrefix.setEnabled(m_removeIONodes.getBooleanValue());
            m_outputIdPrefix.setEnabled(m_removeIONodes.getBooleanValue());
        });
    }

    SettingsModelString getWorkflowName() {
        return m_workflowName;
    }

    SettingsModelBoolean getRemoveIONodes() {
        return m_removeIONodes;
    }

    SettingsModelWorkflowChooser getWorkflowChooserModel() {
        return m_workflowChooser;
    }

    SettingsModelString getInputIdPrefix() {
        return m_inputIdPrefix;
    }

    SettingsModelString getOutputIdPrefix() {
        return m_outputIdPrefix;
    }

    void validateConfigurationForModel(final NodeSettingsRO settings) throws InvalidSettingsException {
        m_workflowChooser.validateSettings(settings);
        m_workflowName.validateSettings(settings);
        m_removeIONodes.validateSettings(settings);
        m_inputIdPrefix.validateSettings(settings);
        m_outputIdPrefix.validateSettings(settings);
    }

    void saveConfigurationForModel(final NodeSettingsWO settings) {
        m_workflowChooser.saveSettingsTo(settings);
        m_workflowName.saveSettingsTo(settings);
        m_removeIONodes.saveSettingsTo(settings);
        m_inputIdPrefix.saveSettingsTo(settings);
        m_outputIdPrefix.saveSettingsTo(settings);
    }

    void loadConfigurationForModel(final NodeSettingsRO settings) throws InvalidSettingsException {
        m_workflowChooser.loadSettingsFrom(settings);
        m_workflowName.loadSettingsFrom(settings);
        m_removeIONodes.loadSettingsFrom(settings);
        m_inputIdPrefix.loadSettingsFrom(settings);
        m_outputIdPrefix.loadSettingsFrom(settings);
    }

    void loadConfigurationForDialog() {
        m_inputIdPrefix.setEnabled(m_removeIONodes.getBooleanValue());
        m_outputIdPrefix.setEnabled(m_removeIONodes.getBooleanValue());
    }
}
