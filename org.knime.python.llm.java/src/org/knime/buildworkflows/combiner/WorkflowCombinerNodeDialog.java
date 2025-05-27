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
 *   Feb 12, 2020 (hornm): created
 */
package org.knime.python.llm.java.combiner;

import java.util.Arrays;
import java.util.Objects;

import javax.swing.BoxLayout;
import javax.swing.JPanel;

import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeDialogPane;
import org.knime.core.node.NodeLogger;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.NotConfigurableException;
import org.knime.core.node.port.PortObjectSpec;
import org.knime.core.node.workflow.capture.WorkflowPortObjectSpec;

/**
 *
 * @author Martin Horn, KNIME GmbH, Konstanz, Germany
 */
class WorkflowCombinerNodeDialog extends NodeDialogPane {
    private static final NodeLogger LOGGER = NodeLogger.getLogger(WorkflowCombinerNodeDialog.class);

    private final JPanel m_panel = new JPanel();

    private WorkflowConnectPanel[] m_wfConnectPanels = null;

    /**
     * A new dialog.
     */
    public WorkflowCombinerNodeDialog() {
        addTab("Workflow Connections", m_panel);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void saveSettingsTo(final NodeSettingsWO settings) throws InvalidSettingsException {
        ConnectionMaps cm = new ConnectionMaps(
            Arrays.stream(m_wfConnectPanels).map(p -> p.createConnectionMap()).toArray(s -> new ConnectionMap[s]));
        NodeSettingsWO cmSettings = settings.addNodeSettings(WorkflowCombinerNodeModel.CFG_CONNECTION_MAPS);
        cm.save(cmSettings);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void loadSettingsFrom(final NodeSettingsRO settings, final PortObjectSpec[] specs)
        throws NotConfigurableException {
        if (Arrays.stream(specs).anyMatch(Objects::isNull)) {
            throw new NotConfigurableException("Some input ports don't have a workflow available.");
        }
        ConnectionMaps cm = new ConnectionMaps();
        if (settings.containsKey(WorkflowCombinerNodeModel.CFG_CONNECTION_MAPS)) {
            try {
                cm.load(settings.getNodeSettings(WorkflowCombinerNodeModel.CFG_CONNECTION_MAPS));
            } catch (InvalidSettingsException e) {
                LOGGER.warn("Settings couldn't be load for dialog. Ignored.", e);
            }
        }
        m_panel.removeAll();
        m_panel.setLayout(new BoxLayout(m_panel, BoxLayout.Y_AXIS));
        m_wfConnectPanels = new WorkflowConnectPanel[specs.length - 1];
        for (int i = 0; i < specs.length - 1; i++) {
            WorkflowPortObjectSpec w1 = (WorkflowPortObjectSpec)specs[i];
            WorkflowPortObjectSpec w2 = (WorkflowPortObjectSpec)specs[i + 1];
            m_wfConnectPanels[i] = new WorkflowConnectPanel(w1.getOutputs(), w2.getInputs(), cm.getConnectionMap(i),
                "Connect workflow " + (i + 1) + " with " + (i + 2), w1.getWorkflowName(), w2.getWorkflowName());
            m_panel.add(m_wfConnectPanels[i]);
        }
    }

}
