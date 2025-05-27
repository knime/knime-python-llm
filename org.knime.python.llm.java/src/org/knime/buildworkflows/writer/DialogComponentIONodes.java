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
 *   Jan 24, 2020 (hornm): created
 */
package org.knime.python.llm.java.writer;

import java.awt.GridLayout;
import java.util.List;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import javax.swing.JLabel;

import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NotConfigurableException;
import org.knime.core.node.defaultnodesettings.DialogComponent;
import org.knime.core.node.port.PortObjectSpec;
import org.knime.core.node.workflow.capture.WorkflowPortObjectSpec;

/**
 * Dialog component to configure (reduced configuration) input and output nodes to be programmatically added to a
 * workflow.
 *
 * @author Martin Horn, KNIME GmbH, Konstanz, Germany
 */
public class DialogComponentIONodes extends DialogComponent {

    private static final String NONE_CHOICE = "None";

    private final PortAndNodeConfigPanel<InputNodeConfig> m_inputs;

    private final PortAndNodeConfigPanel<OutputNodeConfig> m_outputs;

    private int m_workflowInputPortIndex;

    /**
     * @param model the settings model
     * @param workflowInputPortIndex the index of the port that contains the workflow ({@link WorkflowPortObjectSpec})
     */
    public DialogComponentIONodes(final SettingsModelIONodes model, final int workflowInputPortIndex) {
        super(model);
        model.setWorkflowInputPortIndex(workflowInputPortIndex);
        m_workflowInputPortIndex = workflowInputPortIndex;

        m_inputs = new PortAndNodeConfigPanel<>("Add input node", n -> getInputConfigInstanceFor(n), NONE_CHOICE,
                WorkflowInputNodeConfig.NODE_NAME, TableInputNodeConfig.NODE_NAME, RowInputNodeConfig.NODE_NAME);
        m_outputs = new PortAndNodeConfigPanel<>("Add output node", n -> getOutputConfigInstanceFor(n), NONE_CHOICE,
                WorkflowOutputNodeConfig.NODE_NAME, TableOutputNodeConfig.NODE_NAME, RowOutputNodeConfig.NODE_NAME);

        getComponentPanel().setLayout(new GridLayout(2, 1));
        getComponentPanel().add(m_inputs);
        getComponentPanel().add(m_outputs);
    }

    /**
     * Helper to create {@link InputNodeConfig} instances for a given node name.
     *
     * @param nodeName
     * @return the instance
     */
    private static InputNodeConfig getInputConfigInstanceFor(final String nodeName) {
        switch (nodeName) {
            case TableInputNodeConfig.NODE_NAME:
                return new TableInputNodeConfig();
            case RowInputNodeConfig.NODE_NAME:
                return new RowInputNodeConfig();
            case WorkflowInputNodeConfig.NODE_NAME:
                return new WorkflowInputNodeConfig();
            default:
                throw new IllegalStateException("Node name doesn't match. Implementation error!");
        }
    }

    /**
     * Helper to create node config instances for a given node name.
     *
     * @param nodeName
     * @return the {@link OutputNodeConfig} instance
     */
    static OutputNodeConfig getOutputConfigInstanceFor(final String nodeName) {
        switch (nodeName) {
            case TableOutputNodeConfig.NODE_NAME:
                return new TableOutputNodeConfig();
            case RowOutputNodeConfig.NODE_NAME:
                return new RowOutputNodeConfig();
            case WorkflowOutputNodeConfig.NODE_NAME:
                return new WorkflowOutputNodeConfig();
            default:
                throw new IllegalStateException("Node name doesn't match. Implementation error!");
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void updateComponent() {
        WorkflowPortObjectSpec workflowPortObjectSpec =
            (WorkflowPortObjectSpec)getLastTableSpec(m_workflowInputPortIndex);
        if (workflowPortObjectSpec == null) {
            m_inputs.add(new JLabel("No input spec available"));
            return;
        }

        List<String> inputs = workflowPortObjectSpec.getInputs().entrySet().stream()
            .map(Entry::getKey).collect(Collectors.toList());
        List<String> outputs = workflowPortObjectSpec.getOutputs().entrySet().stream()
            .map(Entry::getKey).collect(Collectors.toList());

        SettingsModelIONodes model = (SettingsModelIONodes)getModel();
        model.initWithDefaults(inputs, outputs);

        m_inputs.updatePanel(inputs, p -> model.getInputNodeConfig(p).orElse(null));
        m_outputs.updatePanel(outputs, p -> model.getOutputNodeConfig(p).orElse(null));
    }


    private void updateModel() {
        SettingsModelIONodes model = (SettingsModelIONodes)getModel();
        for (Entry<String, InputNodeConfig> e : m_inputs.getSelectedConfigs().entrySet()) {
            model.setInputNodeConfig(e.getKey(), e.getValue());
        }
        for (Entry<String, OutputNodeConfig> e : m_outputs.getSelectedConfigs().entrySet()) {
            model.setOutputNodeConfig(e.getKey(), e.getValue());
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void validateSettingsBeforeSave() throws InvalidSettingsException {
        var workflowPOS = (WorkflowPortObjectSpec)getLastTableSpec(m_workflowInputPortIndex);

        WorkflowWriterNodeModel.validateIONodeConfigs(workflowPOS.getInputs(), m_inputs.getSelectedConfigs());
        WorkflowWriterNodeModel.validateIONodeConfigs(workflowPOS.getOutputs(), m_outputs.getSelectedConfigs());
        updateModel();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void setEnabledComponents(final boolean enabled) {
        m_inputs.setEnabled(enabled);
        m_outputs.setEnabled(enabled);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setToolTipText(final String text) {
        //
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void checkConfigurabilityBeforeLoad(final PortObjectSpec[] specs) throws NotConfigurableException {
        //
    }
}
