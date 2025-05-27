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
 *   Feb 3, 2020 (hornm): created
 */
package org.knime.python.llm.java.writer;

import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.util.stream.Stream;

import javax.swing.Box;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.knime.core.data.DataTable;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeFactory;
import org.knime.core.node.NodeLogger;
import org.knime.core.node.NodeModel;
import org.knime.core.node.NodeSettings;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.dialog.DialogNode;
import org.knime.core.node.port.PortType;
import org.knime.core.node.util.CheckUtils;
import org.knime.core.node.workflow.NodeContainer;
import org.knime.core.node.workflow.NodeID;
import org.knime.core.node.workflow.NodeUIInformation;
import org.knime.core.node.workflow.WorkflowManager;
import org.knime.core.node.workflow.capture.WorkflowSegment.PortID;
import org.knime.filehandling.core.defaultnodesettings.status.DefaultStatusMessage;
import org.knime.filehandling.core.defaultnodesettings.status.StatusMessage;
import org.knime.filehandling.core.defaultnodesettings.status.StatusView;

/**
 * Represents the (likely reduced) configuration of input and output nodes. Also provides the functionality to
 * programmatically add the pre-configured input/output nodes to a workflow.
 *
 * Possible future TODO: split into model, view, etc.
 *
 * @author Martin Horn, KNIME GmbH, Konstanz, Germany
 * @author Dionysios Stolis, KNIME GmbH, Berlin, Germany
 */
abstract class IONodeConfig implements NodeSettingsConfigurator{

    private static final NodeLogger LOGGER = NodeLogger.getLogger(IONodeConfig.class);

    /**
     * Helper to configure input or output nodes.
     *
     * @param wfm the workflow to add the nodes to
     * @param nodeID a connected {@link NodeID}
     * @param useV2SmartInOutNames true to use non-fully qualified parameter names (false for old deprecated node only)
     * @param dataTable optional {@link DataTable} for specific implementations
     * @throws InvalidSettingsException if the configuration failed
     */
    protected void configureIONode(final WorkflowManager wfm, final NodeID nodeID, final boolean useV2SmartInOutNames,
        final DataTable dataTable) throws InvalidSettingsException {
        NodeSettings settings = new NodeSettings("root");
        wfm.saveNodeSettings(nodeID, settings);
        NodeSettingsWO modelSettings = settings.addNodeSettings("model");
        if (this instanceof DataTableConfigurator) {
            ((DataTableConfigurator)this).saveActualNodeSettingsTo(modelSettings, dataTable, useV2SmartInOutNames);
        } else {
            saveActualNodeSettingsTo(modelSettings, useV2SmartInOutNames);
        }
        wfm.loadNodeSettings(nodeID, settings);
    }

    /**
     * Helper to add and connect input or output nodes.
     *
     * @param wfm the workflow to add the nodes to
     * @param portType specifies the {@link PortType} of the port
     * @param ports a stream of {@link PortID}
     * @param x the x coordinate
     * @param y the y coordinate
     * @return a connected {@link NodeID}
     * @throws InvalidSettingsException if the configuration failed
     */
    protected NodeID addAndConnectIONode(final WorkflowManager wfm, final PortType portType, final Stream<PortID> ports,
        final int x, final int y) throws InvalidSettingsException {
        NodeID nodeID = wfm.createAndAddNode(createNodeFactory(portType));
        NodeContainer nc = wfm.getNodeContainer(nodeID);
        nc.setUIInformation(NodeUIInformation.builder().setNodeLocation(x, y, -1, -1).build());

        ports.forEach(p -> addConnection(wfm, p, nodeID));

        return nodeID;
    }

    private String m_paramName = getDefaultParameterName();

    private JTextField m_dlgParamName;

    private JPanel m_panel;

    private StatusView m_status;

    /**
     * Gets the already created or creates a {@link JPanel} to configure the respective node.
     *
     * @return the panel
     */
    JPanel getOrCreateJPanel() {
        if (m_panel == null) {
            m_panel = new JPanel(new GridBagLayout());
            GridBagConstraints gbc = new GridBagConstraints();
            gbc.gridx = 0;
            gbc.gridy = 0;
            gbc.anchor = GridBagConstraints.BASELINE_LEADING;

            final JPanel param = new JPanel(new GridBagLayout());
            param.add(new JLabel("Parameter name "), gbc);
            m_dlgParamName = new JTextField(20);
            m_dlgParamName.setText("param");
            m_dlgParamName.setText(m_paramName);
            m_paramName = null;
            gbc.gridx++;
            param.add(m_dlgParamName, gbc);
            m_panel.add(param, gbc);

            final JPanel status = new JPanel(new GridBagLayout());
            m_status = new StatusView(400);
            final JPanel statusLabel = m_status.getPanel();
            status.add(statusLabel, gbc);
            gbc.gridy++;
            m_panel.add(status, gbc);

            m_panel.add(Box.createVerticalGlue());

            updateStatus();
            m_dlgParamName.addActionListener(e -> updateStatus());
            m_dlgParamName.getDocument().addDocumentListener(new DocumentListener() {
                @Override
                public void changedUpdate(final DocumentEvent e) {
                    updateStatus();
                }

                @Override
                public void insertUpdate(final DocumentEvent e) {
                    updateStatus();
                }

                @Override
                public void removeUpdate(final DocumentEvent e) {
                    updateStatus();
                }
            });
        }
        return m_panel;
    }

    /**
     * @return the name of the node (will, e.g., appear in the drop-down list of the respective dialog component)
     */
    abstract String getNodeName();

    /**
     * @param portType the {@link PortType} of the port
     * @return the node factory instance of the represented node
     * @throws InvalidSettingsException
     */
    protected abstract NodeFactory<? extends NodeModel> createNodeFactory(PortType portType) throws InvalidSettingsException;

    /**
     * Connects the node to the given port.
     *
     * @param wfm the parent workflow
     * @param p the port to connect to/from
     * @param nodeID the node to connect
     */
    protected abstract void addConnection(WorkflowManager wfm, PortID p, NodeID nodeID);

    /**
     * All container input and output nodes have one configuration in common: the parameter name.
     *
     * @return the configured parameter name
     */
    protected String getParameterName() {
        if (m_dlgParamName != null) {
            return m_dlgParamName.getText();
        } else {
            return m_paramName;
        }
    }

    void saveSettingsTo(final NodeSettingsWO settings) {
        settings.addString("param_name", getParameterName());
    }

    void loadSettingsFrom(final NodeSettingsRO settings) {
        m_paramName = settings.getString("param_name", getDefaultParameterName());
    }

    /**
     * @return the default parameter name
     */
    protected abstract String getDefaultParameterName();

    private void updateStatus() {
        try {
            validateSettings();
            m_status.clearStatus();
        } catch (InvalidSettingsException e) { // NOSONAR -- swallow exception, handled in UI
            m_status.setStatus(new DefaultStatusMessage(StatusMessage.MessageType.ERROR, e.getMessage()));
        }
    }

    void validateSettings() throws InvalidSettingsException {
        String param = getParameterName();
        CheckUtils.checkSetting(StringUtils.isNotEmpty(param), "Parameter name must not be empty");
        CheckUtils.checkSetting(DialogNode.PARAMETER_NAME_PATTERN.matcher(param).matches(),
            "Parameter doesn't match pattern - must start with character, followed by other characters, digits, "
                + "or single dashes or underscores:\n  Input: %s\n  Pattern: %s",
            param, DialogNode.PARAMETER_NAME_PATTERN.pattern());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean equals(final Object obj) {
        if (!(obj instanceof IONodeConfig)) {
            return false;
        }
        IONodeConfig other = (IONodeConfig)obj;
        return new EqualsBuilder().append(getNodeName(), other.getNodeName())
            .append(getParameterName(), other.getParameterName()).build();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int hashCode() {
        return new HashCodeBuilder().append(getNodeName()).append(getParameterName()).build();
    }
}
