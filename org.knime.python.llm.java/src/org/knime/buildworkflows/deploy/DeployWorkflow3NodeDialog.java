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
 *   Nov 22, 2020 (Mark Ortmann, KNIME GmbH, Berlin, Germany): created
 */
package org.knime.python.llm.java.deploy;

import java.awt.Component;
import java.awt.GridBagLayout;
import java.util.Arrays;
import java.util.stream.Stream;

import javax.swing.BorderFactory;
import javax.swing.ButtonGroup;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;

import org.knime.python.llm.java.ExistsOption;
import org.knime.python.llm.java.writer.DialogComponentIONodes;
import org.knime.core.node.FlowVariableModel;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeDialogPane;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.NotConfigurableException;
import org.knime.core.node.context.ports.PortsConfiguration;
import org.knime.core.node.defaultnodesettings.DialogComponentBoolean;
import org.knime.core.node.defaultnodesettings.DialogComponentButtonGroup;
import org.knime.core.node.defaultnodesettings.DialogComponentString;
import org.knime.core.node.defaultnodesettings.SettingsModelBoolean;
import org.knime.core.node.port.PortObjectSpec;
import org.knime.core.node.workflow.capture.WorkflowPortObjectSpec;
import org.knime.filehandling.core.data.location.variable.FSLocationVariableType;
import org.knime.filehandling.core.defaultnodesettings.filechooser.writer.DialogComponentWriterFileChooser;
import org.knime.filehandling.core.defaultnodesettings.filechooser.writer.SettingsModelWriterFileChooser;
import org.knime.filehandling.core.util.GBCBuilder;

/**
 * Dialog of the "Deploy Workflow to Server" node.
 *
 * @author Mark Ortmann, KNIME GmbH, Berlin, Germany
 */
final class DeployWorkflow3NodeDialog extends NodeDialogPane {

    private final DeployWorkflow3Config m_cfg;

    private final DialogComponentWriterFileChooser m_fileChooser;

    private final DialogComponentButtonGroup m_workflowExists;

    private final JRadioButton m_defaultWorkflowRadio;

    private final JRadioButton m_customWorkflowRadio;

    private final JLabel m_defaultWorkflowName;

    private final DialogComponentString m_customWorkflowName;

    private final DialogComponentBoolean m_createSnapshot;

    private final DialogComponentString m_snapshotName;

    private final DialogComponentIONodes m_ioNodes;

    private final int m_workflowInputPortIndex;

    /**
     * @see DeployWorkflow3Config#m_doRemoveTemplateLinks
     */
    private final DialogComponentBoolean m_doRemoveTemplateLinks;

    /**
     * @see DeployWorkflow3Config#m_doUpdateTemplateLinks
     */
    private final DialogComponentBoolean m_doUpdateTemplateLinks;

    DeployWorkflow3NodeDialog(final PortsConfiguration portsConfig) {
        m_cfg = new DeployWorkflow3Config(portsConfig);

        SettingsModelWriterFileChooser writerModel = m_cfg.getFileChooserModel();
        final FlowVariableModel writeFvm = createFlowVariableModel(
            Stream.concat(Stream.of(DeployWorkflow3Config.CFG_SUB_SETTINGS),
                Arrays.stream(writerModel.getKeysForFSLocation())).toArray(String[]::new),
            FSLocationVariableType.INSTANCE);
        m_fileChooser =
            new DialogComponentWriterFileChooser(writerModel, "workflow_writer", writeFvm);

        m_workflowExists =
            new DialogComponentButtonGroup(m_cfg.getWorkflowExistsModel(), null, false, ExistsOption.values());

        m_defaultWorkflowName = new JLabel("");

        m_defaultWorkflowRadio = new JRadioButton("Use default workflow name");
        m_customWorkflowRadio = new JRadioButton("Use custom workflow name");
        final ButtonGroup btnGroup = new ButtonGroup();
        btnGroup.add(m_defaultWorkflowRadio);
        btnGroup.add(m_customWorkflowRadio);

        m_customWorkflowName = new DialogComponentString(m_cfg.getCustomWorkflowNameModel(), "");

        SettingsModelBoolean createSnapshotModel = m_cfg.createSnapshotModel();
        m_createSnapshot = new DialogComponentBoolean(createSnapshotModel, "Create snapshot");
        m_doRemoveTemplateLinks = new DialogComponentBoolean(m_cfg.getDoRemoveTemplateLinksModel(), "Disconnect links of components and metanodes");
        m_doUpdateTemplateLinks = new DialogComponentBoolean(m_cfg.getDoUpdateTemplateLinksModel(), "Update links of components and metanodes");
        m_snapshotName = new DialogComponentString(m_cfg.getSnapshotNameModel(), "Snapshot comment");
        // the port always exists and is unique therefore this cannot cause a NPE
        m_workflowInputPortIndex =
            portsConfig.getInputPortLocation().get(DeployWorkflow3NodeFactory.WORKFLOW_GRP_ID)[0];
        m_ioNodes = new DialogComponentIONodes(m_cfg.getIOModel(), m_workflowInputPortIndex);

        // add listener
        m_defaultWorkflowRadio.addActionListener(l -> toggleWorkflowName());
        m_customWorkflowRadio.addActionListener(l -> toggleWorkflowName());
        createSnapshotModel.addChangeListener(l -> toggleSnapshotName());

        addTab("Settings", createSettingsTab());
        addTab("Inputs and Outputs", m_ioNodes.getComponentPanel());
        addTab("Deployment Options", createDeploymentOptionsTab());
    }

    private void toggleWorkflowName() {
        m_customWorkflowName.getModel().setEnabled(m_customWorkflowRadio.isSelected());
    }

    private void toggleSnapshotName() {
        m_snapshotName.getModel().setEnabled(((SettingsModelBoolean)m_createSnapshot.getModel()).getBooleanValue());
    }

     private Component createDeploymentOptionsTab() {
        final JPanel p = new JPanel(new GridBagLayout());
        final GBCBuilder gbc = new GBCBuilder().resetPos().anchorLineStart().weight(1, 0).fillHorizontal();

        p.add(createSnapshotPanel(), gbc.build());

        gbc.incY();
        p.add(createSegmentManipulationPanel(), gbc.build());

        gbc.incY().weight(1, 1).fillBoth().insetTop(-10);
        p.add(new JPanel(), gbc.build());

        return p;
    }

    private Component createSettingsTab() {
        final JPanel p = new JPanel(new GridBagLayout());

        final GBCBuilder gbc = new GBCBuilder().resetPos().anchorLineStart().weight(1, 0).fillHorizontal();
        p.add(createFileChooserPanel(), gbc.build());

        gbc.incY();
        p.add(createWorkflowPanel(), gbc.build());


        gbc.incY().weight(1, 1).fillBoth().insetTop(-10);
        p.add(new JPanel(), gbc.build());

        return p;
    }

    private Component createFileChooserPanel() {
        final JPanel p = new JPanel(new GridBagLayout());
        p.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(), "Output folder"));

        final GBCBuilder gbc = new GBCBuilder().resetPos().anchorLineStart().weight(1, 0).fillHorizontal().insetLeft(4);
        p.add(m_fileChooser.getComponentPanel(), gbc.build());

        return p;
    }

    private Component createWorkflowPanel() {
        final JPanel p = new JPanel(new GridBagLayout());
        p.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(), "Workflow"));

        final GBCBuilder gbc = new GBCBuilder().resetPos().anchorLineStart().weight(0, 0).fillNone().insetLeft(4);
        p.add(new JLabel("If workflow exists"), gbc.build());

        gbc.incX().insetLeft(0);
        p.add(m_workflowExists.getComponentPanel(), gbc.build());

        gbc.resetX().incY().setWidth(1);
        p.add(m_defaultWorkflowRadio, gbc.build());

        gbc.incX().insetLeft(10);
        p.add(m_defaultWorkflowName, gbc.build());

        gbc.resetX().incY().insetLeft(0);
        p.add(m_customWorkflowRadio, gbc.build());

        gbc.incX();
        p.add(m_customWorkflowName.getComponentPanel(), gbc.build());

        gbc.resetX().incY().weight(1, 1).setWidth(2).insetTop(-10).fillBoth();
        p.add(new JPanel(), gbc.build());

        return p;
    }

    private Component createSegmentManipulationPanel() {
        final JPanel p = new JPanel(new GridBagLayout());
        p.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(),
            "Modify workflow segment before deployment"));

        final GBCBuilder gbc = new GBCBuilder().resetPos().anchorLineStart().weight(0, 0).fillNone();
        p.add(m_doUpdateTemplateLinks.getComponentPanel(), gbc.build());

        gbc.incY();
        p.add(m_doRemoveTemplateLinks.getComponentPanel(), gbc.build());

        gbc.resetX().incY().weight(1, 1).insetTop(-10).fillBoth();
        p.add(new JPanel(), gbc.build());

        return p;
    }

    private Component createSnapshotPanel() {
        final JPanel p = new JPanel(new GridBagLayout());
        p.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(), "Snapshot options"));

        final GBCBuilder gbc = new GBCBuilder().resetPos().anchorLineStart().weight(0, 0).fillNone();
        p.add(m_createSnapshot.getComponentPanel(), gbc.build());

        gbc.incY();
        p.add(m_snapshotName.getComponentPanel(), gbc.build());

        gbc.resetX().incY().weight(1, 1).insetTop(-10).fillBoth();
        p.add(new JPanel(), gbc.build());

        return p;

    }

    @Override
    protected void saveSettingsTo(final NodeSettingsWO settings) throws InvalidSettingsException {
        final NodeSettingsWO subSettings = settings.addNodeSettings(DeployWorkflow3Config.CFG_SUB_SETTINGS);
        m_fileChooser.saveSettingsTo(subSettings);
        m_workflowExists.saveSettingsTo(subSettings);
        m_cfg.saveUseCustomWorkflowNameInDialog(subSettings, m_customWorkflowRadio.isSelected());
        m_customWorkflowName.saveSettingsTo(subSettings);
        m_createSnapshot.saveSettingsTo(subSettings);
        m_doRemoveTemplateLinks.saveSettingsTo(subSettings);
        m_doUpdateTemplateLinks.saveSettingsTo(subSettings);
        m_snapshotName.saveSettingsTo(subSettings);
        m_ioNodes.saveSettingsTo(settings);
    }

    @Override
    protected void loadSettingsFrom(final NodeSettingsRO settings, final PortObjectSpec[] specs)
        throws NotConfigurableException {
        NodeSettingsRO subSettings;
        try {
            subSettings = settings.getNodeSettings(DeployWorkflow3Config.CFG_SUB_SETTINGS);
        } catch (InvalidSettingsException e) { // NOSONAR loading defaults is always done without error reporting
            subSettings = settings;
        }
        m_fileChooser.loadSettingsFrom(subSettings, specs);
        m_workflowExists.loadSettingsFrom(subSettings, specs);
        m_cfg.loadUseCustomWorkflowNameInDialog(subSettings);
        m_customWorkflowName.loadSettingsFrom(subSettings, specs);
        m_createSnapshot.loadSettingsFrom(subSettings, specs);
        m_doRemoveTemplateLinks.loadSettingsFrom(subSettings, specs);
        m_doUpdateTemplateLinks.loadSettingsFrom(subSettings, specs);
        m_snapshotName.loadSettingsFrom(subSettings, specs);
        m_ioNodes.loadSettingsFrom(settings, specs);

        // update default workflow label
        m_defaultWorkflowName.setText(
            DeployWorkflow3NodeModel.getDefaultWorkflowName((WorkflowPortObjectSpec)specs[m_workflowInputPortIndex]));
        // enable / disable models
        m_defaultWorkflowRadio.setSelected(!m_cfg.useCustomWorkflowName());
        m_customWorkflowRadio.setSelected(m_cfg.useCustomWorkflowName());
        toggleSnapshotName();
        toggleWorkflowName();
    }

    @Override
    public void onClose() {
        m_fileChooser.onClose();
    }
}
