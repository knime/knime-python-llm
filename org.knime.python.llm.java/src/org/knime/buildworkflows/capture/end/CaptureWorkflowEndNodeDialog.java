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
 *   Feb 11, 2020 (hornm): created
 */
package org.knime.python.llm.java.capture.end;

import static org.knime.python.llm.java.capture.end.CaptureWorkflowEndNodeModel.loadAndFillInputOutputIDs;
import static org.knime.python.llm.java.capture.end.CaptureWorkflowEndNodeModel.saveInputOutputIDs;
import static org.knime.core.node.workflow.capture.WorkflowPortObjectSpec.ensureInputIDsCount;
import static org.knime.core.node.workflow.capture.WorkflowPortObjectSpec.ensureOutputIDsCount;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Dimension;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JPanel;

import org.knime.python.llm.java.util.ValidatedWorkflowNameField;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeDialogPane;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.NotConfigurableException;
import org.knime.core.node.defaultnodesettings.DialogComponentBoolean;
import org.knime.core.node.defaultnodesettings.DialogComponentNumber;
import org.knime.core.node.defaultnodesettings.SettingsModelBoolean;
import org.knime.core.node.port.PortObjectSpec;
import org.knime.core.node.workflow.NodeContainer;
import org.knime.core.node.workflow.NodeContext;
import org.knime.core.node.workflow.WorkflowCaptureOperation;
import org.knime.core.node.workflow.capture.WorkflowSegment.Input;
import org.knime.core.node.workflow.capture.WorkflowSegment.Output;

/**
 *
 * @author Martin Horn, KNIME GmbH, Konstanz, Germany
 */
final class CaptureWorkflowEndNodeDialog extends NodeDialogPane {

    private static Component group(final String label, final Component... components) {
        final JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        panel.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(), label));
        for (final Component component : components) {
            panel.add(alignLeft(component));
        }
        return panel;
    }

    private static Component alignLeft(final Component component) {
        final Box box = Box.createHorizontalBox();
        component.setMaximumSize(new Dimension(component.getPreferredSize().width, component.getMaximumSize().height));
        box.add(component);
        box.add(Box.createHorizontalGlue());
        return box;
    }

    private final ValidatedWorkflowNameField m_customName =
        new ValidatedWorkflowNameField(CaptureWorkflowEndNodeModel.createCustomWorkflowNameModel(), "Custom workflow " +
                "name", true);

    private final DialogComponentNumber m_maxNumOfRowsModel = new DialogComponentNumber(
        CaptureWorkflowEndNodeModel.createMaxNumOfRowsModel(), "Maximum numbers of rows to store", 1);

    private final DialogComponentBoolean m_addInputDataModel =
        new DialogComponentBoolean(CaptureWorkflowEndNodeModel.createAddInputDataModel(), "Store input tables");

    private final DialogComponentBoolean m_exportAllVariablesModel =
            new DialogComponentBoolean(CaptureWorkflowEndNodeModel.createExportVariablesModel(), "Propagate variables");

    private final DialogComponentBoolean m_doRemoveTemplateLinks =
            new DialogComponentBoolean(CaptureWorkflowEndNodeModel.createDoRemoveTemplateLinksModel(), "Disconnect links of components and metanodes");

    private final JPanel m_ioIds;

    private InputOutputIDsPanel m_idsPanel;

    CaptureWorkflowEndNodeDialog() {
        JPanel options = new JPanel();
        options.setLayout(new BoxLayout(options, BoxLayout.PAGE_AXIS));

        options
            .add(group("Input data", m_addInputDataModel.getComponentPanel(), m_maxNumOfRowsModel.getComponentPanel()));

        options.add(group("Variables", m_exportAllVariablesModel.getComponentPanel()));

        options.add(group("Modify captured segment", m_doRemoveTemplateLinks.getComponentPanel(),
                m_customName.getComponentPanel()));

        addTab("Settings", options);

        m_addInputDataModel.getModel().addChangeListener(l -> m_maxNumOfRowsModel.getModel()
            .setEnabled(((SettingsModelBoolean)m_addInputDataModel.getModel()).getBooleanValue()));

        m_ioIds = new JPanel(new BorderLayout());
        addTab("Input and Output IDs", m_ioIds);
    }

    @Override
    protected void loadSettingsFrom(final NodeSettingsRO settings, final PortObjectSpec[] specs)
        throws NotConfigurableException {
        m_customName.loadSettingsFrom(settings, specs);
        m_maxNumOfRowsModel.loadSettingsFrom(settings, specs);
        m_addInputDataModel.loadSettingsFrom(settings, specs);
        m_exportAllVariablesModel.loadSettingsFrom(settings, specs);
        m_doRemoveTemplateLinks.loadSettingsFrom(settings, specs);

        NodeContainer nc = NodeContext.getContext().getNodeContainer();
        if (nc == null) {
            throw new NotConfigurableException("No node context available.");
        }

        m_maxNumOfRowsModel.getModel()
            .setEnabled(((SettingsModelBoolean)m_addInputDataModel.getModel()).getBooleanValue());

        List<String> customInputIDs = new ArrayList<>();
        List<String> customOutputIDs = new ArrayList<>();
        try {
            loadAndFillInputOutputIDs(settings, customInputIDs, customOutputIDs);
        } catch (InvalidSettingsException e) {
            getLogger().warn("Settings couldn't be load for dialog. Ignored.", e);
        }

        WorkflowCaptureOperation captureOp;
        try {
            captureOp = nc.getParent().createCaptureOperationFor(nc.getID());
            m_ioIds.removeAll();

            //get 'connected' inputs and outputs only
            List<Input> inputs = captureOp.getInputs().stream().filter(Input::isConnected).collect(Collectors.toList());
            List<Output> outputs =
                captureOp.getOutputs().stream().filter(Output::isConnected).collect(Collectors.toList());

            customInputIDs = ensureInputIDsCount(customInputIDs, inputs.size());
            customOutputIDs = ensureOutputIDsCount(customOutputIDs, outputs.size());
            m_idsPanel = new InputOutputIDsPanel(customInputIDs, customOutputIDs);
            m_ioIds.add(m_idsPanel, BorderLayout.CENTER);
        } catch (Exception e) {
            throw new NotConfigurableException(e.getMessage(), e);
        }
    }

    @Override
    protected void saveSettingsTo(final NodeSettingsWO settings) throws InvalidSettingsException {
        m_customName.saveSettingsTo(settings);
        m_maxNumOfRowsModel.saveSettingsTo(settings);
        m_addInputDataModel.saveSettingsTo(settings);
        m_exportAllVariablesModel.saveSettingsTo(settings);
        m_doRemoveTemplateLinks.saveSettingsTo(settings);

        List<String> inputIDs = m_idsPanel.getInputIDs();
        List<String> outputIDs = m_idsPanel.getOutputIDs();
        boolean inDup = hasDuplicates(inputIDs);
        boolean outDup = hasDuplicates(outputIDs);
        m_idsPanel.setInputsOutputsInvalid(inDup, outDup);
        if (inDup || outDup) {
            throw new InvalidSettingsException("Duplicate input or output IDs");
        }
        saveInputOutputIDs(settings, inputIDs, outputIDs);
    }

    private static boolean hasDuplicates(final List<String> ids) {
        Set<String> set = new HashSet<>(ids);
        return set.size() != ids.size();
    }
}
