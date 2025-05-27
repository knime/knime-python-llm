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
 *   Dec 9, 2019 (Adrian Nembach, KNIME GmbH, Konstanz, Germany): created
 */
package org.knime.python.llm.java.capture.end;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

import org.apache.commons.lang3.StringUtils;
import org.knime.python.llm.java.manipulate.WorkflowSegmentManipulations;
import org.knime.core.data.DataTable;
import org.knime.core.data.container.CloseableRowIterator;
import org.knime.core.data.container.filter.TableFilter;
import org.knime.core.node.BufferedDataContainer;
import org.knime.core.node.BufferedDataTable;
import org.knime.core.node.CanceledExecutionException;
import org.knime.core.node.ExecutionContext;
import org.knime.core.node.ExecutionMonitor;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeModel;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.context.ports.PortsConfiguration;
import org.knime.core.node.defaultnodesettings.SettingsModelBoolean;
import org.knime.core.node.defaultnodesettings.SettingsModelInteger;
import org.knime.core.node.defaultnodesettings.SettingsModelIntegerBounded;
import org.knime.core.node.defaultnodesettings.SettingsModelString;
import org.knime.core.node.port.PortObject;
import org.knime.core.node.port.PortObjectSpec;
import org.knime.core.node.workflow.CaptureWorkflowEndNode;
import org.knime.core.node.workflow.CaptureWorkflowStartNode;
import org.knime.core.node.workflow.ConnectionContainer;
import org.knime.core.node.workflow.FlowVariable;
import org.knime.core.node.workflow.NodeContainer;
import org.knime.core.node.workflow.NodeContext;
import org.knime.core.node.workflow.VariableType;
import org.knime.core.node.workflow.WorkflowManager;
import org.knime.core.node.workflow.capture.BuildWorkflowsUtil;
import org.knime.core.node.workflow.capture.WorkflowPortObject;
import org.knime.core.node.workflow.capture.WorkflowPortObjectSpec;
import org.knime.core.node.workflow.capture.WorkflowSegment;
import org.knime.core.node.workflow.capture.WorkflowSegment.Input;
import org.knime.core.node.workflow.capture.WorkflowSegment.PortID;

/**
 * The node model of the Capture Workflow End node that marks the end of a captured workflow segment.
 *
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 */
final class CaptureWorkflowEndNodeModel extends NodeModel implements CaptureWorkflowEndNode {

    static SettingsModelString createCustomWorkflowNameModel() {
        return new SettingsModelString("custom_workflow_name", "");
    }

    static SettingsModelBoolean createAddInputDataModel() {
        return new SettingsModelBoolean("add_input_data", false);
    }

    static SettingsModelBoolean createExportVariablesModel() {
        return new SettingsModelBoolean("export_variables", true);
    }

    static SettingsModelIntegerBounded createMaxNumOfRowsModel() {
        return new SettingsModelIntegerBounded("max_num_rows", 10, 1, Integer.MAX_VALUE);
    }

    public static SettingsModelBoolean createDoRemoveTemplateLinksModel() {
        return new SettingsModelBoolean("do_remove_template_links", false);
    }

    private final SettingsModelString m_customWorkflowName = createCustomWorkflowNameModel();

    private final SettingsModelBoolean m_addInputData = createAddInputDataModel();

    private final SettingsModelInteger m_maxNumRows = createMaxNumOfRowsModel();

    private final SettingsModelBoolean m_exportVariables = createExportVariablesModel();

    /**
     * Whether to remove links of linked metanodes and components upon capture.
     * @since 4.5
     */
    private final SettingsModelBoolean m_doRemoveTemplateLinks = createDoRemoveTemplateLinksModel();

    private WorkflowSegment m_lastSegment;

    private final List<String> m_inputIDs = new ArrayList<>();

    private final List<String> m_outputIDs = new ArrayList<>();

    /**
     * @param portsConfiguration the {@link PortsConfiguration} provided by the user
     */
    protected CaptureWorkflowEndNodeModel(final PortsConfiguration portsConfiguration) {
        super(portsConfiguration.getInputPorts(), portsConfiguration.getOutputPorts());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected PortObjectSpec[] configure(final PortObjectSpec[] inSpecs) throws InvalidSettingsException {
        final WorkflowPortObjectSpec spec = capture();
        return Stream.concat(Arrays.stream(inSpecs), Stream.of(spec)).toArray(PortObjectSpec[]::new);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected PortObject[] execute(final PortObject[] inObjects, final ExecutionContext exec) throws Exception {
        final WorkflowPortObjectSpec spec = capture();
        Map<String, DataTable> inputData = null;
        if (m_addInputData.getBooleanValue()) {
            inputData = getInputData(m_lastSegment.getConnectedInputs(), m_inputIDs, m_maxNumRows.getIntValue(), exec);
        }
        final WorkflowPortObject po = new WorkflowPortObject(spec, inputData);
        exportVariables();
        return Stream.concat(Arrays.stream(inObjects), Stream.of(po)).toArray(PortObject[]::new);
    }

    /** Captures the segment the current node is part of. */
    private WorkflowPortObjectSpec capture() throws InvalidSettingsException {
        removeSegment();

        final String customWorkflowName = StringUtils.defaultIfEmpty(m_customWorkflowName.getStringValue(), null);
        if (customWorkflowName != null) {
            Optional<String> err = BuildWorkflowsUtil.validateCustomWorkflowName(customWorkflowName, true, false);
            if (err.isPresent()) {
                throw new InvalidSettingsException(err.get());
            }
        }

        if (!getScopeStartNode(CaptureWorkflowStartNode.class).isPresent()) {
            throw createMessageBuilder().withSummary("No corresponding 'Capture Workflow Start' node found.") //
                .addTextIssue("This node is not closing a workflow segment defined by a 'Capture Workflow Start'.") //
                .addResolutions("If missing, add a 'Capture Workflow Start' to define the segment start.") //
                .addResolutions("Check no other unclosed scope, e.g. Loop, is contained in the workflow segment.") //
                .build().orElseThrow().toInvalidSettingsException();
        }

        final NodeContainer container = NodeContext.getContext().getNodeContainer();
        final WorkflowManager manager = container.getParent();
        WorkflowSegment wfs;
        try {
            wfs = manager.createCaptureOperationFor(container.getID()).capture(m_customWorkflowName.getStringValue());
        } catch (Exception e) { // NOSONAR
            throw createMessageBuilder().withSummary("Capturing the workflow failed")
                .addTextIssue("Unable to capture the corresponding workflow segment:%n%s".formatted(e.getMessage())) //
                .addResolutions("Review the scope of the captured workflow segment.") //
                .build().orElseThrow().toInvalidSettingsException(e);
        }

        if (m_doRemoveTemplateLinks.getBooleanValue()) {
            try {
                WorkflowSegmentManipulations.REMOVE_TEMPLATE_LINKS.apply(wfs);
            } catch (Exception e) {  // NOSONAR: Exception is handled
                wfs.disposeWorkflow();
                throw new InvalidSettingsException(e.getCause());
            }
        }
        m_lastSegment = wfs;
        return new WorkflowPortObjectSpec(m_lastSegment, customWorkflowName, m_inputIDs, m_outputIDs);

    }

    /**
     * Retrieves the input tables for the given inputs (if they accept tables).
     */
    private static Map<String, DataTable> getInputData(final List<Input> inputs, final List<String> inputIDs,
        final int numRowsToStore, final ExecutionContext exec) throws CanceledExecutionException {
        WorkflowManager wfm = NodeContext.getContext().getNodeContainer().getParent();
        Map<String, DataTable> inputData = new HashMap<>();
        int i = 0;
        for (Input input : inputs) {
            if (input.getType().isPresent() && input.getType().get().equals(BufferedDataTable.TYPE)
                && !input.getConnectedPorts().isEmpty()) {
                PortID connectingPort = input.getConnectedPorts().iterator().next();
                ConnectionContainer cc = wfm.getIncomingConnectionFor(
                    connectingPort.getNodeIDSuffix().prependParent(wfm.getID()), connectingPort.getIndex());
                BufferedDataTable table = (BufferedDataTable)wfm.getNodeContainer(cc.getSource())
                    .getOutPort(cc.getSourcePort()).getPortObject();
                if (numRowsToStore > 0) {
                    BufferedDataContainer container = exec.createDataContainer(table.getDataTableSpec());
                    try (CloseableRowIterator iterator = table
                        .filter(TableFilter.filterRowsToIndex(Math.min(numRowsToStore, table.size()) - 1)).iterator()) {
                        while (iterator.hasNext()) {
                            exec.checkCanceled();
                            container.addRowToTable(iterator.next());
                        }
                    }
                    container.close();
                    table = container.getTable();
                }
                inputData.put(inputIDs.get(i), table);
            }
            i++;
        }
        return inputData;
    }

    private void removeSegment() {
        if (m_lastSegment != null) {
            m_lastSegment.disposeWorkflow();
            m_lastSegment = null;
        }
    }

    /**
     * Calls {@link #pushFlowVariable(String, VariableType, Object)} for variables defined within
     * the scope IFF the corresponding configuration is set.
     */
    @SuppressWarnings({"rawtypes", "unchecked", "java:S3740"})
    private void exportVariables() {
        if (m_exportVariables.getBooleanValue()) {
            for (FlowVariable v : getVariablesDefinedInScope()) {
                VariableType t = v.getVariableType();
                pushFlowVariable(v.getName(), t, v.getValue(t));
            }
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void loadInternals(final File nodeInternDir, final ExecutionMonitor exec)
        throws IOException, CanceledExecutionException {
        // no internals
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void saveInternals(final File nodeInternDir, final ExecutionMonitor exec)
        throws IOException, CanceledExecutionException {
        // no internals
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void saveSettingsTo(final NodeSettingsWO settings) {
        if (!m_customWorkflowName.getStringValue().isEmpty()) {
            m_customWorkflowName.saveSettingsTo(settings);
        }
        m_addInputData.saveSettingsTo(settings);
        m_maxNumRows.saveSettingsTo(settings);
        m_exportVariables.saveSettingsTo(settings);
        m_doRemoveTemplateLinks.saveSettingsTo(settings);
        saveInputOutputIDs(settings, m_inputIDs, m_outputIDs);
    }

    static void saveInputOutputIDs(final NodeSettingsWO settings, final List<String> inputIDs,
        final List<String> outputIDs) {
        if (inputIDs.isEmpty() && outputIDs.isEmpty()) {
            return;
        }
        NodeSettingsWO subSettings = settings.addNodeSettings("input_ids");
        subSettings.addInt("num_ids", inputIDs.size());
        for (int i = 0; i < inputIDs.size(); i++) {
            subSettings.addString("id_" + i, inputIDs.get(i));
        }

        subSettings = settings.addNodeSettings("output_ids");
        subSettings.addInt("num_ids", outputIDs.size());
        for (int i = 0; i < outputIDs.size(); i++) {
            subSettings.addString("id_" + i, outputIDs.get(i));
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void validateSettings(final NodeSettingsRO settings) throws InvalidSettingsException {
        m_addInputData.validateSettings(settings);
        m_maxNumRows.validateSettings(settings);
        // added in 4.4 (but if present it needs to be "valid")
        if (settings.containsKey(m_exportVariables.getConfigName())) {
            m_exportVariables.validateSettings(settings);
        }
        if (settings.containsKey(m_doRemoveTemplateLinks.getConfigName())) {
            m_doRemoveTemplateLinks.validateSettings(settings);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void loadValidatedSettingsFrom(final NodeSettingsRO settings) throws InvalidSettingsException {
        if (settings.containsKey(m_customWorkflowName.getKey())) {
            m_customWorkflowName.loadSettingsFrom(settings);
        }
        m_addInputData.loadSettingsFrom(settings);
        m_maxNumRows.loadSettingsFrom(settings);
        // setting was added in 4.4 (AP-16448)
        if (settings.containsKey(m_exportVariables.getConfigName())) {
            m_exportVariables.loadSettingsFrom(settings);
        } else {
            // was 'false' in prior versions
            m_exportVariables.setBooleanValue(false);
        }
        if (settings.containsKey(m_doRemoveTemplateLinks.getConfigName())) {
            m_doRemoveTemplateLinks.loadSettingsFrom(settings);
        } else {
            m_doRemoveTemplateLinks.setBooleanValue(false);
        }
        m_inputIDs.clear();
        m_outputIDs.clear();
        loadAndFillInputOutputIDs(settings, m_inputIDs, m_outputIDs);
    }

    /**
     * Loads the port names and adds them to the supplied maps.
     *
     * @param settings to load the port names from
     * @param inPortNames the map the input port names are added to
     * @param outPortNames the map the output port names are added to
     * @throws InvalidSettingsException
     */
    static void loadAndFillInputOutputIDs(final NodeSettingsRO settings, final List<String> inputIDs,
        final List<String> outputIDs) throws InvalidSettingsException {

        if (settings.containsKey("input_ids")) {
            NodeSettingsRO subSettings = settings.getNodeSettings("input_ids");
            int num = subSettings.getInt("num_ids");
            for (int i = 0; i < num; i++) {
                inputIDs.add(subSettings.getString("id_" + i));
            }
        }

        if (settings.containsKey("output_ids")) {
            NodeSettingsRO subSettings = settings.getNodeSettings("output_ids");
            int num = subSettings.getInt("num_ids");
            for (int i = 0; i < num; i++) {
                outputIDs.add(subSettings.getString("id_" + i));
            }
        }

    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void reset() {
        removeSegment();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void onDispose() {
        removeSegment();
    }

}
