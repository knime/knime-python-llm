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
 *   9 Dec 2019 (Marc Bux, KNIME GmbH, Berlin, Germany): created
 */
package org.knime.python.llm.java.executor;

import static java.util.stream.Collectors.toList;

import java.util.Arrays;
import java.util.List;
import java.util.ListIterator;
import java.util.Objects;
import java.util.stream.IntStream;

import org.knime.python.llm.java.manipulate.WorkflowSegmentManipulations;
import org.knime.core.node.ExecutionContext;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.KNIMEException;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.context.ports.PortsConfiguration;
import org.knime.core.node.defaultnodesettings.SettingsModelBoolean;
import org.knime.core.node.port.PortObject;
import org.knime.core.node.port.PortObjectSpec;
import org.knime.core.node.port.PortType;
import org.knime.core.node.util.CheckUtils;
import org.knime.core.node.workflow.FlowVariable;
import org.knime.core.node.workflow.NodeContainer;
import org.knime.core.node.workflow.NodeContext;
import org.knime.core.node.workflow.VariableType;
import org.knime.core.node.workflow.capture.WorkflowPortObject;
import org.knime.core.node.workflow.capture.WorkflowPortObjectSpec;
import org.knime.core.node.workflow.capture.WorkflowSegment;
import org.knime.core.node.workflow.capture.WorkflowSegmentExecutor;
import org.knime.core.node.workflow.virtual.AbstractPortObjectRepositoryNodeModel;
import org.knime.core.util.Pair;

/**
 * @author Martin Horn, KNIME GmbH, Konstanz, Germany
 */
final class WorkflowExecutorNodeModel extends AbstractPortObjectRepositoryNodeModel {

    static final String CFG_DEBUG = "debug";

    private WorkflowSegmentExecutor m_executable;

    private boolean m_debug = false;

    private final SettingsModelBoolean m_doUpdateTemplateLinks = createDoUpdateTemplateLinksModel();

    private final SettingsModelBoolean m_doExecuteAllNodes = createExecuteAllNodesModel();

    public static SettingsModelBoolean createDoUpdateTemplateLinksModel() {
        return new SettingsModelBoolean("do_update_template_links", false);
    }

    public static SettingsModelBoolean createExecuteAllNodesModel() {
        return new SettingsModelBoolean("do_execute_entire_workflow", true);
    }

    WorkflowExecutorNodeModel(final PortsConfiguration portsConf) {
        super(portsConf.getInputPorts(), portsConf.getOutputPorts());
    }

    @Override
    protected PortObjectSpec[] configure(final PortObjectSpec[] inSpecs) throws InvalidSettingsException {
        WorkflowPortObjectSpec wpos = (WorkflowPortObjectSpec)inSpecs[0];
        NodeContainer nc = NodeContext.getContext().getNodeContainer();
        CheckUtils.checkArgumentNotNull(nc, "Not a local workflow");
        checkPortCompatibility(wpos, nc);
        return null; // NOSONAR
    }

    @Override
    protected PortObject[] execute(final PortObject[] inObjects, final ExecutionContext exec) throws Exception {
        WorkflowPortObject wpo = (WorkflowPortObject)inObjects[0];
        WorkflowSegment segment = wpo.getSpec().getWorkflowSegment();
        if (m_doUpdateTemplateLinks.getBooleanValue()) {
            WorkflowSegmentManipulations.UPDATE_LINKED_TEMPLATES.apply(segment);
        }

        WorkflowSegmentExecutor we = createWorkflowExecutable(wpo.getSpec());
        m_executable = we;
        boolean success = false;
        try {
            exec.setMessage("Executing workflow segment '" + wpo.getSpec().getWorkflowName() + "'");
            Pair<PortObject[], List<FlowVariable>> output =
                we.executeWorkflow(Arrays.copyOfRange(inObjects, 1, inObjects.length), exec);
            if (output.getFirst() == null || Arrays.stream(output.getFirst()).anyMatch(Objects::isNull)) {
                NodeContainer nc = NodeContext.getContext().getNodeContainer();
                String message = "Execution didn't finish successfully";
                if (!checkPortsCompatibility(workflowInputTypes(wpo.getSpec()), nodeInputTypes(nc), false)) {
                    message += " - node input(s) not compatible with workflow input(s)";
                }
                throw new IllegalStateException(message);
            }

            // Push flow variables, preserving the ordering.
            List<FlowVariable> vars = output.getSecond();
            ListIterator<FlowVariable> reverseIter = vars.listIterator(vars.size());
            while (reverseIter.hasPrevious()) {
                pushFlowVariableInternal(reverseIter.previous());
            }

            success = true;
            return output.getFirst();
        } finally {
            if (!m_debug || success) {
                disposeWorkflowExecutable();
            } else {
                m_executable.cancel();
            }
        }
    }

    @SuppressWarnings("unchecked")
    private <T> void pushFlowVariableInternal(final FlowVariable fv) {
        pushFlowVariable(fv.getName(), (VariableType<T>)fv.getVariableType(), (T)fv.getValue(fv.getVariableType()));
    }

    private WorkflowSegmentExecutor createWorkflowExecutable(final WorkflowPortObjectSpec spec)
        throws InvalidSettingsException, KNIMEException {
        disposeWorkflowExecutable();
        NodeContainer nc = NodeContext.getContext().getNodeContainer();
        CheckUtils.checkArgumentNotNull(nc, "Not a local workflow");
        checkPortCompatibility(spec, nc);
        m_executable = new WorkflowSegmentExecutor(spec.getWorkflowSegment(), spec.getWorkflowName(), nc, m_debug,
            m_doExecuteAllNodes.getBooleanValue(), this::setWarningMessage);
        return m_executable;
    }

    private void disposeWorkflowExecutable() {
        if (m_executable != null) {
            m_executable.dispose();
            m_executable = null;
        }
    }

    /**
     * Checks for compatibility of the node ports and the workflow inputs/outputs. The flow variable ports (0th index at
     * input and output) and workflow port (1st input) are not taken into account.
     *
     * @param spec
     * @param nc
     * @throws InvalidSettingsException if not compatible
     */
    static void checkPortCompatibility(final WorkflowPortObjectSpec spec, final NodeContainer nc)
        throws InvalidSettingsException {
        String configMessage = "Node needs to be re-configured.";
        if (!checkPortsCompatibility(workflowInputTypes(spec), nodeInputTypes(nc), true)) {
            throw new InvalidSettingsException(
                "The node inputs don't match with the workflow inputs. " + configMessage);
        }

        if (!checkPortsCompatibility(workflowOutputTypes(spec), nodeOutputTypes(nc), true)) {
            throw new InvalidSettingsException(
                "The node outputs don't match with the workflow outputs. " + configMessage);
        }
    }

    private static List<PortType> nodeInputTypes(final NodeContainer nc) {
        return IntStream.range(2, nc.getNrInPorts()).mapToObj(i -> nc.getInPort(i).getPortType()).collect(toList());
    }

    private static List<PortType> workflowInputTypes(final WorkflowPortObjectSpec spec) {
        return spec.getWorkflowSegment().getConnectedInputs().stream().map(i -> i.getType().get()).collect(toList());
    }

    private static List<PortType> nodeOutputTypes(final NodeContainer nc) {
        return IntStream.range(1, nc.getNrOutPorts()).mapToObj(i -> nc.getOutPort(i).getPortType()).collect(toList());
    }

    private static List<PortType> workflowOutputTypes(final WorkflowPortObjectSpec spec) {
        return spec.getWorkflowSegment().getConnectedOutputs().stream().map(i -> i.getType().get()).collect(toList());
    }

    /**
     * Checks that the two lists are compatible (same size and same port types) with one peculiarity: If a node port is
     * of type {@link PortObject#TYPE}, i.e. the generic port object, the respective workflow port can be of any port
     * type (if generic node ports are ignored for comparison).
     */
    private static boolean checkPortsCompatibility(final List<PortType> workflowPorts, final List<PortType> nodePorts,
        final boolean ignoreGenericNodePorts) {
        if (workflowPorts.size() != nodePorts.size()) {
            return false;
        }
        for (int i = 0; i < workflowPorts.size(); i++) {
            if (!workflowPorts.get(i).equals(nodePorts.get(i))
                && (!nodePorts.get(i).equals(PortObject.TYPE) || !ignoreGenericNodePorts)) {
                return false;
            }
        }
        return true;
    }

    @Override
    protected void saveSettingsTo(final NodeSettingsWO settings) {
        settings.addBoolean(CFG_DEBUG, m_debug);
        m_doUpdateTemplateLinks.saveSettingsTo(settings);
        m_doExecuteAllNodes.saveSettingsTo(settings);
    }

    @Override
    protected void validateSettings(final NodeSettingsRO settings) throws InvalidSettingsException {
        //
    }

    @Override
    protected void loadValidatedSettingsFrom(final NodeSettingsRO settings) throws InvalidSettingsException {
        m_debug = settings.getBoolean(CFG_DEBUG);
        if (settings.containsKey(m_doUpdateTemplateLinks.getConfigName())) {
            m_doUpdateTemplateLinks.loadSettingsFrom(settings);
        } else {
            m_doUpdateTemplateLinks.setBooleanValue(false);
        }
        if (settings.containsKey(m_doExecuteAllNodes.getConfigName())) {
            m_doExecuteAllNodes.loadSettingsFrom(settings);
        } else {
            // backwards compatibility
            m_doExecuteAllNodes.setBooleanValue(false);
        }
    }

    @Override
    protected void onDispose() {
        disposeInternal(true);
    }

    @Override
    protected void reset() {
        disposeInternal(!m_debug);
    }

    private void disposeInternal(final boolean disposeWorkflowExecutable) {
        if (disposeWorkflowExecutable) {
            disposeWorkflowExecutable();
        }
    }

}
