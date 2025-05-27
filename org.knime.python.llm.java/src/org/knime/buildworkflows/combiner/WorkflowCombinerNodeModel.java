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
 *   Dec 9, 2019 (Mark Ortmann, KNIME GmbH, Berlin, Germany): created
 */
package org.knime.python.llm.java.combiner;

import static org.knime.core.node.workflow.capture.BuildWorkflowsUtil.loadWorkflow;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.Set;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.knime.core.data.DataTable;
import org.knime.core.node.CanceledExecutionException;
import org.knime.core.node.ExecutionContext;
import org.knime.core.node.ExecutionMonitor;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.KNIMEException;
import org.knime.core.node.KNIMEException.KNIMERuntimeException;
import org.knime.core.node.NodeModel;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.context.ports.PortsConfiguration;
import org.knime.core.node.port.PortObject;
import org.knime.core.node.port.PortObjectSpec;
import org.knime.core.node.workflow.NodeID;
import org.knime.core.node.workflow.NodeID.NodeIDSuffix;
import org.knime.core.node.workflow.NodeUIInformation;
import org.knime.core.node.workflow.WorkflowCopyContent;
import org.knime.core.node.workflow.WorkflowCreationHelper;
import org.knime.core.node.workflow.WorkflowLock;
import org.knime.core.node.workflow.WorkflowManager;
import org.knime.core.node.workflow.capture.WorkflowPortObject;
import org.knime.core.node.workflow.capture.WorkflowPortObjectSpec;
import org.knime.core.node.workflow.capture.WorkflowSegment;
import org.knime.core.node.workflow.capture.WorkflowSegment.Input;
import org.knime.core.node.workflow.capture.WorkflowSegment.Output;
import org.knime.core.node.workflow.capture.WorkflowSegment.PortID;
import org.knime.core.util.Pair;

/**
 * Merges several {@link WorkflowPortObject} objects into a single {@link WorkflowPortObject}.
 *
 * @author Mark Ortmann, KNIME GmbH, Berlin, Germany
 * @author Martin Horn, KNIME GmbH, Konstanz, Germany
 */
final class WorkflowCombinerNodeModel extends NodeModel {

    static final String CFG_CONNECTION_MAPS = "connection_maps";

    private NodeID m_wfmID;

    private ConnectionMaps m_connectionMaps = ConnectionMaps.PAIR_WISE_CONNECTION_MAPS;

    /**
     * Constructor.
     *
     * @param portsConfiguration the ports configuration
     */
    WorkflowCombinerNodeModel(final PortsConfiguration portsConfiguration) {
        super(portsConfiguration.getInputPorts(), portsConfiguration.getOutputPorts());
    }

    @Override
    protected PortObjectSpec[] configure(final PortObjectSpec[] inSpecs) throws InvalidSettingsException {
        for (int i = 1; i < inSpecs.length; i++) {
            canConnect((WorkflowPortObjectSpec)inSpecs[i - 1], (WorkflowPortObjectSpec)inSpecs[i], i - 1);
        }
        return null;
    }

    /**
     * Checks whether two workflow segments can be connected.
     *
     * @param pred the predecessor workflow segment
     * @param succ the successor workflow segment
     * @param connectionIdx the index of the workflow segment connection
     * @throws InvalidSettingsException if the workflow segment couldn't be connected completely
     */
    private void canConnect(final WorkflowPortObjectSpec pred, final WorkflowPortObjectSpec succ,
        final int connectionIdx) throws InvalidSettingsException {
        m_connectionMaps.getConnectionMap(connectionIdx).getConnectionsFor(pred.getOutputs(), succ.getInputs());
    }

    @Override
    protected PortObject[] execute(final PortObject[] input, final ExecutionContext exec) throws Exception {
        final WorkflowPortObjectSpec[] inWorkflowSpecs =
            Arrays.stream(input).map(s -> ((WorkflowPortObject)s).getSpec()).toArray(WorkflowPortObjectSpec[]::new);

        // clear if needed - should not be necessary
        clear();

        // create the metanode storing the combined workflow
        WorkflowManager wfm = createWFM();
        // store the id so we can remove that node in case something goes wrong or we reset/dispose the node
        m_wfmID = wfm.getID();

        //transfer the editor settings from the first segment
        //the loaded workflow will be disposed in the copy-method below
        wfm.setEditorUIInformation(
            loadWorkflow(inWorkflowSpecs[0].getWorkflowSegment(), this::setWarningMessage).getEditorUIInformation());

        try {
            // copy and paste all segments to the new wfm
            final WorkflowSegmentMeta[] inWorkflowSegments = Arrays.stream(inWorkflowSpecs)
                .map(s -> copy(wfm, s, this::setWarningMessage)).toArray(WorkflowSegmentMeta[]::new);

            final Set<NodeIDSuffix> objectReferenceReaderNodes =
                new HashSet<>(inWorkflowSegments[0].m_objectReferenceReaderNodes);

            for (int i = 1; i < inWorkflowSegments.length; i++) {
                connect(wfm, m_connectionMaps.getConnectionMap(i - 1), inWorkflowSegments[i - 1].m_outputs,
                    inWorkflowSegments[i - 1].m_outIdMapping, inWorkflowSegments[i].m_inputs,
                    inWorkflowSegments[i].m_inIdMapping);
                objectReferenceReaderNodes.addAll(inWorkflowSegments[i].m_objectReferenceReaderNodes);
            }

            WorkflowPortObject firstWorkflowPortObject = (WorkflowPortObject)input[0];
            Map<String, DataTable> inputData = inWorkflowSegments[0].m_inputs.entrySet().stream()
                .filter(e -> firstWorkflowPortObject.getInputDataFor(e.getKey()).isPresent()).collect(
                    Collectors.toMap(Entry::getKey, e -> firstWorkflowPortObject.getInputDataFor(e.getKey()).get()));

            String workflowName = inWorkflowSpecs[0].getWorkflowName(); //TODO make configurable
            Pair<List<String>, List<Input>> newInputs = collectAndMapAllRemainingInputPorts(inWorkflowSegments);
            Pair<List<String>, List<Output>> newOutputs = collectAndMapAllRemainingOutputPorts(inWorkflowSegments);
            List<String> duplicates = getDuplicates(newInputs.getFirst());
            if (!duplicates.isEmpty()) {
                throw new IllegalStateException(
                    "Duplicate input IDs: " + duplicates.stream().collect(Collectors.joining(",")));
            }
            duplicates = getDuplicates(newOutputs.getFirst());
            if (!duplicates.isEmpty()) {
                throw new IllegalStateException(
                    "Duplicate output IDs: " + duplicates.stream().collect(Collectors.joining(",")));
            }
            return new PortObject[]{new WorkflowPortObject(
                new WorkflowPortObjectSpec(new WorkflowSegment(wfm, newInputs.getSecond(), newOutputs.getSecond(),
                    objectReferenceReaderNodes), workflowName, newInputs.getFirst(), newOutputs.getFirst()),
                inputData)};
        } catch (final Exception e) {
            // in case something goes wrong ensure that newly created metanode/component is removed
            clear();
            throw (e);
        }
    }

    private static List<String> getDuplicates(final List<String> ids) {
        Set<String> set = new HashSet<>(ids);
        if (ids.size() == set.size()) {
            return Collections.emptyList();
        } else {
            set.clear();
            List<String> res = new ArrayList<>();
            for (String s : ids) {
                if (set.contains(s)) {
                    res.add(s);
                } else {
                    set.add(s);
                }
            }
            return res;
        }
    }

    private static WorkflowManager createWFM() {
        return WorkflowManager.EXTRACTED_WORKFLOW_ROOT.createAndAddProject("workflow_combiner",
            new WorkflowCreationHelper());
    }

    /**
     * Makes sure that the list of ids doesn't contain duplicates and returns 'fixed' list it does.
     *
     * @param ids the list to check
     * @return an empty optional if there are no duplicates, otherwise a list with fixed ids to be unique ('*' appended)
     */
    private static Optional<List<String>> ensureUniqueness(final List<String> ids) {
        Set<String> set = new HashSet<>(ids);
        if (set.size() == ids.size()) {
            return Optional.empty();
        }
        List<String> res = new ArrayList<>();
        set.clear();
        for (int i = 0; i < ids.size(); i++) {
            if (set.contains(ids.get(i))) {
                res.add(ids.get(i) + "*");
            } else {
                res.add(ids.get(i));
            }
            set.add(res.get(i));
        }
        return Optional.of(res);
    }

    private static WorkflowSegmentMeta copy(final WorkflowManager wfm, final WorkflowPortObjectSpec toCopy,
        final Consumer<String> warningConsumer) throws KNIMERuntimeException {
        // copy and paste the workflow segment into the new wfm
        // calculate the mapping between the toCopy node ids and the new node ids
        final HashMap<NodeIDSuffix, NodeIDSuffix> inIdMapping = new HashMap<>();
        final HashMap<NodeIDSuffix, NodeIDSuffix> outIdMapping = new HashMap<>();
        final HashSet<NodeIDSuffix> objectReferenceReaderNodes = new HashSet<>();

        WorkflowSegment wfToCopy = toCopy.getWorkflowSegment();
        WorkflowManager toCopyWFM;
        try {
            toCopyWFM = loadWorkflow(wfToCopy, warningConsumer);
        } catch (KNIMEException e) {
            throw e.toUnchecked();
        }
        try (WorkflowLock lock = wfm.lock()) {
            int[] wfmBoundingBox = NodeUIInformation.getBoundingBoxOf(wfm.getNodeContainers());
            final int yOffset = wfmBoundingBox[1]; // top
            final int xOffset = wfmBoundingBox[2]; // far right

            final NodeID[] ids = toCopyWFM.getNodeContainers().stream().map(c -> c.getID()).toArray(NodeID[]::new);
            WorkflowCopyContent.Builder sourceContent =
                WorkflowCopyContent.builder().setNodeIDs(ids).setIncludeInOutConnections(false);
            sourceContent.setPositionOffset(new int[] {xOffset, yOffset});
            final WorkflowCopyContent pastedContent = wfm.copyFromAndPasteHere(toCopyWFM, sourceContent.build());

            // store the new ids
            final NodeID[] newIds = pastedContent.getNodeIDs();
            for (int i = 0; i < ids.length; i++) {
                final NodeIDSuffix toCopyID = NodeIDSuffix.create(toCopyWFM.getID(), ids[i]);
                for (NodeIDSuffix refNode : wfToCopy.getPortObjectReferenceReaderNodes()) {
                    int[] suffixArray = refNode.getSuffixArray();
                    //compare and replace the very first id only (because reference reader nodes can be summarized
                    //in metanodes)
                    if (suffixArray[0] == toCopyID.getSuffixArray()[0]) {
                        suffixArray[0] = newIds[i].getIndex();
                        objectReferenceReaderNodes.add(new NodeIDSuffix(suffixArray));
                    }
                }

                final List<Input> inputs = wfToCopy.getConnectedInputs();
                Optional<PortID> inPort = inputs.stream().flatMap(in -> in.getConnectedPorts().stream())
                    .filter(p -> p.getNodeIDSuffix().equals(toCopyID)).findAny();
                if (inPort.isPresent()) {
                    inIdMapping.put(toCopyID, NodeIDSuffix.create(wfm.getID(), newIds[i]));
                }
                Optional<PortID> outPort = wfToCopy.getConnectedOutputs().stream().map(Output::getConnectedPort)
                    .flatMap(o -> o.isPresent() ? Stream.of(o.get()) : Stream.empty())
                    .filter(p -> p.getNodeIDSuffix().equals(toCopyID)).findAny();
                if (outPort.isPresent()) {
                    outIdMapping.put(toCopyID, NodeIDSuffix.create(wfm.getID(), newIds[i]));
                }
            }
        } finally {
            wfToCopy.disposeWorkflow();
        }

        // return the new segment metadata
        WorkflowSegmentMeta res = new WorkflowSegmentMeta();
        res.m_inputs = new LinkedHashMap<>(toCopy.getInputs());
        res.m_inIdMapping = inIdMapping;
        res.m_outputs = new LinkedHashMap<>(toCopy.getOutputs());
        res.m_outIdMapping = outIdMapping;
        res.m_objectReferenceReaderNodes = objectReferenceReaderNodes;
        return res;
    }

    /**
     * Adds the configured connections (represented by a {@link ConnectionMap}) to a workflow manager.
     *
     * The output and input that have been connected are removed from the supplied outputs and inputs lists!
     *
     * @param wfm the workflow manager to add the connections to
     * @param connectionMap the chosen connections
     * @param outputs will be modified - connected outports are removed!
     * @param outIdMapping the mapping to the actual (new) node id
     * @param inputs will be modified - connected inputs are removed!
     * @param inIdMapping the mapping to the actual (new) node id
     * @throws InvalidSettingsException
     */
    private static void connect(final WorkflowManager wfm, final ConnectionMap connectionMap,
        final Map<String, Output> outputs, final Map<NodeIDSuffix, NodeIDSuffix> outIdMapping,
        final Map<String, Input> inputs, final Map<NodeIDSuffix, NodeIDSuffix> inIdMapping)
        throws InvalidSettingsException {
        List<Pair<String, String>> connectionsToAdd = connectionMap.getConnectionsFor(outputs, inputs);
        for (Pair<String, String> connection : connectionsToAdd) {
            Output output = outputs.get(connection.getFirst());
            Input input = inputs.get(connection.getSecond());

            final Optional<PortID> outPort = output.getConnectedPort();
            final Set<PortID> inPorts = input.getConnectedPorts();

            if (outPort.isPresent() && !inPorts.isEmpty()) {
                for (PortID inPort : inPorts) {
                    NodeIDSuffix outNodeId = outIdMapping.get(outPort.get().getNodeIDSuffix());
                    NodeIDSuffix inNodeId = inIdMapping.get(inPort.getNodeIDSuffix());
                    wfm.addConnection(outNodeId.prependParent(wfm.getID()), outPort.get().getIndex(),
                        inNodeId.prependParent(wfm.getID()), inPort.getIndex());
                }
            }
        }
        connectionsToAdd.forEach(c -> {
            outputs.remove(c.getFirst());
            inputs.remove(c.getSecond());
        });

    }

    private static Pair<List<String>, List<Input>>
        collectAndMapAllRemainingInputPorts(final WorkflowSegmentMeta[] segments) {
        List<Input> inputs = new ArrayList<>();
        List<String> inputIDs = new ArrayList<>();
        for (WorkflowSegmentMeta f : segments) {
            for (Entry<String, Input> input : f.m_inputs.entrySet()) {
                inputIDs.add(input.getKey());
                inputs.add(getMappedInput(input.getValue(), f.m_inIdMapping));
            }
        }
        return Pair.create(inputIDs, inputs);
    }

    private static Pair<List<String>, List<Output>>
        collectAndMapAllRemainingOutputPorts(final WorkflowSegmentMeta[] segments) {
        List<Output> outputs = new ArrayList<>();
        List<String> outputIDs = new ArrayList<>();
        for (int i = segments.length - 1; i >= 0; i--) {
            for (Entry<String, Output> output : segments[i].m_outputs.entrySet()) {
                outputIDs.add(output.getKey());
                outputs.add(getMappedOutput(output.getValue(), segments[i].m_outIdMapping));
            }
        }
        return Pair.create(outputIDs, outputs);
    }

    private static Input getMappedInput(final Input input, final Map<NodeIDSuffix, NodeIDSuffix> inIdMapping) {
        Set<PortID> connectedPorts =
            input.getConnectedPorts().stream().map(p -> getMappedPortID(p, inIdMapping)).collect(Collectors.toSet());
        return new Input(input.getType().orElse(null), input.getSpec().orElse(null), connectedPorts);
    }

    private static Output getMappedOutput(final Output output, final Map<NodeIDSuffix, NodeIDSuffix> outIdMapping) {
        PortID connectedPort = output.getConnectedPort().map(p -> getMappedPortID(p, outIdMapping)).orElse(null);
        return new Output(output.getType().orElse(null), output.getSpec().orElse(null), connectedPort);
    }

    private static PortID getMappedPortID(final PortID p, final Map<NodeIDSuffix, NodeIDSuffix> nodeIdMapping) {
        return new PortID(nodeIdMapping.get(p.getNodeIDSuffix()), p.getIndex());
    }

    @Override
    protected void loadInternals(final File nodeInternDir, final ExecutionMonitor exec)
        throws IOException, CanceledExecutionException {
        //
    }

    @Override
    protected void saveInternals(final File nodeInternDir, final ExecutionMonitor exec)
        throws IOException, CanceledExecutionException {
        //
    }

    @Override
    protected void saveSettingsTo(final NodeSettingsWO settings) {
        NodeSettingsWO cmSettings = settings.addNodeSettings(CFG_CONNECTION_MAPS);
        m_connectionMaps.save(cmSettings);
    }

    @Override
    protected void validateSettings(final NodeSettingsRO settings) throws InvalidSettingsException {
        if (settings.containsKey(CFG_CONNECTION_MAPS)) {
            NodeSettingsRO cmSettings = settings.getNodeSettings(CFG_CONNECTION_MAPS);
            new ConnectionMaps().load(cmSettings);
        }
    }

    @Override
    protected void loadValidatedSettingsFrom(final NodeSettingsRO settings) throws InvalidSettingsException {
        m_connectionMaps = new ConnectionMaps();
        if (settings.containsKey(CFG_CONNECTION_MAPS)) {
            NodeSettingsRO cmSettings = settings.getNodeSettings(CFG_CONNECTION_MAPS);
            m_connectionMaps.load(cmSettings);
        }
    }

    @Override
    protected void reset() {
        clear();
    }

    @Override
    protected void onDispose() {
        clear();
        super.onDispose();
    }

    private void clear() {
        if (m_wfmID != null) {
            WorkflowManager.EXTRACTED_WORKFLOW_ROOT.removeNode(m_wfmID);
        }
        m_wfmID = null;
    }

    /**
     * Internal helper class to be able to summarize and return the metadata of a workflow segment.
     */
    private static class WorkflowSegmentMeta {

        Map<String, Input> m_inputs;

        Map<NodeIDSuffix, NodeIDSuffix> m_inIdMapping;

        Map<String, Output> m_outputs;

        Map<NodeIDSuffix, NodeIDSuffix> m_outIdMapping;

        Set<NodeIDSuffix> m_objectReferenceReaderNodes;

    }

}
