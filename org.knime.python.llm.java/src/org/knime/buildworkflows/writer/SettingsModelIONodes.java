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

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Optional;

import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeLogger;
import org.knime.core.node.NodeSettings;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.NotConfigurableException;
import org.knime.core.node.defaultnodesettings.SettingsModel;
import org.knime.core.node.port.PortObjectSpec;
import org.knime.core.node.workflow.capture.WorkflowPortObjectSpec;

/**
 * Settings model for {@link DialogComponentIONodes}.
 *
 * @author Martin Horn, KNIME GmbH, Konstanz, Germany
 */
public class SettingsModelIONodes extends SettingsModel {

    private static final String CFG_KEY_INPUT_NODE = "input_node_";

    private static final String CFG_KEY_OUTPUT_NODE = "output_node_";

    private static final String CFG_KEY_NODE_CONFIG_CLASS = "node_config_class";

    private static final String CFG_KEY_INPUT_ID = "input_id";

    private static final String CFG_KEY_OUTPUT_ID = "output_id";

    private static final String CFG_KEY_NODE_CONFIG = "node_config";

    private static final String CFG_NUM_INPUTS = "num_inputs";

    private static final String CFG_NUM_OUTPUTS = "num_outputs";

    private final String m_configName;

    private int m_workflowInputPortIndex = -1;

    /* null indicates the default config */
    private Map<String, InputNodeConfig> m_inputNodeConfigs = null;

    /* null indicates the default config */
    private Map<String, OutputNodeConfig> m_outputNodeConfigs = null;

    public SettingsModelIONodes(final String configName) {
        m_configName = configName;
    }

    void initWithDefaults(final List<String> inputs, final List<String> outputs) {
        if (m_inputNodeConfigs == null) {
            m_inputNodeConfigs = new HashMap<>();
            for (String input : inputs) {
                m_inputNodeConfigs.put(input, new WorkflowInputNodeConfig());
            }
        }
        if (m_outputNodeConfigs == null) {
            m_outputNodeConfigs = new HashMap<>();
            for (String output : outputs) {
                m_outputNodeConfigs.put(output, new WorkflowOutputNodeConfig());
            }
        }
    }

    Optional<InputNodeConfig> getInputNodeConfig(final String p) {
        return Optional.ofNullable(m_inputNodeConfigs.get(p));
    }

    List<String> getConfiguredInputs(final List<String> availableInputs) {
        return getIntersection(availableInputs, m_inputNodeConfigs.keySet());
    }

    void setInputNodeConfig(final String p, final InputNodeConfig config) {
        m_inputNodeConfigs.put(p, config);
    }

    Optional<OutputNodeConfig> getOutputNodeConfig(final String p) {
        return Optional.ofNullable(m_outputNodeConfigs.get(p));
    }

    List<String> getConfiguredOutputs(final List<String> availableOutputs) {
        return getIntersection(availableOutputs, m_outputNodeConfigs.keySet());
    }

    void setOutputNodeConfig(final String p, final OutputNodeConfig config) {
        m_outputNodeConfigs.put(p, config);
    }

    void setWorkflowInputPortIndex(final int idx) {
        m_workflowInputPortIndex = idx;
    }

    private static List<String> getIntersection(final Collection<String> c1, final Collection<String> c2) {
        List<String> intersect = new ArrayList<>();
        for (String p : c1) {
            if (c2.contains(p)) {
                intersect.add(p);
            }
        }
        return intersect;
    }

    /**
     * {@inheritDoc}
     */
    @SuppressWarnings("unchecked")
    @Override
    protected SettingsModelIONodes createClone() {
        NodeSettings settings = new NodeSettings("clone");
        saveSettingsForModel(settings);
        SettingsModelIONodes clone = new SettingsModelIONodes(m_configName);
        try {
            clone.loadSettingsForModel(settings);
        } catch (InvalidSettingsException e) {
            //should never happen
            throw new RuntimeException(e);
        }
        return clone;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected String getModelTypeID() {
        return "SMID_IOnodes";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected String getConfigName() {
        return m_configName;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void loadSettingsForDialog(final NodeSettingsRO settings, final PortObjectSpec[] specs)
        throws NotConfigurableException {
        assert m_workflowInputPortIndex >= 0; // must be set by associated the component
        if (m_workflowInputPortIndex >= specs.length) {
            String msg = "Specified port index is out of bounds";
            NodeLogger.getLogger(SettingsModelIONodes.class).coding(msg);
            throw new NotConfigurableException(msg);
        }
        if (specs[m_workflowInputPortIndex] == null) {
            String msg = "No workflow given.";
            NodeLogger.getLogger(SettingsModelIONodes.class).coding(msg);
            throw new NotConfigurableException(msg);
        }
        if (!(specs[m_workflowInputPortIndex] instanceof WorkflowPortObjectSpec)) {
            String msg = "Not a workflow port at specified index";
            NodeLogger.getLogger(SettingsModelIONodes.class).coding(msg);
            throw new NotConfigurableException(msg);
        }

        WorkflowPortObjectSpec spec = (WorkflowPortObjectSpec)specs[m_workflowInputPortIndex];
        try {
            loadSettingsInternal(settings, spec.getInputIDs(), spec.getOutputIDs());
        } catch (InvalidSettingsException e) {
            //should never happen
            throw new RuntimeException(e);
        }
    }

    private static IONodeConfig createConfigInstanceForName(final String nodeConfigClass) {
        try {
            return (IONodeConfig)Class.forName(nodeConfigClass).newInstance();
        } catch (InstantiationException | IllegalAccessException | ClassNotFoundException e) {
            throw new IllegalStateException(
                "Node config instance couldn't be created. Most likely and implementation error.", e);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void saveSettingsForDialog(final NodeSettingsWO settings) throws InvalidSettingsException {
        saveSettingsForModel(settings);
    }

    public void validateSettings() throws InvalidSettingsException {
        if (m_inputNodeConfigs != null) {
            for (InputNodeConfig config : m_inputNodeConfigs.values()) {
                config.validateSettings();
            }
        }
        if (m_outputNodeConfigs != null) {
            for (OutputNodeConfig config : m_outputNodeConfigs.values()) {
                config.validateSettings();
            }
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void validateSettingsForModel(final NodeSettingsRO settings) throws InvalidSettingsException {
        loadSettingsInternal(settings, null, null);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void loadSettingsForModel(final NodeSettingsRO settings) throws InvalidSettingsException {
        loadSettingsInternal(settings, null, null);
    }

    private void loadSettingsInternal(final NodeSettingsRO settings, final List<String> inputs,
        final List<String> outputs) throws InvalidSettingsException {
        boolean changed = false;
        if (settings.containsKey(m_configName)) {
            NodeSettingsRO subsettings = settings.getNodeSettings(m_configName);
            //load actual input configs
            if (subsettings.containsKey(CFG_NUM_INPUTS)) {
                Map<String, InputNodeConfig> inputNodeConfigs = new HashMap<>();
                for (int i = 0; i < subsettings.getInt(CFG_NUM_INPUTS); i++) {
                    NodeSettingsRO node = subsettings.getNodeSettings(CFG_KEY_INPUT_NODE + i);
                    InputNodeConfig config =
                        (InputNodeConfig)createConfigInstanceForName(node.getString(CFG_KEY_NODE_CONFIG_CLASS));
                    config.loadSettingsFrom(node.getNodeSettings(CFG_KEY_NODE_CONFIG));
                    String inputID = node.getString(CFG_KEY_INPUT_ID);
                    if (inputs != null) {
                        if (inputs.contains(inputID)) {
                            inputNodeConfigs.put(inputID, config);
                        }
                    } else {
                        inputNodeConfigs.put(inputID, config);
                    }
                }
                if (!inputNodeConfigs.equals(m_inputNodeConfigs)) {
                    m_inputNodeConfigs = inputNodeConfigs;
                    changed = true;
                }
            } else {
                if (m_inputNodeConfigs != null) {
                    changed = true;
                }
                // set to null such that m_inputNodeConfigs will be initialized with defaults in initWithDefaults
                m_inputNodeConfigs = null;
            }

            //load actual output configs
            if (subsettings.containsKey(CFG_NUM_OUTPUTS)) {
                Map<String, OutputNodeConfig> outputNodeConfigs = new HashMap<>();
                for (int i = 0; i < subsettings.getInt(CFG_NUM_OUTPUTS); i++) {
                    NodeSettingsRO node = subsettings.getNodeSettings(CFG_KEY_OUTPUT_NODE + i);
                    OutputNodeConfig config =
                        (OutputNodeConfig)createConfigInstanceForName(node.getString(CFG_KEY_NODE_CONFIG_CLASS));
                    config.loadSettingsFrom(node.getNodeSettings(CFG_KEY_NODE_CONFIG));
                    String outputID = node.getString(CFG_KEY_OUTPUT_ID);
                    if (outputs != null) {
                        if (outputs.contains(outputID)) {
                            outputNodeConfigs.put(outputID, config);
                        }
                    } else {
                        outputNodeConfigs.put(outputID, config);
                    }
                }
                if (!outputNodeConfigs.equals(m_outputNodeConfigs)) {
                    m_outputNodeConfigs = outputNodeConfigs;
                    changed = true;
                }
            } else {
                if (m_outputNodeConfigs != null) {
                    changed = true;
                }
                // set to null such that m_outputNodeConfigs will be initialized with defaults in initWithDefaults
                m_outputNodeConfigs = null;
            }
        }
        if (changed) {
            notifyChangeListeners();
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void saveSettingsForModel(final NodeSettingsWO settings) {
        NodeSettingsWO subsettings = settings.addNodeSettings(m_configName);

        if (m_inputNodeConfigs != null) {
            int numInPorts = (int)m_inputNodeConfigs.values().stream().filter(Objects::nonNull).count();
            subsettings.addInt(CFG_NUM_INPUTS, numInPorts);
            int i = 0;
            for (Entry<String, InputNodeConfig> inConfigs : m_inputNodeConfigs.entrySet()) {
                if (inConfigs.getValue() != null) {
                    NodeSettingsWO node = subsettings.addNodeSettings(CFG_KEY_INPUT_NODE + i);
                    node.addString(CFG_KEY_NODE_CONFIG_CLASS, inConfigs.getValue().getClass().getCanonicalName());
                    node.addString(CFG_KEY_INPUT_ID, inConfigs.getKey());
                    NodeSettingsWO config = node.addNodeSettings(CFG_KEY_NODE_CONFIG);
                    inConfigs.getValue().saveSettingsTo(config);
                    i++;
                }
            }
        }

        if (m_outputNodeConfigs != null) {
            int numOutPorts = (int)m_outputNodeConfigs.values().stream().filter(Objects::nonNull).count();
            subsettings.addInt(CFG_NUM_OUTPUTS, numOutPorts);
            int i = 0;
            for (Entry<String, OutputNodeConfig> outConfigs : m_outputNodeConfigs.entrySet()) {
                if (outConfigs.getValue() != null) {
                    NodeSettingsWO node = subsettings.addNodeSettings(CFG_KEY_OUTPUT_NODE + i);
                    node.addString(CFG_KEY_NODE_CONFIG_CLASS, outConfigs.getValue().getClass().getCanonicalName());
                    node.addString(CFG_KEY_OUTPUT_ID, outConfigs.getKey());
                    NodeSettingsWO config = node.addNodeSettings(CFG_KEY_NODE_CONFIG);
                    outConfigs.getValue().saveSettingsTo(config);
                    i++;
                }
            }
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        return getClass().getSimpleName() + " ('" + m_configName + "')";
    }

}
