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

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.stream.Collectors;

import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.util.CheckUtils;
import org.knime.core.node.workflow.capture.WorkflowSegment.Input;
import org.knime.core.node.workflow.capture.WorkflowSegment.Output;
import org.knime.core.util.Pair;

/**
 * Represents the connections between two workflow segments (i.e. connections between the outputs of the first and
 * inputs of the second workflow segments). The inputs and outputs are referenced by their id.
 *
 * @author Martin Horn, KNIME GmbH, Konstanz, Germany
 */
class ConnectionMap {

    private static final String CFG_NUM_CONNECTIONS = "num_connections";

    private static final String CFG_INPUT_ID = "input_id_";

    private static final String CFG_OUTPUT_ID = "output_id_";

    private List<Pair<String, String>> m_connections = null;

    /**
     * A new connection map.
     *
     * @param connections connections represented as a list of pairs of ids
     */
    ConnectionMap(final List<Pair<String, String>> connections) {
        m_connections = connections;
    }

    /**
     * For deserialization or a simple pair-wise connected map.
     */
    ConnectionMap() {
        //
    }

    /**
     * Gets all configured connections for the given outputs and inputs, based on their respective id.
     *
     * @param out the available outputs to get the connections for (deterministic iteration in the order ot the ports
     *            expected!!)
     * @param in the available inputs to get the connections for (deterministic iteration in the order of the ports
     *            expected!!)
     * @return the connections represented as a list of pairs of output id connected to an input id
     * @throws InvalidSettingsException if the given outputs and inputs couldn't be connected appropriately
     */
    List<Pair<String, String>> getConnectionsFor(final Map<String, Output> out, final Map<String, Input> in)
        throws InvalidSettingsException {
        List<Pair<String, String>> res = new ArrayList<>();
        if (m_connections == null) {
            CheckUtils.checkSetting(out.size() == in.size(),
                "Can't pair-wise connect outputs to inputs: The number of output and input ports differ");
            Iterator<Entry<String, Output>> outIt = out.entrySet().iterator();
            Iterator<Entry<String, Input>> inIt = in.entrySet().iterator();
            while (outIt.hasNext() && inIt.hasNext()) {
                Entry<String, Output> outEntry = outIt.next();
                Entry<String, Input> inEntry = inIt.next();
                CheckUtils.checkSetting(outEntry.getValue().getType().equals(inEntry.getValue().getType()),
                    "Can't pair-wise connect outputs to inputs: types of some ports differ.");
                res.add(Pair.create(outEntry.getKey(), inEntry.getKey()));
            }
        } else {
            res = m_connections.stream().filter(c -> out.containsKey(c.getFirst()) && in.containsKey(c.getSecond()))
                .collect(Collectors.toList());
        }
        return res;
    }

    /**
     * Gets the output for a given input as configured by this connection map if<br/>
     * 1. there is connection to the given input id<br/>
     * 2. the output-id of the connection is contained in the list of all available outputs<br/>
     * 3. the port types of the output and input are compatible<br/>
     *
     * Otherwise an empty optional is returned.
     *
     * @param in the input to get the output for - must be contained in the 'ins'-list
     * @param outs map of all available outputs (in deterministic order of the ports!)
     * @param ins the list of all available inputs (in deterministic order of the ports!)
     * @return the output or an empty optional if there is no mapping
     */
    Optional<String> getOutPortForInPort(final String in, final Map<String, Output> outs,
        final Map<String, Input> ins) {
        assert ins.keySet().contains(in);
        if (m_connections == null) {
            //default pair-wise connection
            Iterator<Entry<String, Output>> outIt = outs.entrySet().iterator();
            Iterator<Entry<String, Input>> inIt = ins.entrySet().iterator();
            while (inIt.hasNext() && outIt.hasNext()) {
                Entry<String, Output> output = outIt.next();
                Entry<String, Input> input = inIt.next();
                if (input.getKey().equals(in)) {
                    //check port type compatibility
                    if (output.getValue().getType().equals(input.getValue().getType())) {
                        return Optional.of(output.getKey());
                    }
                }
            }
            return Optional.empty();
        } else {
            return m_connections.stream().filter(c -> c.getSecond().equals(in)).filter(p -> {
                //make sure that
                //1. the give input-id is part of a connection in this connection map
                //2. there is a connection where the output-id is contained in the list of available outputs
                //3. those port types are compatible
                return in.equals(p.getSecond()) && outs.keySet().contains(p.getFirst())
                    && outs.get(p.getFirst()).getType().equals(ins.get(p.getSecond()).getType());
            }).map(Pair::getFirst).findFirst();
        }
    }

    /**
     * Saves this connection map to a settings object.
     *
     * @param settings
     */
    void save(final NodeSettingsWO settings) {
        if (m_connections == null) {
            settings.addInt(CFG_NUM_CONNECTIONS, -1);
            return;
        }
        settings.addInt(CFG_NUM_CONNECTIONS, m_connections.size());
        for (int i = 0; i < m_connections.size(); i++) {
            settings.addString(CFG_INPUT_ID + i, m_connections.get(i).getSecond());
            settings.addString(CFG_OUTPUT_ID + i, m_connections.get(i).getFirst());
        }
    }

    /**
     * Loads this connection map from a settings object.
     *
     * @param settings
     * @throws InvalidSettingsException
     */
    void load(final NodeSettingsRO settings) throws InvalidSettingsException {
        int num = settings.getInt(CFG_NUM_CONNECTIONS);
        if (num == -1) {
            return;
        }
        m_connections = new ArrayList<>(num);
        for (int i = 0; i < num; i++) {
            String inputID = settings.getString(CFG_INPUT_ID + i);
            String outputID = settings.getString(CFG_OUTPUT_ID + i);
            m_connections.add(Pair.create(outputID, inputID));
        }
    }
}
