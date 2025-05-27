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
 *   Feb 13, 2020 (hornm): created
 */
package org.knime.python.llm.java.combiner;

import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;

/**
 * Represents a list of many {@link ConnectionMap}s.
 *
 * @author Martin Horn, KNIME GmbH, Konstanz, Germany
 */
class ConnectionMaps {

    /**
     * If all connection maps represent a simple pair-wise connection of ports.
     */
    public static final ConnectionMaps PAIR_WISE_CONNECTION_MAPS = new ConnectionMaps();

    private static final String CFG_NUM_CONNECTION_MAPS = "num_connection_maps";

    private static final String CFG_CONNECTION_MAP = "connection_map_";

    static final ConnectionMap SIMPLE_PAIR_WISE_CONNECTED_MAP = new ConnectionMap();

    private ConnectionMap[] m_maps;

    /**
     * Creates a new list of connection maps.
     *
     * @param maps
     */
    ConnectionMaps(final ConnectionMap... maps) {
        m_maps = maps;
    }

    /**
     * For deserialization.
     */
    ConnectionMaps() {
        //
    }

    /**
     * Gets the connection map at a certain index in the chain of connected workflows. If there is no manually
     * configured connection map at a given index, the default connection map will be returned which is the simple
     * pair-wise connection map (i.e. the output port is connected to the input port at the same index - if compatible
     * and exists).
     *
     * @param idx
     * @return the connection map, never <code>null</code>
     */
    ConnectionMap getConnectionMap(final int idx) {
        if (m_maps != null && idx < m_maps.length) {
            return m_maps[idx];
        } else {
            return SIMPLE_PAIR_WISE_CONNECTED_MAP;
        }
    }

    /**
     * Saves the list of connection maps to a settings object.
     *
     * @param settings
     */
    void save(final NodeSettingsWO settings) {
        if (m_maps != null) {
            settings.addInt(CFG_NUM_CONNECTION_MAPS, m_maps.length);
            for (int i = 0; i < m_maps.length; i++) {
                NodeSettingsWO cmSettings = settings.addNodeSettings(CFG_CONNECTION_MAP + i);
                m_maps[i].save(cmSettings);
            }
        }
    }

    /**
     * Loads the list of connection maps from a settings object.
     *
     * @param settings
     * @throws InvalidSettingsException if the loading failed
     */
    void load(final NodeSettingsRO settings) throws InvalidSettingsException {
        if (settings.containsKey(CFG_NUM_CONNECTION_MAPS)) {
            int numWorkflowsToConnect = settings.getInt(CFG_NUM_CONNECTION_MAPS);
            m_maps = new ConnectionMap[numWorkflowsToConnect];
            for (int i = 0; i < numWorkflowsToConnect; i++) {
                NodeSettingsRO cmSettings = settings.getNodeSettings(CFG_CONNECTION_MAP + i);
                ConnectionMap cm = new ConnectionMap();
                cm.load(cmSettings);
                m_maps[i] = cm;
            }
        }
    }
}
