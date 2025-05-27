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
 *   Feb 6, 2020 (hornm): created
 */
package org.knime.python.llm.java.writer;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.event.ItemEvent;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import javax.swing.BorderFactory;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.border.Border;

import org.apache.commons.lang3.ArrayUtils;
import org.knime.core.util.Pair;

/**
 * Panel that let select (input or output) nodes for ports.
 *
 * @author Martin Horn, KNIME GmbH, Konstanz, Germany
 */
@SuppressWarnings("serial")
class PortAndNodeConfigPanel<C extends IONodeConfig> extends JPanel {

    private JComboBox<String> m_portSelection;

    private JComboBox<String> m_nodeSelection;

    private Map<String, C> m_selectedConfigs;

    private Map<Pair<String, String>, C> m_configCache;

    private String m_warningMessage;

    private String m_noneChoice;

    private Function<String, C> m_instantiateConfig;

    @SuppressWarnings("java:S1699") // calls to derived methods in constructor (sonar)
    PortAndNodeConfigPanel(final String title, final Function<String, C> instantiateConfig, final String noneChoice,
        final String... nodeNames) {
        setLayout(new BorderLayout());
        setBorder(createBorder(title));

        var comboBoxes = new JPanel(new BorderLayout());
        m_portSelection = new JComboBox<>();
        m_portSelection.addItemListener(e -> {
            if (e.getStateChange() == ItemEvent.SELECTED) {
                portSelectionChanged();
            }
        });
        m_nodeSelection = new JComboBox<>(ArrayUtils.insert(0, nodeNames, noneChoice));
        m_nodeSelection.addItemListener(e -> {
            if (e.getStateChange() == ItemEvent.SELECTED) {
                nodeSelectionChanged();
            }
        });
        comboBoxes.add(m_portSelection, BorderLayout.WEST);
        comboBoxes.add(m_nodeSelection, BorderLayout.CENTER);
        add(comboBoxes, BorderLayout.NORTH);
        m_selectedConfigs = new HashMap<>();
        m_configCache = new HashMap<>();

        m_noneChoice = noneChoice;
        m_instantiateConfig = instantiateConfig;
    }

    Map<String, C> getSelectedConfigs() {
        return m_selectedConfigs;
    }

    void updatePanel(final List<String> ports, final Function<String, C> getConfig) {
        m_selectedConfigs.clear();
        m_portSelection.removeAllItems();
        if (ports.isEmpty()) {
            removeAll();
            add(new JLabel("No ports"));
            revalidate();
            repaint();
        } else {
            ports.forEach(m_portSelection::addItem);
            m_portSelection.setEnabled(ports.size() > 1);
            for (String port : ports) {
                C config = getConfig.apply(port);
                m_selectedConfigs.put(port, config);
                if (config != null) {
                    m_configCache.put(Pair.create(port, config.getNodeName()), config);
                }
            }
            String firstPort = ports.get(0);
            C firstPortConfig = getConfig.apply(firstPort);
            if (firstPortConfig != null) {
                m_nodeSelection.setSelectedItem(firstPortConfig.getNodeName());
                m_portSelection.setSelectedItem(firstPort);
            } else {
                m_nodeSelection.setSelectedIndex(0);
                m_portSelection.setSelectedIndex(0);
            }
        }
    }

    private void nodeSelectionChanged() {
        //remove configure panel and warning
        //but keep comboboxes
        Component tmp = getComponent(0);
        removeAll();
        add(tmp, BorderLayout.NORTH);

        String nodeName = (String)m_nodeSelection.getSelectedItem();
        String port = (String)m_portSelection.getSelectedItem();
        if (!nodeName.equals(m_noneChoice)) {
            Pair<String, String> key = Pair.create(port, nodeName);
            C nodeConfig = m_configCache.computeIfAbsent(key, k -> {
                return m_instantiateConfig.apply(nodeName);
            });
            m_selectedConfigs.put(port, nodeConfig);

            JPanel configPanel = nodeConfig.getOrCreateJPanel();
            JPanel border = new JPanel(new BorderLayout());
            border.add(configPanel, BorderLayout.WEST);
            border.setBorder(createBorder("Configuration"));
            add(border, BorderLayout.CENTER);
        } else {
            m_selectedConfigs.put(port, null);
        }
        if (m_warningMessage != null) {
            JLabel msg = new JLabel(m_warningMessage);
            msg.setForeground(Color.RED);
            add(msg, BorderLayout.SOUTH);
        }

        revalidate();
        repaint();
    }

    private void portSelectionChanged() {
        String port = (String)m_portSelection.getSelectedItem();
        C nodeConfig = m_selectedConfigs.get(port);
        if (nodeConfig == null) {
            m_nodeSelection.setSelectedItem(m_noneChoice);
        } else {
            m_nodeSelection.setSelectedItem(nodeConfig.getNodeName());
        }
    }

    private static Border createBorder(final String title) {
        return BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(), title);
    }
}
