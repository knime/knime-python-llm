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

import java.awt.BorderLayout;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.stream.Collectors;

import javax.swing.BorderFactory;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;

import org.knime.core.node.workflow.capture.WorkflowSegment.Input;
import org.knime.core.node.workflow.capture.WorkflowSegment.Output;
import org.knime.core.util.Pair;

/**
 * UI to choose/'draw' connections between the output and input of a workflow segment.
 *
 * @author Martin Horn, KNIME GmbH, Konstanz, Germany
 */
@SuppressWarnings("serial")
final class WorkflowConnectPanel extends JPanel {

    private static final String NONE_SELECTION = "<NONE>";

    private final List<JComboBox<String>> m_selectedOutputs;

    private final List<String> m_inputsWF2;


    /**
     * Creates a new connection panel.
     *
     * @param outputsWF1 the outputs of the first workflow segment (to be connected to the inports of the second workflow)
     * @param inputsWF2 the inputs of the second workflow segment
     * @param connectionMap the original connection map to initialize the selected connections
     * @param title a custom title for the panel (titled border)
     * @param outPortNames optional custom names for the output ports
     * @param inPortNames optional custom names for the input ports
     */
    WorkflowConnectPanel(final Map<String, Output> outputsWF1, final Map<String, Input> inputsWF2,
        final ConnectionMap connectionMap, final String title, final String leftWf, final String rightWf) {
        m_inputsWF2 = inputsWF2.entrySet().stream().map(Entry::getKey)
            .collect(Collectors.toList());
        setBorder(BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(), title));
        setLayout(new BorderLayout());
        JPanel selection = new JPanel(new GridBagLayout());

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.anchor = GridBagConstraints.WEST;
        gbc.insets = new Insets(0, 0, 10, 20);

        gbc.gridx = 1;
        gbc.gridy = 0;
        selection.add(new JLabel(leftWf), gbc);
        gbc.gridx = 2;
        selection.add(new JLabel("\u2192"), gbc);

        gbc.gridx = 3;
        selection.add(new JLabel(rightWf), gbc);

        m_selectedOutputs = new ArrayList<>(inputsWF2.size());
        Iterator<Entry<String, Input>> inputsIt = inputsWF2.entrySet().iterator();
        while(inputsIt.hasNext()) {
            Entry<String, Input> input = inputsIt.next();
            List<String> compatibleOutputs = outputsWF1.entrySet().stream()
                .filter(o -> o.getValue().getType().equals(input.getValue().getType())).map(Entry::getKey)
                .collect(Collectors.toList());
            gbc.gridx = 0;
            gbc.gridy++;
            selection.add(new JLabel("Connect"), gbc);
            JComboBox<String> cbox = new JComboBox<>();
            cbox.addItem(NONE_SELECTION);
            compatibleOutputs.forEach(cbox::addItem);
            //set selected item
            Optional<String> selectedOutput = connectionMap.getOutPortForInPort(input.getKey(), outputsWF1, inputsWF2);
            if (selectedOutput.isPresent()) {
                cbox.setSelectedItem(selectedOutput.get());
            } else {
                cbox.setSelectedItem(NONE_SELECTION);
            }
            m_selectedOutputs.add(cbox);
            JPanel panel = new JPanel(new GridBagLayout());
            GridBagConstraints innerGbc = new GridBagConstraints();
            innerGbc.anchor = GridBagConstraints.WEST;
            innerGbc.gridx = 0;
            panel.add(new JLabel("Output "), innerGbc);
            innerGbc.gridx = 1;
            panel.add(cbox, innerGbc);
            gbc.gridx = 1;
            selection.add(panel, gbc);
            gbc.gridx = 2;
            selection.add(new JLabel("to"), gbc);
            gbc.gridx = 3;
            selection.add(new JLabel("Input \"" + input.getKey() + "\""), gbc);
        }

        add(selection, BorderLayout.WEST);

        if (connectionMap == ConnectionMaps.SIMPLE_PAIR_WISE_CONNECTED_MAP) {
            add(new JLabel("Note: outputs automatically selected per default configuration"), BorderLayout.SOUTH);
        }
    }

    /**
     * @return creates a {@link ConnectionMap} from the user choice
     */
    ConnectionMap createConnectionMap() {
        List<Pair<String, String>> l = new ArrayList<>();
        for (int i = 0; i < m_inputsWF2.size(); i++) {
            if (m_selectedOutputs.get(i).getSelectedItem() != NONE_SELECTION) {
                l.add(Pair.create((String)m_selectedOutputs.get(i).getSelectedItem(), m_inputsWF2.get(i)));
            }
        }
        return new ConnectionMap(l);
    }
}
