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
 *   Feb 14, 2020 (hornm): created
 */
package org.knime.python.llm.java.capture.end;

import java.awt.Color;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

/**
 * Panel to assign custom input and output ids.
 *
 * @author Martin Horn, KNIME GmbH, Konstanz, Germany
 */
@SuppressWarnings("serial")
class InputOutputIDsPanel extends JPanel {

    private List<JTextField> m_inputIDs;

    private List<JTextField> m_outputIDs;

    InputOutputIDsPanel(final List<String> inputIDs, final List<String> outputIDs) {
        JPanel inputsPanel = new JPanel();
        setBorder(inputsPanel, "Input port IDs");
        m_inputIDs = fillPanel(inputsPanel, inputIDs, "Input port");
        JPanel outputsPanel = new JPanel();
        setBorder(outputsPanel, "Output port IDs");
        m_outputIDs = fillPanel(outputsPanel, outputIDs, "Output port");

        setLayout(new BoxLayout(this, BoxLayout.X_AXIS));
        add(inputsPanel);
        add(outputsPanel);
    }

    private static void setBorder(final JPanel p, final String title) {
        p.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(), title));
    }

    private static List<JTextField> fillPanel(final JPanel p, final List<String> ids, final String label) {
        p.setLayout(new GridBagLayout());
        GridBagConstraints c = new GridBagConstraints();
        c.gridwidth = 1;
        c.weightx = 1;
        c.weighty = 0;
        c.anchor = GridBagConstraints.FIRST_LINE_START;
        List<JTextField> res = new ArrayList<>(ids.size());
        int i = 1;
        for (String id : ids) {
            c.insets = new Insets(7, 0, 0, 0);
            c.gridy++;
            JLabel jLabel = new JLabel(label + " [" + i + "]");
            p.add(jLabel, c);
            c.insets = new Insets(0, 0, 7, 0);
            JTextField text = new JTextField(12);
            text.setText(id);
            c.gridy++;
            p.add(text, c);
            res.add(text);
            i++;
        }
        c.gridy++;
        c.weighty = 1;
        p.add(Box.createHorizontalGlue(), c);
        return res;
    }

    void setInputsOutputsInvalid(final boolean inputsInvalid, final boolean outputsInvalid) {
        for (JTextField t : m_inputIDs) {
            t.setBackground(inputsInvalid ? Color.red : Color.white);
        }
        for (JTextField t : m_outputIDs) {
            t.setBackground(outputsInvalid ? Color.red : Color.white);
        }
    }

    List<String> getInputIDs() {
        return m_inputIDs.stream().map(JTextField::getText).collect(Collectors.toList());
    }

    List<String> getOutputIDs() {
        return m_outputIDs.stream().map(JTextField::getText).collect(Collectors.toList());
    }
}
