# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
#  Copyright by KNIME AG, Zurich, Switzerland
#  Website: http://www.knime.com; Email: contact@knime.com
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License, Version 3, as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, see <http://www.gnu.org/licenses>.
#
#  Additional permission under GNU GPL version 3 section 7:
#
#  KNIME interoperates with ECLIPSE solely via ECLIPSE's plug-in APIs.
#  Hence, KNIME and ECLIPSE are both independent programs and are not
#  derived from each other. Should, however, the interpretation of the
#  GNU GPL Version 3 ("License") under any applicable laws result in
#  KNIME and ECLIPSE being a combined program, KNIME AG herewith grants
#  you the additional permission to use and propagate KNIME together with
#  ECLIPSE with only the license terms in place for ECLIPSE applying to
#  ECLIPSE and the GNU GPL Version 3 applying for KNIME, provided the
#  license terms of ECLIPSE themselves allow for the respective use and
#  propagation of ECLIPSE together with KNIME.
#
#  Additional permission relating to nodes for KNIME that extend the Node
#  Extension (and in particular that are based on subclasses of NodeModel,
#  NodeDialog, and NodeView) and that only interoperate with KNIME through
#  standard APIs ("Nodes"):
#  Nodes are deemed to be separate and independent programs and to not be
#  covered works.  Notwithstanding anything to the contrary in the
#  License, the License does not apply to Nodes, you are not required to
#  license Nodes under the License, and you are granted a license to
#  prepare and propagate Nodes, in each case even if such Nodes are
#  propagated with or for interoperation with KNIME.  The owner of a Node
#  may freely choose the license terms applicable to such Node, including
#  when such Node is propagated with or for interoperation with KNIME.
# ------------------------------------------------------------------------

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class ModelInfo(BaseModel):
	"""Representation of a single model returned by the AI gateway.

	Older gateway versions did not include an explicit `id` field and used
	`name` as the identifier. For backward compatibility the `id` field is
	optional in the payload and is transparently defaulted to `name` when
	absent or null.

	Unknown fields are ignored to remain forward compatible.
	"""

	id: Optional[str] = Field(None, description="Stable unique identifier of the model (falls back to name if missing)")
	name: str = Field(..., description="Display name of the model")
	mode: Optional[str] = Field(None, description="Operational mode (e.g. chat, embeddings)")
	description: Optional[str] = Field(None, description="Human readable description")

	@model_validator(mode="after")
	def _default_id(cls, values):  # type: ignore[override]
		# Ensure id always populated for downstream consumers
		if not values.id:
			values.id = values.name
		return values

	class Config:
		extra = "ignore"  # be robust against newly added fields


class ModelsResponse(BaseModel):
	"""Top-level response envelope from /models endpoint."""

	models: List[ModelInfo]

	class Config:
		extra = "ignore"


