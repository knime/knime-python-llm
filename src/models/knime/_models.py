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

from pydantic import BaseModel, Field, model_validator, field_validator


class ModelInfo(BaseModel):
	"""Representation of a single model returned by the AI gateway.

	Older gateway versions did not include an explicit `id` field and used
	`name` as the identifier. For backward compatibility the `id` field is
	optional in the payload and is transparently defaulted to `name` when
	absent or null.

	Unknown fields are ignored to remain forward compatible.
	"""

	# Public type is always Optional[str] so downstream code can rely on a string.
	# Older gateways may emit numeric ids; we coerce them to str in a field validator.
	id: Optional[str] = Field(
		None,
		description="Stable unique identifier of the model (falls back to name if missing)",
	)

	@field_validator("id", mode="before")
	@classmethod
	def _coerce_id(cls, v):  # type: ignore[override]
		if isinstance(v, int):
			# The previous version of the gateway returned numeric IDs for the models
			# but that ID was only used internally and useless to clients
			# Instead the lookup happened by name
			return None
		return v
	name: str = Field(..., description="Display name of the model")
	mode: Optional[str] = Field(None, description="Operational mode (e.g. chat, embeddings)")
	# Public attribute should always be a string (never None) for backward compatibility
	# The new gateway version omits the description; we normalize to "".
	description: str = Field("", description="Human readable description (may be empty)")
	platform: Optional[str] = Field(None, description="Platform identifier of the model (e.g. OpenAI, Anthropic)")

	scope_id: Optional[str] = Field(None, description="Scope identifier of the model", alias="scopeId")

	@field_validator("description", mode="before")
	@classmethod
	def _coerce_description(cls, v):  # type: ignore[override]
		if v is None:
			return ""
		return v

	@model_validator(mode="after")
	def _default_id(cls, values):  # type: ignore[override]
		# Ensure id always populated for downstream consumers
		if values.id is None:
			values.id = values.name
		return values

	class Config:
		extra = "ignore"  # be robust against newly added fields


class ModelsResponse(BaseModel):
	"""Top-level response envelope from /models endpoint."""

	models: List[ModelInfo]

	class Config:
		extra = "ignore"


class TeamInfo(BaseModel):
	"""Team information from accounts service."""
	id: str
	name: str
	
	class Config:
		extra = "ignore"


class AccountIdentityResponse(BaseModel):
	"""Response from /accounts/identity endpoint."""
	teams: List[TeamInfo] = []
	
	class Config:
		extra = "ignore"


