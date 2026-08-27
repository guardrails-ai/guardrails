"""
EPI Recorder integration for Guardrails AI.
Produces signed, tamper-evident .epi artifacts per Guard execution.
"""

from guardrails.integrations.epi_recorder.instrumentor import EPIInstrumentor

__all__ = ["EPIInstrumentor"]
