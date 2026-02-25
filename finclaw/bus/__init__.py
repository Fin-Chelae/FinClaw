"""Message bus module for decoupled channel-agent communication."""

from finclaw.bus.events import InboundMessage, OutboundMessage
from finclaw.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]
