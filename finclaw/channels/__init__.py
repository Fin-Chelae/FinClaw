"""Chat channels module with plugin architecture."""

from finclaw.channels.base import BaseChannel
from finclaw.channels.manager import ChannelManager

__all__ = ["BaseChannel", "ChannelManager"]
