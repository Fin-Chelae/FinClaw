"""Base channel interface for chat platforms."""

from abc import ABC, abstractmethod
from typing import Any
from time import time

from loguru import logger

from finclaw.bus.events import InboundMessage, OutboundMessage
from finclaw.bus.queue import MessageBus


class BaseChannel(ABC):
    """
    Abstract base class for chat channel implementations.
    
    Each channel (Telegram, Discord, etc.) should implement this interface
    to integrate with the finclaw message bus.
    """
    
    name: str = "base"
    _rate_limits: dict[str, list[float]] = {}
    
    def __init__(self, config: Any, bus: MessageBus):
        """
        Initialize the channel.
        
        Args:
            config: Channel-specific configuration.
            bus: The message bus for communication.
        """
        self.config = config
        self.bus = bus
        self._running = False
    
    @abstractmethod
    async def start(self) -> None:
        """
        Start the channel and begin listening for messages.
        
        This should be a long-running async task that:
        1. Connects to the chat platform
        2. Listens for incoming messages
        3. Forwards messages to the bus via _handle_message()
        """
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel and clean up resources."""
        pass
    
    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """
        Send a message through this channel.
        
        Args:
            msg: The message to send.
        """
        pass
    
    def is_allowed(self, sender_id: str) -> bool:
        """
        Check if a sender is allowed to use this bot.
        
        Args:
            sender_id: The sender's identifier.
        
        Returns:
            True if allowed, False otherwise.
        """
        allow_list = getattr(self.config, "allow_from", [])
        
        # Check for open_access (explicit opt-in to public access)
        if getattr(self.config, "open_access", False):
            return True
        
        # Default-deny: If no allow list, deny access
        if not allow_list:
            logger.warning(
                f"Channel {self.name}: no allow_from configured — denying access for {sender_id}. "
                "Set allow_from or open_access=True to allow."
            )
            return False
        
        sender_str = str(sender_id)
        if sender_str in allow_list:
            return True
        if "|" in sender_str:
            for part in sender_str.split("|"):
                if part and part in allow_list:
                    return True
        return False
    
    async def _handle_message(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: list[str] | None = None,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Handle an incoming message from the chat platform.
        
        This method checks permissions and forwards to the bus.
        
        Args:
            sender_id: The sender's identifier.
            chat_id: The chat/channel identifier.
            content: Message text content.
            media: Optional list of media URLs.
            metadata: Optional channel-specific metadata.
        """
        if not self.is_allowed(sender_id):
            logger.warning(
                f"Access denied for sender {sender_id} on channel {self.name}. "
                f"Add them to allowFrom list in config to grant access."
            )
            return
        
        # Rate limiting: check if sender has >20 messages in last 60 seconds
        sender_key = f"{self.name}:{sender_id}"
        now = time()
        
        # Clean up old entries lazily (older than 60 seconds)
        if sender_key in self._rate_limits:
            self._rate_limits[sender_key] = [
                ts for ts in self._rate_limits[sender_key] if now - ts < 60
            ]
        
        # Check rate limit
        timestamps = self._rate_limits.get(sender_key, [])
        if len(timestamps) >= 20:
            logger.warning(
                f"Rate limit exceeded for sender {sender_id} on channel {self.name}. "
                f"Dropping message."
            )
            return
        
        # Record this message
        timestamps.append(now)
        self._rate_limits[sender_key] = timestamps
        
        msg = InboundMessage(
            channel=self.name,
            sender_id=str(sender_id),
            chat_id=str(chat_id),
            content=content,
            media=media or [],
            metadata=metadata or {}
        )
        
        await self.bus.publish_inbound(msg)
    
    @property
    def is_running(self) -> bool:
        """Check if the channel is running."""
        return self._running
