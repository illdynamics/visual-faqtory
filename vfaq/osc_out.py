#!/usr/bin/env python3
"""
osc_out.py - Optional OSC Output for Visual FaQtory
═══════════════════════════════════════════════════════════════════════════════

This module provides a lightweight OSC client that can be used to broadcast
internal TURBO state to external systems such as TouchDesigner. It exposes a
simple API for sending the current macro state, crowd activity flag, and a
general energy level over UDP using the `python-osc` library. If the library
is not installed, the client will gracefully disable itself and log a warning.

Usage:

    from .osc_out import OSCClient

    client = OSCClient(host='127.0.0.1', port=6000, address='/visual_faqtory')
    client.send(macro='DROP', crowd_active=True, energy=0.42)

Part of QonQrete Visual FaQtory v0.3.5-beta
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    # Optional dependency: python-osc
    from pythonosc import udp_client  # type: ignore
except Exception:
    udp_client = None
    logger.debug("python-osc not installed; OSC output disabled")


class OSCClient:
    """Minimal OSC client for broadcasting TURBO state."""

    def __init__(self, host: str = "127.0.0.1", port: int = 6000, address: str = "/visual_faqtory"):
        """
        Initialise the OSC client.

        Args:
            host: Destination host (default 127.0.0.1).
            port: Destination port (default 6000).
            address: OSC address prefix (e.g. '/visual_faqtory').
        """
        self.host = host
        self.port = int(port)
        self.address = address
        if udp_client is None:
            self.client = None
            logger.warning("python-osc library not available; OSC output disabled")
        else:
            try:
                self.client = udp_client.SimpleUDPClient(self.host, self.port)
            except Exception as e:
                self.client = None
                logger.warning(f"Failed to initialise OSC client: {e}")

    def is_active(self) -> bool:
        """Return True if OSC is operational."""
        return self.client is not None

    def send(self, macro: str, crowd_active: bool, energy: float) -> None:
        """
        Send a single OSC message containing macro, crowd flag and energy.

        Args:
            macro: Current macro string (e.g. 'DROP', 'BUILD', 'CHILL').
            crowd_active: Whether a crowd prompt is currently active.
            energy: A float representing the current energy/intensity (0–1).
        """
        if not self.client:
            return
        try:
            # Compose payload as a list. The order and types are fixed.
            payload = [str(macro), int(bool(crowd_active)), float(energy)]
            self.client.send_message(self.address, payload)
        except Exception as e:
            # Do not raise exceptions on send failure
            logger.debug(f"OSC send error: {e}")
