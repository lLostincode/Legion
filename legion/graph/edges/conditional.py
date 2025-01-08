from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..nodes.base import NodeBase
from ..state import GraphState
from .base import EdgeBase
from .routing import RoutingCondition


@dataclass
class Target:
    """Target node and channel with optional condition"""

    node: NodeBase
    channel: str
    condition: Optional[RoutingCondition] = None
    priority: int = 0

class ConditionalEdge(EdgeBase):
    """Edge with conditional routing support"""

    def __init__(
        self,
        graph_state: GraphState,
        source_node: NodeBase,
        source_channel: str,
        default_target: Target,
        conditional_targets: Optional[List[Target]] = None
    ):
        """Initialize conditional edge

        Args:
        ----
            graph_state: Graph state manager
            source_node: Source node
            source_channel: Name of source output channel
            default_target: Default target if no conditions match
            conditional_targets: Optional list of conditional targets

        """
        super().__init__(
            graph_state,
            source_node,
            default_target.node,
            source_channel,
            default_target.channel
        )
        self._default_target = default_target
        self._conditional_targets = sorted(
            conditional_targets or [],
            key=lambda t: t.priority,
            reverse=True  # Higher priority first
        )

        # Validate all targets
        self._validate_target(default_target)
        for target in self._conditional_targets:
            self._validate_target(target)

    def _validate_target(self, target: Target) -> None:
        """Validate a target's channel compatibility"""
        source_channel = self._source_node.get_output_channel(self._source_channel)
        target_channel = target.node.get_input_channel(target.channel)

        if not source_channel or not target_channel:
            raise ValueError("Invalid channel configuration")

        # Check channel type compatibility
        if not isinstance(source_channel, type(target_channel)):
            raise TypeError(
                f"Channel type mismatch: {type(source_channel)} -> {type(target_channel)}"
            )

        # Check value type compatibility if available
        source_type = getattr(source_channel, "type_hint", None)
        target_type = getattr(target_channel, "type_hint", None)

        if source_type and target_type and source_type != target_type:
            raise TypeError(
                f"Channel value type mismatch: {source_type} -> {target_type}"
            )

    async def get_active_target(self) -> Target:
        """Get the currently active target based on conditions

        Returns
        -------
            Target that should receive the next value

        """
        # Check conditional targets in priority order
        for target in self._conditional_targets:
            if target.condition and await target.condition.evaluate(self._source_node):
                return target

        # Fall back to default target
        return self._default_target

    def checkpoint(self) -> Dict[str, Any]:
        """Create a checkpoint of current state"""
        checkpoint = super().checkpoint()
        checkpoint.update({
            "default_target": {
                "node": self._default_target.node.node_id,
                "channel": self._default_target.channel,
                "priority": self._default_target.priority
            },
            "conditional_targets": [
                {
                    "node": target.node.node_id,
                    "channel": target.channel,
                    "priority": target.priority,
                    "condition": target.condition.checkpoint() if target.condition else None
                }
                for target in self._conditional_targets
            ]
        })
        return checkpoint

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        """Restore from checkpoint"""
        super().restore(checkpoint)
        # Note: Full restoration happens through registry to reconnect nodes
