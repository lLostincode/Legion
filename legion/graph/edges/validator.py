from dataclasses import dataclass
from typing import Dict, Optional

from ..nodes.base import NodeBase


@dataclass
class ValidationResult:
    """Result of edge validation"""

    is_valid: bool
    error: Optional[str] = None

class EdgeValidator:
    """Validator for edge connections"""

    def __init__(self):
        self._validation_cache: Dict[str, ValidationResult] = {}

    def validate_edge(
        self,
        source_node: NodeBase,
        target_node: NodeBase,
        source_channel: str,
        target_channel: str
    ) -> ValidationResult:
        """Validate edge connection

        Args:
        ----
            source_node: Source node
            target_node: Target node
            source_channel: Name of source output channel
            target_channel: Name of target input channel

        Returns:
        -------
            ValidationResult with validation status and error message if any

        """
        # Generate cache key
        cache_key = f"{source_node.node_id}:{source_channel}->{target_node.node_id}:{target_channel}"

        # Check cache
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]

        # Validate channels exist
        source_channel_obj = source_node.get_output_channel(source_channel)
        if not source_channel_obj:
            result = ValidationResult(
                is_valid=False,
                error=f"Source channel '{source_channel}' not found in node {source_node.node_id}"
            )
            self._validation_cache[cache_key] = result
            return result

        target_channel_obj = target_node.get_input_channel(target_channel)
        if not target_channel_obj:
            result = ValidationResult(
                is_valid=False,
                error=f"Target channel '{target_channel}' not found in node {target_node.node_id}"
            )
            self._validation_cache[cache_key] = result
            return result

        # Validate channel type compatibility
        if not isinstance(source_channel_obj, type(target_channel_obj)):
            result = ValidationResult(
                is_valid=False,
                error=f"Channel type mismatch: {type(source_channel_obj)} -> {type(target_channel_obj)}"
            )
            self._validation_cache[cache_key] = result
            return result

        # Validate value type compatibility
        source_type = getattr(source_channel_obj, "type_hint", None)
        target_type = getattr(target_channel_obj, "type_hint", None)

        if source_type and target_type and source_type != target_type:
            result = ValidationResult(
                is_valid=False,
                error=f"Channel value type mismatch: {source_type} -> {target_type}"
            )
            self._validation_cache[cache_key] = result
            return result

        # All validations passed
        result = ValidationResult(is_valid=True)
        self._validation_cache[cache_key] = result
        return result

    def invalidate_cache(
        self,
        node_id: Optional[str] = None,
        channel_name: Optional[str] = None
    ) -> None:
        """Invalidate validation cache

        Args:
        ----
            node_id: Optional node ID to invalidate cache for
            channel_name: Optional channel name to invalidate cache for

        """
        if not node_id and not channel_name:
            self._validation_cache.clear()
            return

        keys_to_remove = set()
        for key in self._validation_cache:
            if node_id and node_id in key:
                keys_to_remove.add(key)
            elif channel_name and channel_name in key:
                keys_to_remove.add(key)

        for key in keys_to_remove:
            del self._validation_cache[key]

    @staticmethod
    def check_cycle(
        source_node: NodeBase,
        target_node: NodeBase,
        get_node_dependencies: callable
    ) -> ValidationResult:
        """Check if adding an edge would create a cycle

        Args:
        ----
            source_node: Source node
            target_node: Target node
            get_node_dependencies: Callable that returns set of node IDs that a node depends on

        Returns:
        -------
            ValidationResult with validation status and error message if any

        """
        visited = set()
        path = set()

        def has_cycle(node_id: str) -> bool:
            if node_id in path:
                return True
            if node_id in visited:
                return False

            visited.add(node_id)
            path.add(node_id)

            # Check dependencies including the potential new edge
            dependencies = get_node_dependencies(node_id)
            if node_id == source_node.node_id:
                dependencies.add(target_node.node_id)

            for dep_id in dependencies:
                if has_cycle(dep_id):
                    return True

            path.remove(node_id)
            return False

        # Start cycle check from source node
        if has_cycle(source_node.node_id):
            return ValidationResult(
                is_valid=False,
                error="Adding edge would create cycle"
            )

        return ValidationResult(is_valid=True)
