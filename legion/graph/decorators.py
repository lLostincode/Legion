from typing import Optional

from .graph import Graph, GraphConfig


class GraphDecorator:
    """Decorator for creating graph classes"""

    def __init__(
        self,
        name: Optional[str] = None,
        config: Optional[GraphConfig] = None,
        **kwargs
    ):
        self.name = name
        self.config = config or GraphConfig(**kwargs)

    def __call__(self, cls):
        # Extract name and description from class
        graph_name = self.name or cls.__name__
        graph_description = cls.__doc__ or f"Graph: {graph_name}"

        # Create graph instance
        graph = Graph(
            name=graph_name,
            description=graph_description,
            config=self.config
        )

        # Store original class attributes
        original_attrs = {
            name: value for name, value in cls.__dict__.items()
            if not name.startswith("__")
        }

        # Create wrapper class
        class GraphClass(cls):
            """Graph wrapper class"""

            def __init__(self, *args, **kwargs):
                # Initialize base class
                super().__init__(*args, **kwargs)

                # Store graph instance
                self.__graph = graph
                self.__original_attrs = original_attrs

                # Process class attributes
                self._process_attributes()

            def _process_attributes(self):
                """Process class attributes to set up graph structure"""
                # First, process parent class attributes if any
                parent_attrs = {}
                for base in reversed(self.__class__.__bases__):
                    if hasattr(base, "__dict__"):
                        parent_attrs.update({
                            name: value for name, value in base.__dict__.items()
                            if not name.startswith("__")
                        })

                # Process parent attributes first
                for name, value in parent_attrs.items():
                    if isinstance(value, GraphConfig):
                        config_dict = value.model_dump()
                        for key, val in config_dict.items():
                            setattr(self.__graph._config, key, val)

                # Then process class attributes, which can override parent attributes
                for name, value in self.__original_attrs.items():
                    # Skip already processed attributes
                    if hasattr(self, f"_{name}_processed"):
                        continue

                    # Handle different attribute types
                    if hasattr(value, "__node_decorator__"):
                        # Node definition - will be processed by node decorator
                        pass
                    elif hasattr(value, "__edge_decorator__"):
                        # Edge definition - will be processed by edge decorator
                        pass
                    elif hasattr(value, "__channel_decorator__"):
                        # Channel definition - will be processed by channel decorator
                        pass
                    elif isinstance(value, GraphConfig):
                        # Update graph configuration
                        config_dict = value.model_dump()
                        for key, val in config_dict.items():
                            setattr(self.__graph._config, key, val)

                    # Mark attribute as processed
                    setattr(self, f"_{name}_processed", True)

            @property
            def graph(self) -> Graph:
                """Get the underlying graph instance"""
                return self.__graph

        # Copy class attributes
        for attr in ["__module__", "__name__", "__qualname__", "__doc__", "__annotations__"]:
            try:
                setattr(GraphClass, attr, getattr(cls, attr))
            except AttributeError:
                pass

        return GraphClass

# Create decorator function
def graph(
    name: Optional[str] = None,
    config: Optional[GraphConfig] = None,
    **kwargs
):
    """Decorator for creating graph classes

    Args:
    ----
        name: Optional graph name
        config: Optional graph configuration
        **kwargs: Additional configuration parameters

    Example:
    -------
        @graph(name="analysis_workflow")
        class AnalysisWorkflow:
            '''Multi-stage analysis workflow.'''

            config = GraphConfig(
                execution_mode=ExecutionMode.PARALLEL,
                debug_mode=True
            )

    """
    if isinstance(name, type):
        # @graph used without parameters
        return GraphDecorator()(name)
    # @graph() used with parameters
    return GraphDecorator(name=name, config=config, **kwargs)
